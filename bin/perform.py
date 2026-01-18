#!/usr/bin/env python3
"""
Real-time grain-based corpus playback with OSC control.
Uses pre-rendered grains (no VAE decoding at runtime).
"""
import os
import argparse
import threading
import time
import numpy as np

from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus, find_grain_manifest, load_grain_manifest
from stable_audio_wanderer.runtime.player import NavigationEngine
from stable_audio_wanderer.runtime.grain_player import GrainPlayer, GrainScheduler
from stable_audio_wanderer.runtime.osc_server import run_server


def find_corpus_files(corpus_dir: str):
    """Find corpus and grain manifest in directory."""
    corpus_npz = find_latest(corpus_dir, "*_corpus_*.npz")
    grain_manifest = find_grain_manifest(corpus_dir)
    return corpus_npz, grain_manifest


def main():
    ap = argparse.ArgumentParser(
        description="Real-time grain-based corpus playback (OSC-controlled, no VAE)."
    )
    ap.add_argument("--corpus_dir", required=True, help="Directory containing corpus & grains")
    ap.add_argument("--osc_ip", default="127.0.0.1")
    ap.add_argument("--osc_port", type=int, default=9000)
    ap.add_argument("--ws_port", type=int, default=8765,
                    help="WebSocket port for visualization (0 to disable).")
    
    # Grain playback (basic)
    ap.add_argument("--grain_rate", type=float, default=21.5,
                    help="Initial grain trigger rate (min 21.5 Hz = latent rate).")
    ap.add_argument("--grain_jitter", type=float, default=0.0,
                    help="Grain timing jitter (0-1).")
    ap.add_argument("--voices", type=int, default=64,
                    help="Number of overlapping grain voices.")
    ap.add_argument("--buffersize", type=int, default=512,
                    help="Audio buffer size.")
    
    # Grain playback (synthesis)
    ap.add_argument("--grain_pitch", type=float, default=1.0,
                    help="Initial pitch ratio (0.25-4.0).")
    ap.add_argument("--grain_dur", type=float, default=0.04,
                    help="Grain duration in seconds (default 40ms for resynthesis).")
    ap.add_argument("--grain_filter", type=float, default=20000,
                    help="Initial filter cutoff frequency (20-20000 Hz).")

    # Multi-stream granular synthesis (resynthesis quality)
    ap.add_argument("--num_streams", type=int, default=4,
                    help="Number of concurrent grain streams (more = smoother).")
    ap.add_argument("--grain_overlap", type=float, default=0.75,
                    help="Grain overlap ratio (0-0.95). Higher = smoother time-stretch.")
    ap.add_argument("--position_jitter", type=float, default=0.1,
                    help="Position jitter (0-1). Small values for natural sound.")
    ap.add_argument("--dur_jitter", type=float, default=0.05,
                    help="Duration jitter (0-1). Small values avoid artifacts.")
    ap.add_argument("--rate_jitter", type=float, default=0.02,
                    help="Trigger rate jitter (0-1). Very small for stability.")
    ap.add_argument("--stereo_spread", type=float, default=0.3,
                    help="Stereo spread of grain streams (0-1).")
    ap.add_argument("--nav_speed", type=float, default=1.0,
                    help="Navigation speed (1.0 = normal, 0.5 = half speed/time-stretch, 2.0 = double).")
    
    # Policy controls
    ap.add_argument("--policy_path", default=None,
                    help="Checkpoint .pt for navigation policy.")
    ap.add_argument("--policy_temperature", type=float, default=1.0,
                    help="Base temperature for policy sampling.")
    ap.add_argument("--policy_sample", action=argparse.BooleanOptionalAction, default=True,
                    help="Stochastically sample from policy (default: yes).")
    
    # Control parameters (7 dimensions)
    ap.add_argument("--ctrl_width", type=float, default=0.5,
                    help="Initial width control (0-1): temperature scaling.")
    ap.add_argument("--ctrl_energy", type=float, default=0.5,
                    help="Initial energy control (0-1): displacement magnitude.")
    ap.add_argument("--ctrl_gravity", type=float, default=0.5,
                    help="Initial gravity control (0-1): forward/backward bias.")
    ap.add_argument("--ctrl_memory", type=float, default=0.0,
                    help="Initial memory control (0-1): pull toward recent positions.")
    ap.add_argument("--ctrl_coherence", type=float, default=0.0,
                    help="Initial coherence control (0-1): stay within same file.")
    ap.add_argument("--ctrl_exploration", type=float, default=0.0,
                    help="Initial exploration control (0-1): entropy injection.")
    ap.add_argument("--ctrl_regime_bias", type=float, default=0.0,
                    help="Initial regime bias (0-2): 0=drift, 1=turn, 2=linger.")
    
    args = ap.parse_args()
    
    # Find corpus files
    corpus_npz, grain_manifest_path = find_corpus_files(args.corpus_dir)
    print(f"[info] Using corpus: {corpus_npz}")
    
    if grain_manifest_path is None:
        raise FileNotFoundError(
            f"No grain manifest found in {args.corpus_dir}. "
            "Run preprocess.py with --render_grains to generate grains."
        )
    print(f"[info] Using grain manifest: {grain_manifest_path}")
    
    # Load corpus data
    data = load_corpus(corpus_npz)
    ZZ = data["ZZ"].astype(np.float32)
    meta = data["meta"]
    
    # Create navigation engine with all control parameters
    nav = NavigationEngine(
        ZZ=ZZ,
        meta=meta,
        policy_path=args.policy_path,
        policy_temperature=args.policy_temperature,
        policy_sample=bool(args.policy_sample),
        control_width=args.ctrl_width,
        control_energy=args.ctrl_energy,
        control_gravity=args.ctrl_gravity,
        control_memory=args.ctrl_memory,
        control_coherence=args.ctrl_coherence,
        control_exploration=args.ctrl_exploration,
        control_regime_bias=args.ctrl_regime_bias,
        grain_rate=args.grain_rate,
        grain_jitter=args.grain_jitter,
    )
    
    # Create grain player
    player = GrainPlayer(
        manifest_path=grain_manifest_path,
        num_voices=args.voices,
        buffersize=args.buffersize,
    )

    # Sync initial grain parameters to player
    player.set_pitch(args.grain_pitch)
    player.set_grain_dur(args.grain_dur)
    player.set_filter_freq(args.grain_filter)

    # Create multi-stream grain scheduler for resynthesis-quality playback
    scheduler = GrainScheduler(
        grain_player=player,
        num_streams=args.num_streams,
        grain_dur=args.grain_dur,
        overlap=args.grain_overlap,
        position_jitter=args.position_jitter,
        dur_jitter=args.dur_jitter,
        rate_jitter=args.rate_jitter,
        stereo_spread=args.stereo_spread,
        nav_speed=args.nav_speed,
    )

    # Start audio
    print("[info] Starting audio...")
    player.boot()
    player.start()
    scheduler.start()

    # Navigation loop: picks next segment and updates scheduler playhead
    # The scheduler's internal thread handles grain triggering at high rate
    running = threading.Event()
    running.set()

    def navigation_loop():
        """Navigation loop that updates the playhead for the grain scheduler."""
        print("[info] Navigation loop started")
        try:
            while running.is_set() and player.is_running:
                # Pick next segment from navigation policy
                segment_idx = nav.pick_next_index()

                # Update scheduler playhead (scheduler handles grain triggering)
                scheduler.set_playhead(segment_idx)

                # Wait for next navigation step, adjusted by nav_speed
                # nav_speed > 1 = faster movement, < 1 = slower (time-stretch)
                base_interval = nav.get_trigger_interval()
                interval = scheduler.get_nav_interval(base_interval)
                time.sleep(interval)
        except Exception as e:
            print(f"[error] Navigation loop exception: {e}")
        finally:
            print("[info] Navigation loop stopped")

    nav_thread = threading.Thread(target=navigation_loop, daemon=True)
    nav_thread.start()
    
    # Start WebSocket server for visualization (if enabled)
    ws_server = None
    ws_thread = None
    if args.ws_port > 0:
        try:
            from stable_audio_wanderer.runtime.ws_server import start_ws_server
            ws_server, ws_thread = start_ws_server(nav, player, scheduler, port=args.ws_port)
            print(f"[info] WebSocket server running on ws://127.0.0.1:{args.ws_port}")
        except ImportError:
            print("[warn] WebSocket server not available (missing dependencies)")
        except Exception as e:
            print(f"[warn] Failed to start WebSocket server: {e}")
    
    print(f"[info] Running with {nav.N} segments, {args.voices} voices, {args.num_streams} streams")
    print(f"[info] Granular: dur={args.grain_dur*1000:.0f}ms, overlap={args.grain_overlap*100:.0f}%, "
          f"rate={scheduler.total_grain_rate:.0f} grains/sec, nav_speed={args.nav_speed:.2f}x")
    print(f"[info] Controls: width={args.ctrl_width}, energy={args.ctrl_energy}, "
          f"gravity={args.ctrl_gravity}, memory={args.ctrl_memory}")
    print(f"[info] Advanced: coherence={args.ctrl_coherence}, exploration={args.ctrl_exploration}, "
          f"regime_bias={args.ctrl_regime_bias}")

    try:
        # Run OSC server (blocks until interrupted)
        run_server(nav, player, scheduler, ip=args.osc_ip, port=args.osc_port)
    except KeyboardInterrupt:
        print("\n[info] Shutting down...")
    finally:
        running.clear()
        scheduler.stop()
        player.shutdown()
        nav_thread.join(timeout=1.0)
        if ws_server is not None:
            ws_server.shutdown()

    print("[info] Done.")


if __name__ == "__main__":
    main()
