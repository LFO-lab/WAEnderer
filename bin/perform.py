#!/usr/bin/env python3
"""
Real-time grain-based corpus playback with OSC control.
Uses pre-rendered grains.
"""
import os
# Fix OpenMP duplicate library issue on macOS (must be set before any imports that use OpenMP)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import threading
import time
import numpy as np

from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus, find_grain_manifest
from stable_audio_wanderer.runtime.player import LatentNavigationEngine
from stable_audio_wanderer.runtime.grain_player import GrainPlayer, GrainScheduler
from stable_audio_wanderer.runtime.osc_server import run_server
from stable_audio_wanderer.policy.latent_geometry import load_geometry_from_dict


def find_corpus_files(corpus_dir: str):
    """Find corpus and grain manifest in directory."""
    try:
        corpus_npz = find_latest(corpus_dir, "*_corpus_*.npz")
    except FileNotFoundError:
        try:
            corpus_npz = find_latest(corpus_dir, "corpus.npz")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No corpus file found in {corpus_dir}. Expected 'corpus.npz'."
            )
    grain_manifest = find_grain_manifest(corpus_dir)
    return corpus_npz, grain_manifest


def load_navigation_engine(
    data: dict,
    policy_path: str = None,
    policy_temperature: float = 1.0,
    policy_sample: bool = True,
    control_width: float = 0.5,
    control_energy: float = 0.5,
    control_gravity: float = 0.5,
    control_memory: float = 0.0,
    control_coherence: float = 0.0,
    control_exploration: float = 0.0,
    grain_rate: float = 21.5,
    grain_jitter: float = 0.0,
):
    """Factory function to create the LatentNavigationEngine."""
    geometry = load_geometry_from_dict(data)
    if geometry is None:
        raise RuntimeError(
            "Corpus is missing latent geometry. Re-run preprocess.py to generate geometry-enabled corpora."
        )

    print("[info] Using LatentNavigationEngine (64D latent navigation)")
    GG = data["GG"].astype(np.float32)
    meta = data["meta"]

    return LatentNavigationEngine(
        GG=GG,
        meta=meta,
        geometry=geometry,
        policy_path=policy_path,
        policy_temperature=policy_temperature,
        policy_sample=policy_sample,
        control_width=control_width,
        control_energy=control_energy,
        control_gravity=control_gravity,
        control_memory=control_memory,
        control_coherence=control_coherence,
        control_exploration=control_exploration,
        grain_rate=grain_rate,
        grain_jitter=grain_jitter,
    )


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

    # Control parameters (6 dimensions)
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

    args = ap.parse_args()

    corpus_npz, grain_manifest_path = find_corpus_files(args.corpus_dir)
    print(f"[info] Using corpus: {corpus_npz}")

    if grain_manifest_path is None:
        raise FileNotFoundError(
            f"No grain manifest found in {args.corpus_dir}. "
            "Run preprocess.py with --render_grains to generate grains."
        )
    print(f"[info] Using grain manifest: {grain_manifest_path}")

    data = load_corpus(corpus_npz)

    nav = load_navigation_engine(
        data=data,
        policy_path=args.policy_path,
        policy_temperature=args.policy_temperature,
        policy_sample=bool(args.policy_sample),
        control_width=args.ctrl_width,
        control_energy=args.ctrl_energy,
        control_gravity=args.ctrl_gravity,
        control_memory=args.ctrl_memory,
        control_coherence=args.ctrl_coherence,
        control_exploration=args.ctrl_exploration,
        grain_rate=args.grain_rate,
        grain_jitter=args.grain_jitter,
    )

    player = GrainPlayer(
        manifest_path=grain_manifest_path,
        num_voices=args.voices,
        buffersize=args.buffersize,
    )

    player.set_pitch(args.grain_pitch)
    player.set_grain_dur(args.grain_dur)
    player.set_filter_freq(args.grain_filter)

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

    print("[info] Starting audio...")
    player.boot()
    player.start()
    scheduler.start()

    running = threading.Event()
    running.set()

    def navigation_loop():
        print("[info] Navigation loop started (mode=latent)")
        try:
            while running.is_set() and player.is_running:
                indices, weights, times, file_ids = nav.get_render_weights()
                scheduler.set_latent_render_data(indices, weights, times, file_ids)
                scheduler.set_coherence(nav.ctrl_coherence)

                base_interval = nav.get_trigger_interval()
                interval = scheduler.get_nav_interval(base_interval)
                time.sleep(interval)
        except Exception as e:
            print(f"[error] Navigation loop exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[info] Navigation loop stopped")

    nav_thread = threading.Thread(target=navigation_loop, daemon=True)
    nav_thread.start()

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
    print("[info] Navigation mode: latent (64D)")
    print(
        f"[info] Granular: dur={args.grain_dur*1000:.0f}ms, overlap={args.grain_overlap*100:.0f}%, "
        f"rate={scheduler.total_grain_rate:.0f} grains/sec, nav_speed={args.nav_speed:.2f}x"
    )
    print(
        f"[info] Controls: width={args.ctrl_width}, energy={args.ctrl_energy}, "
        f"gravity={args.ctrl_gravity}, memory={args.ctrl_memory}"
    )
    print(
        f"[info] Advanced: coherence={args.ctrl_coherence}, exploration={args.ctrl_exploration}"
    )

    try:
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
