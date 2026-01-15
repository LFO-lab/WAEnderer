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
from stable_audio_wanderer.runtime.grain_player import GrainPlayer
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
    
    # Grain playback
    ap.add_argument("--grain_rate", type=float, default=10.0,
                    help="Initial grain trigger rate (grains/sec).")
    ap.add_argument("--grain_jitter", type=float, default=0.0,
                    help="Grain timing jitter (0-1).")
    ap.add_argument("--voices", type=int, default=4,
                    help="Number of overlapping grain voices.")
    ap.add_argument("--buffersize", type=int, default=512,
                    help="Audio buffer size.")
    
    # Policy controls
    ap.add_argument("--policy_path", default=None,
                    help="Checkpoint .pt for navigation policy.")
    ap.add_argument("--policy_temperature", type=float, default=1.0,
                    help="Base temperature for policy sampling.")
    ap.add_argument("--policy_sample", action=argparse.BooleanOptionalAction, default=True,
                    help="Stochastically sample from policy (default: yes).")
    ap.add_argument("--ctrl_width", type=float, default=0.5,
                    help="Initial width control (0-1).")
    ap.add_argument("--ctrl_energy", type=float, default=0.5,
                    help="Initial energy control (0-1).")
    ap.add_argument("--ctrl_gravity", type=float, default=0.5,
                    help="Initial gravity control (0-1).")
    ap.add_argument("--ctrl_memory", type=float, default=0.0,
                    help="Initial memory control (0-1).")
    
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
    
    # Create navigation engine
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
        grain_rate=args.grain_rate,
        grain_jitter=args.grain_jitter,
    )
    
    # Create grain player
    player = GrainPlayer(
        manifest_path=grain_manifest_path,
        num_voices=args.voices,
        buffersize=args.buffersize,
    )
    
    # Sync initial grain rate
    player.set_trigger_rate(args.grain_rate)
    player.set_trigger_jitter(args.grain_jitter)
    
    # Start audio
    print("[info] Starting audio...")
    player.boot()
    player.start()
    
    # Audio loop: navigation -> grain triggering
    running = threading.Event()
    running.set()
    
    def audio_loop():
        print("[info] Audio loop started")
        try:
            while running.is_set() and player.is_running:
                # Pick next segment from navigation
                segment_idx = nav.pick_next_index()
                
                # Trigger grain for this segment
                player.trigger_grain(segment_idx, amp=1.0, pan=0.5)
                
                # Wait for next trigger
                interval = nav.get_trigger_interval()
                time.sleep(interval)
        except Exception as e:
            print(f"[error] Audio loop exception: {e}")
        finally:
            print("[info] Audio loop stopped")
    
    audio_thread = threading.Thread(target=audio_loop, daemon=True)
    audio_thread.start()
    
    print(f"[info] Running with {nav.N} segments, {args.voices} voices")
    
    try:
        # Run OSC server (blocks until interrupted)
        run_server(nav, player, ip=args.osc_ip, port=args.osc_port)
    except KeyboardInterrupt:
        print("\n[info] Shutting down...")
    finally:
        running.clear()
        player.shutdown()
        audio_thread.join(timeout=1.0)
    
    print("[info] Done.")


if __name__ == "__main__":
    main()
