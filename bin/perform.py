#!/usr/bin/env python3
"""
Real-time manifold-constrained decoding with OSC control.
"""
import os
# Fix OpenMP duplicate library issue on macOS (must be set before any imports that use OpenMP)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
import queue
import threading
import time
import numpy as np

from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus
from stable_audio_wanderer.runtime.player import LatentNavigationEngine
from stable_audio_wanderer.runtime.manifold import ManifoldConstrainedGenerator, ManifoldConfig
from stable_audio_wanderer.runtime.decoder_player import DecoderPlayer
from stable_audio_wanderer.runtime.osc_server import run_server
from stable_audio_wanderer.policy.latent_geometry import load_geometry_from_dict
from stable_audio_wanderer.vae.sae import load_vae
from stable_audio_wanderer.vae.decoder import decode_latents


def find_corpus_file(corpus_dir: str):
    """Find corpus in directory."""
    try:
        return find_latest(corpus_dir, "*_corpus_*.npz")
    except FileNotFoundError:
        corpus_npz = os.path.join(corpus_dir, "corpus.npz")
        if not os.path.exists(corpus_npz):
            raise FileNotFoundError(
                f"No corpus file found in {corpus_dir}. Expected 'corpus.npz'."
            )
        return corpus_npz


def smooth_window_log2(current_log2: float, target_log2: float, alpha: float = 0.3) -> float:
    """Smooth window size transitions in log2 space (perceptually uniform)."""
    return (1.0 - alpha) * current_log2 + alpha * target_log2


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
    )


def main():
    ap = argparse.ArgumentParser(
        description="Real-time manifold-constrained decoding (OSC-controlled)."
    )
    ap.add_argument("--corpus_dir", required=True, help="Directory containing corpus.npz")
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--osc_ip", default="127.0.0.1")
    ap.add_argument("--osc_port", type=int, default=9000)
    ap.add_argument("--ws_port", type=int, default=8765,
                    help="WebSocket port for visualization (0 to disable).")

    # Decoder controls
    ap.add_argument("--output_gain", type=float, default=1.0,
                    help="Initial output gain (0-2).")
    ap.add_argument("--smoothing", type=float, default=0.1,
                    help="Crossfade smoothing between frames (0-1).")

    # Manifold parameters
    ap.add_argument("--manifold_k", type=int, default=16)
    ap.add_argument("--manifold_n_local", type=int, default=8)
    ap.add_argument("--manifold_n_global", type=int, default=32)
    ap.add_argument("--manifold_sparse_quantile", type=float, default=0.75)

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

    # Window size for batched decoding
    ap.add_argument("--window_size", type=int, default=2,
                    help="Number of latent frames to decode together (2-64). "
                         "Higher values increase audio coherence but add latency.")
    ap.add_argument("--adaptive_window", action="store_true",
                    help="Enable adaptive window sizing based on policy prediction. "
                         "Overrides --window_size with dynamic per-frame predictions.")

    args = ap.parse_args()

    corpus_npz = find_corpus_file(args.corpus_dir)
    print(f"[info] Using corpus: {corpus_npz}")

    data = load_corpus(corpus_npz)
    geometry = load_geometry_from_dict(data)
    if geometry is None:
        raise RuntimeError("Corpus missing latent geometry. Re-run preprocess.py.")

    Z_mean = data["Z_mean"].astype(np.float32)
    Z_std = data["Z_std"].astype(np.float32)

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
    )

    manifold_cfg = ManifoldConfig(
        k=args.manifold_k,
        n_local=args.manifold_n_local,
        n_global=args.manifold_n_global,
        sparse_quantile=args.manifold_sparse_quantile,
    )
    manifold = ManifoldConstrainedGenerator(nav.GG, geometry, manifold_cfg)

    print("[info] Loading VAE decoder...")
    vae = load_vae(args.pretrained)

    decoder = DecoderPlayer(
        gain=args.output_gain,
        smoothing=args.smoothing,
        adaptive_crossfade=args.adaptive_window,
    )

    # Queue for latent batches (navigation -> decode)
    latent_queue = queue.Queue(maxsize=4)  # Buffer up to 4 windows ahead

    running = threading.Event()
    running.set()

    adaptive_mode = args.adaptive_window
    window_size = max(2, min(args.window_size, 64))
    hop_size = (window_size + 1) // 2  # 50% overlap (rounded up)

    if adaptive_mode:
        print("[info] Using adaptive window sizing (policy-predicted)")
    elif window_size > 1:
        print(f"[info] Using window_size={window_size}, hop_size={hop_size} (overlap-add mode)")

    # Pre-buffer: fill audio buffer with ~1 second of audio BEFORE starting stream
    # With overlap-add, each hop outputs hop_size frames worth of audio
    print("[info] Pre-buffering audio (overlap-add)...")
    audio_per_hop = hop_size * 0.0465  # seconds per hop
    num_hops = max(4, int(1.0 / audio_per_hop) + 1)

    # First window: no prev_half yet, decode full window_size frames
    frame_buffer = [nav.step() for _ in range(window_size)]
    z_batch_norm = manifold.generate_batch(frame_buffer, exploration=nav.ctrl_exploration)
    z_batch_raw = z_batch_norm * Z_std + Z_mean
    audio = decode_latents(vae, z_batch_raw)
    decoder.write_frame(audio)
    prev_half = frame_buffer[hop_size:]  # Save second half for overlap

    # Subsequent windows with overlap
    for _ in range(num_hops - 1):
        new_frames = [nav.step() for _ in range(hop_size)]
        full_window = prev_half + new_frames
        z_batch_norm = manifold.generate_batch(full_window, exploration=nav.ctrl_exploration)
        z_batch_raw = z_batch_norm * Z_std + Z_mean
        audio = decode_latents(vae, z_batch_raw)
        decoder.write_frame(audio)
        prev_half = full_window[hop_size:]

    print(f"[info] Pre-buffered {num_hops} windows ({decoder.buffer_duration():.2f}s)")

    # Pre-fill the latent queue so decode thread has work immediately
    print("[info] Pre-filling latent queue...")
    for _ in range(latent_queue.maxsize):
        new_frames = [nav.step() for _ in range(hop_size)]
        full_window = prev_half + new_frames
        z_batch_norm = manifold.generate_batch(full_window, exploration=nav.ctrl_exploration)
        z_batch_raw = z_batch_norm * Z_std + Z_mean
        latent_queue.put(z_batch_raw)
        prev_half = full_window[hop_size:]
    print(f"[info] Latent queue filled ({latent_queue.qsize()} batches)")

    # Store prev_half in a mutable container for nav_loop access
    nav_state = {
        "prev_half": prev_half,
        "current_window_log2": math.log2(max(2, window_size)),
        "adaptive_mode": adaptive_mode,
    }

    def nav_loop():
        """Navigation thread with overlap context (fixed or adaptive)."""
        mode_str = "adaptive" if nav_state["adaptive_mode"] else "overlap-add"
        print(f"[info] Navigation loop started ({mode_str} mode)")
        frame_buffer = []
        prev_half = nav_state["prev_half"]
        current_window_log2 = nav_state["current_window_log2"]

        try:
            while running.is_set():
                # Get current window/hop sizes
                if nav_state["adaptive_mode"]:
                    current_window = max(2, min(64, round(2 ** current_window_log2)))
                    current_hop = (current_window + 1) // 2
                else:
                    current_window = window_size
                    current_hop = hop_size

                # Accumulate new frames
                frame = nav.step()
                frame_buffer.append(frame)

                # Update smoothed window size from policy prediction (in log2 space)
                if nav_state["adaptive_mode"]:
                    target_log2 = math.log2(max(2, frame.predicted_window_size))
                    current_window_log2 = smooth_window_log2(current_window_log2, target_log2)

                # Need current_hop new frames to complete next window
                if len(frame_buffer) >= current_hop:
                    # Adjust prev_half to match current window requirements
                    target_prev_len = current_window - current_hop
                    if len(prev_half) < target_prev_len:
                        while len(prev_half) < target_prev_len:
                            prev_half.append(prev_half[-1] if prev_half else frame_buffer[0])
                    elif len(prev_half) > target_prev_len:
                        prev_half = prev_half[-target_prev_len:]

                    # Combine: prev_half + new frames = full window
                    full_window = prev_half + frame_buffer[:current_hop]

                    # Batch manifold constraint (fast: ~2-5ms)
                    z_batch_norm = manifold.generate_batch(
                        full_window,
                        exploration=nav.ctrl_exploration
                    )
                    z_batch_raw = z_batch_norm * Z_std + Z_mean  # [N, 64]

                    # Queue for decode thread - blocks when queue is full (backpressure)
                    try:
                        latent_queue.put(z_batch_raw, timeout=1.0)
                    except queue.Full:
                        pass  # Drop if decode can't keep up

                    # Second half becomes first half of next window
                    prev_half = full_window[current_hop:]
                    frame_buffer = frame_buffer[current_hop:]
        except Exception as e:
            print(f"[error] Navigation loop exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[info] Navigation loop stopped")

    # Target buffer duration: keep 0.5-1.0 seconds buffered
    TARGET_BUFFER_LOW = 0.3   # Start decoding when buffer drops below this
    TARGET_BUFFER_HIGH = 1.0  # Slow down when buffer exceeds this

    def decode_loop():
        """Decode thread: VAE decode to audio, feeds the dual-buffer."""
        print("[info] Decode loop started")
        frame_duration = 0.0465  # Will be updated after first decode

        try:
            while running.is_set():
                # Check buffer level and pace accordingly
                buf_dur = decoder.buffer_duration()

                if buf_dur > TARGET_BUFFER_HIGH:
                    # Buffer is healthy - sleep a bit to avoid overproduction
                    time.sleep(0.05)
                    continue

                # Get next latent batch
                try:
                    z_batch = latent_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # VAE decode (~12ms per frame in batch, ~48ms for window_size=4)
                audio = decode_latents(vae, z_batch)

                # Write to dual-buffer (fast, lock-free append)
                decoder.write_frame(audio)

                # Update frame duration if available
                if decoder.frame_duration:
                    frame_duration = decoder.frame_duration
        except Exception as e:
            print(f"[error] Decode loop exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[info] Decode loop stopped")

    # Now start the audio stream (after buffer is pre-filled)
    decoder.start()

    # Start threads
    nav_thread = threading.Thread(target=nav_loop, daemon=True)
    decode_thread = threading.Thread(target=decode_loop, daemon=True)
    nav_thread.start()
    decode_thread.start()

    ws_server = None
    ws_thread = None
    if args.ws_port > 0:
        try:
            from stable_audio_wanderer.runtime.ws_server import start_ws_server
            ws_server, ws_thread = start_ws_server(nav, decoder, port=args.ws_port)
            print(f"[info] WebSocket server running on ws://127.0.0.1:{args.ws_port}")
        except Exception as e:
            print(f"[warn] Failed to start WebSocket server: {e}")

    print(f"[info] Running with {nav.N} segments")
    print("[info] Navigation mode: latent (64D)")
    print(
        f"[info] Controls: width={args.ctrl_width}, energy={args.ctrl_energy}, "
        f"gravity={args.ctrl_gravity}, memory={args.ctrl_memory}"
    )
    print(
        f"[info] Advanced: coherence={args.ctrl_coherence}, exploration={args.ctrl_exploration}"
    )
    if args.adaptive_window:
        print("[info] Window size: adaptive (policy-predicted, 2-64 frames)")
    elif args.window_size > 1:
        print(f"[info] Window size: {args.window_size} frames (batched decoding)")

    try:
        run_server(nav, decoder, ip=args.osc_ip, port=args.osc_port)
    except KeyboardInterrupt:
        print("\n[info] Shutting down...")
    finally:
        running.clear()
        nav_thread.join(timeout=1.0)
        decode_thread.join(timeout=1.0)
        decoder.stop()
        decoder.close()
        if ws_server is not None:
            ws_server.shutdown()

    print("[info] Done.")


if __name__ == "__main__":
    main()
