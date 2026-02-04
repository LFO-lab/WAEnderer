#!/usr/bin/env python3
"""
Real-time manifold-constrained decoding with OSC control.
"""
import os
# Fix OpenMP duplicate library issue on macOS (must be set before any imports that use OpenMP)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
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

    decoder = DecoderPlayer(gain=args.output_gain, smoothing=args.smoothing)
    decoder.start()

    running = threading.Event()
    running.set()

    def decode_loop():
        print("[info] Decode loop started")
        try:
            while running.is_set():
                t0 = time.perf_counter()
                frame = nav.step()
                z_decode_norm = manifold.generate(frame, exploration=nav.ctrl_exploration)
                z_decode_raw = z_decode_norm * Z_std + Z_mean
                audio = decode_latents(vae, z_decode_raw)
                decoder.write_frame(audio)

                if decoder.frame_duration is not None:
                    elapsed = time.perf_counter() - t0
                    sleep_time = decoder.frame_duration - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except Exception as e:
            print(f"[error] Decode loop exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[info] Decode loop stopped")

    decode_thread = threading.Thread(target=decode_loop, daemon=True)
    decode_thread.start()

    ws_server = None
    ws_thread = None
    if args.ws_port > 0:
        try:
            from stable_audio_wanderer.runtime.ws_server import start_ws_server
            ws_server, ws_thread = start_ws_server(nav, decoder, port=args.ws_port)
            print(f"[info] WebSocket server running on ws://127.0.0.1:{args.ws_port}")
        except ImportError:
            print("[warn] WebSocket server not available (missing dependencies)")
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

    try:
        run_server(nav, decoder, ip=args.osc_ip, port=args.osc_port)
    except KeyboardInterrupt:
        print("\n[info] Shutting down...")
    finally:
        running.clear()
        decode_thread.join(timeout=1.0)
        decoder.stop()
        decoder.close()
        if ws_server is not None:
            ws_server.shutdown()

    print("[info] Done.")


if __name__ == "__main__":
    main()
