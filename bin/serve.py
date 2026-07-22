#!/usr/bin/env python3
"""
Unified GUI pipeline server for Stable Audio Wanderer.
Serves web UI over HTTP and manages preprocess/train/perform phases over WebSocket.

Usage:
    python bin/serve.py
    python bin/serve.py --pretrained stabilityai/stable-audio-open-1.0
"""
import argparse
import functools
import http.server
import os
import signal
import socketserver
import threading

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


def _start_http_server(web_dir: str, port: int) -> threading.Thread:
    """Start a simple HTTP server for static files in a daemon thread."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=web_dir
    )
    # Allow address reuse for quick restart
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", port), handler)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="http-server")
    thread.start()
    return thread


def _setup_perform_phase(corpus_dir, latent_decoder, config, broadcaster, ws_port):
    """Set up the fail-closed SAME-S ONNX Web performance phase."""
    from bin.perform import (
        load_manual_artifact,
        load_navigation_engine,
        load_policy_v2_artifact,
        _resolve_manual_artifact_path,
        _resolve_optional_model_path,
        _resolve_reorganized_units_path,
    )
    from stable_audio_wanderer.io.corpus_io import load_corpus
    from stable_audio_wanderer.policy.latent_geometry import load_geometry_from_dict
    from stable_audio_wanderer.runtime.decoder_player import DecoderPlayer
    from stable_audio_wanderer.runtime.manifold import ManifoldConfig, ManifoldConstrainedGenerator
    from stable_audio_wanderer.runtime.manual_player import ManualNavigationEngine
    from stable_audio_wanderer.runtime.onnx_transport import OnnxTransportController
    from stable_audio_wanderer.io.corpus_io import read_scalar

    corpus_npz = os.path.join(corpus_dir, "corpus.npz")
    if not os.path.isfile(corpus_npz):
        raise FileNotFoundError(f"Corpus file is missing: {corpus_npz}")
    print(f"[serve] Using corpus: {corpus_npz}")
    data = load_corpus(corpus_npz)

    Z_concat = data["Z_concat"].astype(np.float32)
    Z_mean = data["Z_mean"].astype(np.float32)
    Z_std = data["Z_std"].astype(np.float32)
    file_offsets = data["file_offsets"].astype(np.int64)

    # Read VAE params from corpus
    corpus_vae_id = str(read_scalar(data, "vae_id", ""))
    corpus_sr = int(read_scalar(data, "sr", 44100))
    corpus_latent_hz = float(read_scalar(data, "latent_hz", 21.5))
    latent_frame_sec = 1.0 / corpus_latent_hz

    manual_artifact_path = _resolve_manual_artifact_path(corpus_dir, None)
    manual_data = load_manual_artifact(manual_artifact_path, expected_frames=Z_concat.shape[0])
    print(f"[serve] Manual artifact: {manual_artifact_path}")

    random_model_path = _resolve_optional_model_path(corpus_dir, None, "latent_policy_*.pt")
    reorganized_units_path = _resolve_reorganized_units_path(corpus_dir, None)
    v2_data = None
    if reorganized_units_path is not None:
        v2_data = load_policy_v2_artifact(reorganized_units_path, expected_frames=Z_concat.shape[0])
    reorganized_model_path = _resolve_optional_model_path(corpus_dir, None, "policy_v2_*.pt")

    nav = load_navigation_engine(
        data=data,
        random_model_path=random_model_path,
        desc_weighted=manual_data["manual_desc_weighted"],
        reorganized_enabled=bool(v2_data is not None),
        reorganized_artifact=v2_data,
        reorganized_model_path=reorganized_model_path,
        latent_frame_seconds=latent_frame_sec,
    )

    geometry = load_geometry_from_dict(data)
    manifold_cfg = ManifoldConfig(k=16, n_local=8, n_global=32, sparse_quantile=0.75)
    manifold = ManifoldConstrainedGenerator(nav.GG, geometry, manifold_cfg)

    manual_engine = ManualNavigationEngine(
        manual_points=manual_data["manual_embed_points"],
        fader_p01=manual_data["manual_fader_p01"],
        fader_p99=manual_data["manual_fader_p99"],
        desc_weighted=manual_data["manual_desc_weighted"],
        pca_components=manual_data["manual_pca_components"],
        pca_mean=manual_data["manual_pca_mean"],
        reducer=str(manual_data["manual_embed_reducer"]),
        leafsize=int(manual_data["kdtree_leafsize"]),
    )

    if corpus_vae_id != "same_s":
        raise RuntimeError(
            f"Unified Web performance requires a SAME-S corpus, got {corpus_vae_id!r}"
        )
    if latent_decoder is None:
        raise RuntimeError("App-owned SAME-S ONNX decoder is required")
    selected_window = int(config.get("decoder_window", latent_decoder.default_window))
    audio_player = DecoderPlayer(gain=1.0, sr=corpus_sr)

    try:
        controller = OnnxTransportController(
            nav=nav,
            manual_engine=manual_engine,
            manifold=manifold,
            player=audio_player,
            latent_decoder=latent_decoder,
            z_concat=Z_concat,
            file_offsets=file_offsets,
            frame_file_ids=manual_data["frame_file_ids"],
            z_mean=Z_mean,
            z_std=Z_std,
            initial_mode="random",
            initial_window=selected_window,
        )
    except Exception:
        audio_player.close()
        raise

    # Bind to the existing broadcaster
    try:
        broadcaster.bind_nav_decoder(
            nav=nav,
            decoder=audio_player,
            message_handler=controller.handle_ws_message,
            extra_state_provider=controller.get_extra_state,
            manual_points_3d=manual_data["manual_embed_points"],
            manual_file_ids=manual_data["frame_file_ids"],
            manual_fader_p01=manual_data["manual_fader_p01"],
            manual_fader_p99=manual_data["manual_fader_p99"],
        )
    except Exception:
        controller.close()
        raise

    print("[serve] Perform phase ready. Use the web UI to start transport.")
    return controller


def main():
    ap = argparse.ArgumentParser(
        description="Unified GUI pipeline server for Stable Audio Wanderer."
    )
    ap.add_argument("--http_port", type=int, default=8080, help="HTTP port for web UI.")
    ap.add_argument("--port", type=int, default=8765, help="WebSocket port.")
    ap.add_argument(
        "--pretrained",
        default="stabilityai/stable-audio-open-1.0",
        help="HuggingFace model ID for the VAE.",
    )
    args = ap.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    web_dir = os.path.join(project_root, "web")
    if not os.path.isdir(web_dir):
        raise FileNotFoundError(f"Web directory not found: {web_dir}")

    from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
    from stable_audio_wanderer.runtime.ws_server import start_ws_server

    pipeline = PipelineManager(pretrained=args.pretrained)

    # Start HTTP server
    _start_http_server(web_dir, args.http_port)

    # Mutable container for the controller reference
    _controller_ref = [None]

    def on_perform_setup(corpus_dir, decoder, config):
        controller = _setup_perform_phase(
            corpus_dir, decoder, config, broadcaster, args.port
        )
        _controller_ref[0] = controller

    pipeline.set_perform_setup_callback(on_perform_setup)

    shutdown_requested = threading.Event()

    def request_shutdown(reason="web"):
        if shutdown_requested.is_set():
            return
        shutdown_requested.set()
        print(f"[serve] Shutdown requested via {reason}")
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except Exception:
            pass

    # Start WebSocket server with pipeline message handler (no nav/decoder yet)
    broadcaster, ws_thread = start_ws_server(
        nav=None,
        decoder=None,
        port=args.port,
        on_exit_request=request_shutdown,
        pipeline_message_handler=pipeline.handle_message,
    )
    pipeline.set_broadcaster(broadcaster)

    print()
    print(f"  Open http://localhost:{args.http_port} in your browser")
    print(f"  WebSocket on ws://127.0.0.1:{args.port}")
    print()

    try:
        # Block main thread
        signal.signal(signal.SIGINT, lambda *_: shutdown_requested.set())
        while not shutdown_requested.is_set():
            shutdown_requested.wait(timeout=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[serve] Shutting down...")
        if _controller_ref[0] is not None:
            _controller_ref[0].close()
        broadcaster.shutdown()

    print("[serve] Done.")


if __name__ == "__main__":
    main()
