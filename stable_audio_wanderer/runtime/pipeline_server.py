"""
PipelineManager: orchestrates preprocess -> train -> perform phases
over a shared WebSocket connection.
"""
import glob
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ..vae.decoder_contract import LatentDecoder


class PipelineManager:
    """
    Manages the phase state machine and background workers for the
    preprocess -> train -> perform pipeline.

    Phases: idle -> preprocess -> train -> perform
    """

    PHASES = ("idle", "preprocess", "train", "preparing", "perform", "stopping", "error", "closed")

    def __init__(self, pretrained: str = "stabilityai/stable-audio-open-1.0",
                 decoder_resource_dir=None):
        self.pretrained = pretrained  # legacy fallback
        self.phase = "idle"
        self._vae = None
        self._corpus_dir: Optional[str] = None
        self._cancel = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._broadcaster = None  # set by serve.py after WS server starts
        self._perform_setup_callback = None  # called when perform phase starts
        self._decoder_resource_dir = decoder_resource_dir
        self._app_decoder: Optional["LatentDecoder"] = None
        self._decoder_selection = None
        self._perform_controller = None
        self._perform_teardown_callback = None
        self._perform_error = None
        self._lifecycle_lock = threading.RLock()
        self._closed = False

        # Preprocess result for passing VAE + corpus to train/perform
        self._preprocess_result: Optional[dict] = None

    def set_broadcaster(self, broadcaster):
        """Attach the WSBroadcaster for pushing progress messages."""
        self._broadcaster = broadcaster

    def set_perform_setup_callback(self, callback):
        """Set callback invoked when entering perform phase.
        Signature: callback(corpus_dir: str, decoder: LatentDecoder, config: dict)
        Return the controller owned and closed by this manager. If construction
        fails, the callback must close any partially constructed controller.
        """
        self._perform_setup_callback = callback

    def set_perform_teardown_callback(self, callback):
        """Detach external command/state bindings; this manager closes the controller."""
        self._perform_teardown_callback = callback

    def _emit(self, data: dict):
        """Push a message to all connected WebSocket clients."""
        if self._broadcaster is not None and not self._closed:
            self._broadcaster.broadcast_pipeline_message(data)

    def handle_message(self, data: dict):
        with self._lifecycle_lock:
            if not self._closed:
                self._dispatch_message(data)

    def _dispatch_message(self, data: dict):
        """Route incoming pipeline_* WebSocket messages."""
        msg_type = data.get("type", "")

        if msg_type == "pipeline_list_files":
            self._handle_list_files(data)
        elif msg_type == "pipeline_list_corpora":
            self._handle_list_corpora()
        elif msg_type == "pipeline_list_vaes":
            self._handle_list_vaes()
        elif msg_type == "pipeline_start_preprocess":
            self._handle_start_preprocess(data)
        elif msg_type == "pipeline_start_train":
            self._handle_start_train(data)
        elif msg_type == "pipeline_start_perform":
            self._handle_start_perform(data)
        elif msg_type == "pipeline_stop_perform":
            self._handle_stop_perform()
        elif msg_type == "pipeline_cancel":
            self._handle_cancel()
        elif msg_type == "pipeline_get_state":
            self._emit_phase_state()

    def _emit_phase_state(self):
        self._emit({
            "type": "pipeline_state",
            "phase": self.phase,
            "corpus_dir": self._corpus_dir,
            "decoder": self._active_decoder_state(),
            "error": self._perform_error,
        })

    # ------------------------------------------------------------------
    # File listing
    # ------------------------------------------------------------------

    def _handle_list_files(self, data: dict):
        audio_dir = data.get("audio_dir", "")
        if not audio_dir or not os.path.isdir(audio_dir):
            self._emit({
                "type": "pipeline_file_list",
                "audio_dir": audio_dir,
                "files": [],
                "error": "Directory not found",
            })
            return

        wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        file_infos = []
        for p in wav_files:
            try:
                size_bytes = os.path.getsize(p)
                # Approximate duration: WAV 44100 Hz stereo 16-bit ~ 176400 bytes/sec
                approx_duration = size_bytes / 176400.0
            except OSError:
                approx_duration = 0.0
            file_infos.append({
                "name": os.path.basename(p),
                "path": p,
                "approx_duration": round(approx_duration, 1),
            })

        self._emit({
            "type": "pipeline_file_list",
            "audio_dir": audio_dir,
            "files": file_infos,
        })

    def _handle_list_corpora(self):
        corpus_root = os.path.join(os.getcwd(), "corpus")
        corpora = []
        if os.path.isdir(corpus_root):
            for entry in sorted(os.listdir(corpus_root)):
                entry_path = os.path.join(corpus_root, entry)
                if os.path.isdir(entry_path):
                    corpus_npz = os.path.join(entry_path, "corpus.npz")
                    if os.path.exists(corpus_npz):
                        corpora.append({
                            "name": entry,
                            "path": entry_path,
                        })
        self._emit({
            "type": "pipeline_corpus_list",
            "corpora": corpora,
        })

    # ------------------------------------------------------------------
    # VAE listing
    # ------------------------------------------------------------------

    def _handle_list_vaes(self):
        from stable_audio_wanderer.vae import list_vaes
        vaes = []
        for info in list_vaes():
            vaes.append({
                "vae_id": info.vae_id,
                "display_name": info.display_name,
                "sample_rate": info.sample_rate,
                "latent_hz": info.latent_hz,
                "latent_dim": info.latent_dim,
                "channels": info.channels,
                "requires_path": info.requires_path,
                "path_label": info.path_label,
            })
        self._emit({"type": "pipeline_vae_list", "vaes": vaes})

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    def _handle_start_preprocess(self, data: dict):
        if self.phase not in ("idle",):
            self._emit({"type": "pipeline_state", "phase": self.phase, "error": "Cannot preprocess now"})
            return

        config = data.get("config", {})
        audio_dir = config.get("audio_dir", "")
        if not audio_dir or not os.path.isdir(audio_dir):
            self._emit({"type": "pipeline_state", "phase": self.phase, "error": "Invalid audio_dir"})
            return

        self._release_app_decoder()
        self.phase = "preprocess"
        self._cancel.clear()
        self._emit({"type": "pipeline_phase_change", "phase": "preprocess"})

        self._worker = threading.Thread(
            target=self._preprocess_worker, args=(config,), daemon=True
        )
        self._worker.start()

    def _preprocess_worker(self, config: dict):
        try:
            from bin.preprocess import run_preprocess

            def progress_cb(event_data):
                self._emit({"type": "pipeline_stats", "phase": "preprocess", **event_data})

            result = run_preprocess(
                audio_dir=config.get("audio_dir"),
                out_prefix=config.get("out_prefix", "corpus"),
                pretrained=self.pretrained,
                vae=self._vae,
                vae_id=str(config.get("vae_id", "")),
                vae_weight_path=str(config.get("vae_weight_path", "")),
                progress_callback=progress_cb,
                cancel_event=self._cancel,
                latent_nav_k=int(config.get("latent_nav_k", 32)),
                encode_chunk_sec=float(config.get("encode_chunk_sec", 60.0)),
                encode_chunk_overlap_sec=float(config.get("encode_chunk_overlap_sec", 1.0)),
                trim_silence=bool(config.get("trim_silence", True)),
                silence_threshold_db=float(config.get("silence_threshold_db", -45.0)),
                silence_min_duration_sec=float(config.get("silence_min_duration_sec", 0.25)),
                silence_keep_sec=float(config.get("silence_keep_sec", 0.10)),
                manual_reducer=str(config.get("manual_reducer", "pca")),
                manual_embed_dim=int(config.get("manual_embed_dim", 4)),
                reorg_min_sec=float(config.get("reorg_min_sec", 2.0)),
                reorg_max_sec=float(config.get("reorg_max_sec", 10.0)),
                reorg_target_sec=float(config.get("reorg_target_sec", 5.0)),
                reorg_candidate_k=int(config.get("reorg_candidate_k", 64)),
                reorg_graph_k=int(config.get("reorg_graph_k", 24)),
                reorg_weight_entry=float(config.get("reorg_weight_entry", 0.70)),
                reorg_weight_delta=float(config.get("reorg_weight_delta", 0.30)),
                reorg_crossfile_penalty=float(config.get("reorg_crossfile_penalty", 0.10)),
                reorg_boundary_smoothness_weight=float(config.get("reorg_boundary_smoothness_weight", 0.35)),
            )

            if result.get("cancelled"):
                self.phase = "idle"
                self._emit({"type": "pipeline_phase_change", "phase": "idle", "reason": "cancelled"})
                return

            self._preprocess_result = result
            self._corpus_dir = result["corpus_dir"]
            self._vae = result.get("vae")
            self._release_preprocessing_vae()
            self.phase = "idle"
            self._emit({
                "type": "pipeline_phase_change",
                "phase": "idle",
                "completed": "preprocess",
                "corpus_dir": self._corpus_dir,
                "total_files": result.get("total_files", 0),
                "total_frames": result.get("total_frames", 0),
                "silence_removed_pct": result.get("silence_removed_pct", 0.0),
            })

        except Exception as e:
            self.phase = "idle"
            self._emit({"type": "pipeline_phase_change", "phase": "idle", "error": str(e)})
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def _handle_start_train(self, data: dict):
        if self.phase not in ("idle",):
            self._emit({"type": "pipeline_state", "phase": self.phase, "error": "Cannot train now"})
            return

        config = data.get("config", {})
        corpus_dir = config.get("corpus_dir") or self._corpus_dir
        if not corpus_dir or not os.path.isdir(corpus_dir):
            self._emit({"type": "pipeline_state", "phase": self.phase, "error": "No corpus_dir"})
            return

        self._release_app_decoder()
        self.phase = "train"
        self._cancel.clear()
        self._corpus_dir = corpus_dir
        self._emit({"type": "pipeline_phase_change", "phase": "train"})

        self._worker = threading.Thread(
            target=self._train_worker, args=(config, corpus_dir), daemon=True
        )
        self._worker.start()

    def _train_worker(self, config: dict, corpus_dir: str):
        try:
            from bin.train_policy import run_train

            def progress_cb(event_data):
                self._emit({"type": "pipeline_stats", "phase": "train", **event_data})

            result = run_train(
                corpus_dir=corpus_dir,
                navigation_mode=str(config.get("navigation_mode", "all")),
                progress_callback=progress_cb,
                cancel_event=self._cancel,
                epochs=int(config.get("epochs", 2000)),
                batch_size=int(config.get("batch_size", 64)),
                lr=float(config.get("lr", 1e-3)),
                weight_decay=float(config.get("weight_decay", 1e-4)),
                hidden=int(config.get("hidden", 256)),
                layers=int(config.get("layers", 2)),
                seq_len=int(config.get("seq_len", 32)),
                lambda_recon=float(config.get("lambda_recon", 1.0)),
                lambda_smooth=float(config.get("lambda_smooth", 0.1)),
                lambda_manifold=float(config.get("lambda_manifold", 0.1)),
                lambda_diversity=float(config.get("lambda_diversity", 0.01)),
                lambda_window=float(config.get("lambda_window", 0.5)),
                reorganized_epochs=int(config.get("reorganized_epochs", 120)),
                reorganized_batch_size=int(config.get("reorganized_batch_size", 128)),
                reorganized_lr=float(config.get("reorganized_lr", 1e-3)),
                reorganized_hidden_dim=int(config.get("reorganized_hidden_dim", 192)),
                reorganized_layers=int(config.get("reorganized_layers", 3)),
                reorganized_dropout=float(config.get("reorganized_dropout", 0.10)),
                log_every=int(config.get("log_every", 50)),
                verbose=bool(config.get("verbose", False)),
            )

            self.phase = "idle"
            self._emit({
                "type": "pipeline_phase_change",
                "phase": "idle",
                "completed": "train",
                "corpus_dir": corpus_dir,
                **{k: v for k, v in result.items() if isinstance(v, (str, int, float))},
            })

        except Exception as e:
            self.phase = "idle"
            self._emit({"type": "pipeline_phase_change", "phase": "idle", "error": str(e)})
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Perform
    # ------------------------------------------------------------------

    def _release_preprocessing_vae(self):
        """Release the encode-side model before preparing a performance decoder."""
        retained_vae = self._vae
        device_types = set()
        candidates = [
            retained_vae,
            getattr(retained_vae, "raw_model", None),
            getattr(retained_vae, "_model", None),
        ]
        candidates.extend(getattr(item, "autoencoder", None) for item in tuple(candidates))
        for candidate in candidates:
            parameters = getattr(candidate, "parameters", None)
            if not callable(parameters):
                continue
            try:
                for parameter in parameters():
                    device_type = getattr(getattr(parameter, "device", None), "type", None)
                    if isinstance(device_type, str):
                        device_types.add(device_type)
                    break
            except Exception:
                pass

        self._vae = None
        if isinstance(self._preprocess_result, dict):
            self._preprocess_result["vae"] = None

        # Drop the local inspection references before collection/cache release.
        candidates.clear()
        retained_vae = None
        candidate = None
        parameter = None
        parameters = None

        import gc

        gc.collect()
        torch = sys.modules.get("torch")
        if torch is None:
            return
        try:
            if (
                "mps" in device_types
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "empty_cache")
            ):
                torch.mps.empty_cache()
        except Exception:
            pass
        try:
            if (
                "cuda" in device_types
                and hasattr(torch, "cuda")
                and hasattr(torch.cuda, "empty_cache")
            ):
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _handle_start_perform(self, data: dict):
        if self.phase not in ("idle",):
            self._emit({"type": "pipeline_state", "phase": self.phase,
                        "error": "Cannot start perform now; use pipeline_stop_perform before reconfiguration"})
            return
        if self._perform_setup_callback is None:
            self._emit({"type": "pipeline_state", "phase": self.phase,
                        "error": "Perform setup callback is unavailable"})
            return
        try:
            from ..vae.decoder_factory import select_decoder, create_decoder
            from ..vae.onnx_decoder import validate_same_s_corpus
            config = data.get("config", {})
            if not isinstance(config, dict):
                raise ValueError("Perform config must be an object")
            corpus_dir = config.get("corpus_dir") or self._corpus_dir
            if not corpus_dir or not os.path.isdir(corpus_dir):
                raise ValueError("No corpus_dir")
            window = config.get("decoder_window", 2)
            if isinstance(window, bool) or int(window) != float(window):
                raise ValueError("Invalid decoder_window")
            decoder_window = int(window)
            # Corpus checks are mandatory even when an instance is cached.
            validate_same_s_corpus(corpus_dir)
            selection = select_decoder(config, resource_dir=self._decoder_resource_dir)
            self._perform_error = None
            self.phase = "preparing"
            self._emit({"type": "pipeline_phase_change", "phase": self.phase, "decoder": None})
            self._teardown_perform()
            if selection != self._decoder_selection:
                self._release_app_decoder()
            self._release_preprocessing_vae()
            if self._app_decoder is None:
                self._app_decoder = create_decoder(selection)
                self._decoder_selection = selection
            decoder = self._app_decoder
            if decoder_window not in decoder.supported_windows:
                raise RuntimeError(f"Decoder window T{decoder_window} is unavailable; supported: {decoder.supported_windows}")
            self._perform_controller = self._perform_setup_callback(corpus_dir, decoder, dict(config))
            self._corpus_dir = corpus_dir
            self.phase = "perform"
            self._emit({"type": "pipeline_phase_change", "phase": self.phase,
                        "corpus_dir": corpus_dir, "decoder": self._active_decoder_state()})
        except Exception as exc:
            self._perform_failed(exc)

    def _active_decoder_state(self):
        if self.phase != "perform" or self._app_decoder is None:
            return None
        info = self._app_decoder.info
        return {"backend": info.backend, "provider": info.provider,
                "device": getattr(info, "device", "cpu"), "vae_id": info.vae_id}

    def _teardown_perform(self):
        # Close even if detach fails, so a stale external controller cannot Start.
        try:
            if self._perform_teardown_callback is not None:
                self._perform_teardown_callback()
        finally:
            if self._perform_controller is not None:
                self._perform_controller.close()
                self._perform_controller = None

    def _release_app_decoder(self):
        if self._perform_controller is not None:
            raise RuntimeError("Cannot release decoder before transport has drained")
        self._decoder_selection = None
        if self._app_decoder is not None:
            self._app_decoder.close()
            self._app_decoder = None

    def _perform_failed(self, exc):
        self._perform_error = str(exc)
        try:
            self._teardown_perform()
            self._release_app_decoder()
            self.phase = "idle"
        except Exception as cleanup_error:
            # Retain ownership if draining/release failed; never free under a worker.
            self.phase = "error"
            self._perform_error += f"; cleanup failed: {cleanup_error}"
        self._emit({"type": "pipeline_phase_change", "phase": self.phase,
                    "decoder": None, "error": self._perform_error})

    def _handle_stop_perform(self):
        if self.phase not in ("perform", "error", "idle"):
            self._emit({"type": "pipeline_state", "phase": self.phase, "error": "Cannot leave perform now"})
            return
        self.phase = "stopping"
        self._emit({"type": "pipeline_phase_change", "phase": self.phase, "decoder": None})
        try:
            self._teardown_perform()
            self.phase = "idle"
            self._perform_error = None
            # Keep one prepared decoder for an identical restart. A changed
            # selection releases it before allocating the replacement.
            self._emit({"type": "pipeline_phase_change", "phase": self.phase, "decoder": None})
        except Exception as exc:
            self._perform_failed(exc)

    def close(self):
        """Shut down producers before releasing their model resources."""
        with self._lifecycle_lock:
            self._closed = True
            self._cancel.set()
            if self._worker is not None and self._worker is not threading.current_thread():
                self._worker.join()
                self._worker = None
            self._teardown_perform()
            self._release_app_decoder()
            self._release_preprocessing_vae()
            self.phase = "closed"

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def _handle_cancel(self):
        if self.phase in ("preprocess", "train"):
            self._cancel.set()
            self._emit({"type": "pipeline_state", "phase": self.phase, "cancelling": True})
        else:
            self._emit({"type": "pipeline_state", "phase": self.phase, "error": "Nothing to cancel"})
