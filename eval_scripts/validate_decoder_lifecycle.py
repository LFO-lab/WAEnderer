"""Real model lifecycle probe; no audio hardware, listening or realtime claim.

PYTHONPATH=. python eval_scripts/validate_decoder_lifecycle.py --device mps --output report.json
Uses cached pinned weights only; alternates ONNX/native twice through PipelineManager.
"""
import argparse
import gc
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import weakref

import numpy as np
import torch

from stable_audio_wanderer.runtime.pipeline_server import PipelineManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    pipeline = PipelineManager()
    references, rows = [], []
    def setup(_, decoder, config):
        references.append(weakref.ref(decoder))
        decoded = decoder.decode(np.zeros((2, 256), dtype=np.float32))
        assert decoded.audio.shape == (8192, 2) and np.isfinite(decoded.audio).all()
        return SimpleNamespace(close=lambda: None)
    pipeline.set_perform_setup_callback(setup)
    with tempfile.TemporaryDirectory() as directory:
        np.savez(Path(directory) / "corpus.npz", vae_id="same_s", sr=44100,
                 latent_hz=44100 / 4096, Z_concat=np.zeros((32, 256), dtype=np.float32),
                 Z_mean=np.zeros(256, dtype=np.float32), Z_std=np.ones(256, dtype=np.float32))
        try:
            for index in range(4):
                backend = "pytorch" if index % 2 else "onnxruntime"
                start = time.perf_counter()
                pipeline.handle_message({"type": "pipeline_start_perform", "config": {
                    "corpus_dir": directory, "decoder_backend": backend,
                    "decoder_device": args.device if backend == "pytorch" else "cpu"}})
                assert pipeline.phase == "perform", pipeline._perform_error
                pipeline.handle_message({"type": "pipeline_stop_perform"})
                gc.collect()
                assert sum(ref() is not None for ref in references) == 1
                rows.append({"backend": backend, "prepare_and_decode_seconds": time.perf_counter()-start,
                             "live_decoder_instances": 1,
                             "mps_allocated_bytes": torch.mps.current_allocated_memory()
                             if args.device.startswith("mps") else None})
        finally:
            pipeline.close()
        gc.collect()
        assert all(ref() is None for ref in references)
    args.output.write_text(json.dumps({"device": args.device, "torch": torch.__version__,
        "cycles": rows, "live_instances_after_shutdown": 0,
        "mps_allocated_bytes_after_shutdown": torch.mps.current_allocated_memory()
        if args.device.startswith("mps") else None}, indent=2) + "\n")


if __name__ == "__main__":
    main()
