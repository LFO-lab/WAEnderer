"""Real packaged ONNX pipeline smoke with native SAME-S imports forbidden."""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'stable_audio_3':
            raise AssertionError('ONNX pipeline tried to import native SAME-S')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.meta_path.insert(0, RejectNative())
    from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
    pipeline = PipelineManager()
    observations = {}
    def setup(_, decoder, config):
        pcm = decoder.decode(np.zeros((2,256), dtype=np.float32)).audio
        assert pcm.shape == (8192,2) and np.isfinite(pcm).all()
        observations.update(backend=decoder.info.backend, provider=decoder.info.provider,
                            pcm_shape=list(pcm.shape), native_imported='stable_audio_3' in sys.modules)
        return SimpleNamespace(close=lambda: None)
    pipeline.set_perform_setup_callback(setup)
    try:
        pipeline.handle_message({'type':'pipeline_start_perform', 'config':{'corpus_dir':str(args.corpus)}})
        assert pipeline.phase == 'perform', pipeline._perform_error
        pipeline.handle_message({'type':'pipeline_stop_perform'})
        assert pipeline.phase == 'idle'
    finally:
        pipeline.close()
    assert observations['native_imported'] is False
    observations['closed'] = pipeline.phase == 'closed'
    args.output.write_text(json.dumps(observations, indent=2)+'\n')


if __name__ == '__main__': main()
