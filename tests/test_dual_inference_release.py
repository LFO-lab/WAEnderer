"""Release invariants independent of GPUs and optional native installation."""
from pathlib import Path
import subprocess
import sys


def test_onnx_startup_and_standalone_cli_do_not_import_native_same_s():
    script = '''
import importlib.abc
import runpy
import sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'stable_audio_3':
            raise AssertionError('CPU startup unexpectedly loaded native SAME-S')
sys.meta_path.insert(0, NoNative())
from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
from stable_audio_wanderer.vae.decoder_factory import select_decoder
from stable_audio_wanderer.runtime.ws_server import WSBroadcaster
from stable_audio_wanderer.vae.onnx_decoder import SameSAppOnnxDecoder
sys.argv = ['perform.py', '--help']
runpy.run_module('bin.perform', run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--help' in result.stdout


def test_standalone_stops_audio_before_waiting_for_workers():
    import threading
    from types import SimpleNamespace
    from bin.perform import TransportController

    controller = object.__new__(TransportController)
    controller._running = threading.Event()
    controller._running.set()
    controller._decoder_started = True
    events = []
    controller.decoder = SimpleNamespace(
        stop=lambda: events.append('audio stopped'),
        reset_buffers=lambda: events.append('buffers cleared'))

    def join(timeout):
        assert not controller._running.is_set()
        assert events[0] == 'audio stopped'
        assert 'buffers cleared' not in events
        events.append('worker joined')

    controller._nav_thread = SimpleNamespace(join=join)
    controller._decode_thread = SimpleNamespace(join=join)
    controller._stats_thread = SimpleNamespace(join=join)
    controller._latent_queue = object()
    assert controller.stop()[0]
    assert events == ['audio stopped'] + ['worker joined'] * 3 + ['buffers cleared']
    assert controller._latent_queue is None
    assert not controller._decoder_started
