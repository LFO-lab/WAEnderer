from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np

from stable_audio_wanderer.runtime.onnx_transport import OnnxTransportController
from stable_audio_wanderer.vae.onnx_decoder import (
    DecodedAudioWindow,
    DecoderWindowMetadata,
)


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("condition did not become true before timeout")


class FakeNavigation:
    def __init__(self):
        self.counter = 0
        self.variant = "random"
        self.fixed_retrieval_windows = []

    def has_variant(self, variant):
        return variant in ("random", "reorganized")

    def set_policy_variant(self, variant):
        if not self.has_variant(variant):
            return False
        self.variant = variant
        return True

    def step(self, fixed_retrieval_window=None):
        self.fixed_retrieval_windows.append(fixed_retrieval_window)
        self.counter += 1
        return self.counter

    def get_active_jump_rate(self, variant=None):
        return 0.5

    def set_random_controls(self, **_controls):
        pass

    def set_reorganized_controls(self, **_controls):
        pass


class FakeManifold:
    def generate_batch(self, frames, exploration=0.0):
        output = np.zeros((len(frames), 256), dtype=np.float32)
        output[:, 0] = np.asarray(frames, dtype=np.float32)
        return output


class FakeManual:
    control_dim = 3

    def __init__(self):
        self.faders = np.full(3, 0.5, dtype=np.float32)
        self.anchor = 0
        self.query_count = 0

    def set_faders(self, values):
        self.faders = np.asarray(values, dtype=np.float32)

    def set_wander_params(self, **_kwargs):
        pass

    def step_with_faders(self, values):
        self.faders = np.asarray(values, dtype=np.float32)
        self.query_count += 1
        return SimpleNamespace(nearest_index=self.anchor, distance=0.25)

    def get_state(self):
        return {
            "faders": self.faders.tolist(),
            "control_dim": 3,
            "position": [0.0, 0.0, 0.0],
            "wander_k": 1,
            "wander_speed": 0.0,
        }


class FakeDecoder:
    def __init__(self):
        self.supported_windows = (2, 4)
        self.default_window = 2
        self.info = SimpleNamespace(
            bundle_path=Path("/tmp/test.sawbundle"),
            backend="onnxruntime",
            provider="CPUExecutionProvider",
            vae_id="same_s",
        )
        self.calls = []
        self._control_lock = threading.Lock()
        self._block_next = False
        self._fail_next = False
        self.decode_entered = threading.Event()
        self.decode_release = threading.Event()

    def metadata_for(self, window):
        window = int(window)
        hop = (window + 1) // 2
        return DecoderWindowMetadata(
            latent_window=window,
            latent_dim=256,
            sample_rate=100,
            channels=2,
            samples_per_latent=8,
            audio_window_samples=window * 8,
            latent_hop=hop,
            audio_hop_samples=hop * 8,
            ola_mode="full_overlap_add",
        )

    def block_once(self):
        with self._control_lock:
            self._block_next = True
        self.decode_entered.clear()
        self.decode_release.clear()

    def fail_once(self):
        with self._control_lock:
            self._fail_next = True

    def decode(self, raw_latents):
        window = int(raw_latents.shape[0])
        with self._control_lock:
            should_block = self._block_next
            should_fail = self._fail_next
            self._block_next = False
            self._fail_next = False
        self.calls.append(window)
        if should_block:
            self.decode_entered.set()
            if not self.decode_release.wait(timeout=5.0):
                raise RuntimeError("test decode release timeout")
        if should_fail:
            raise RuntimeError("injected ORT failure")
        metadata = self.metadata_for(window)
        audio = np.full(
            (metadata.audio_window_samples, 2),
            np.float32(window),
            dtype=np.float32,
        )
        return DecodedAudioWindow(audio=audio, metadata=metadata, decode_time_ms=3.0)


class FakePlayer:
    sr = 100

    def __init__(self):
        self._lock = threading.Lock()
        self.buffers = {}
        self.writes = []
        self.current_generation = None
        self.pending_generation = None
        self.started = False
        self.closed = False
        self.reset_count = 0
        self.stop_count = 0
        self.fade_count = 0
        self.underruns = 0
        self.requested_generation = None

    def request_generation(self, generation):
        with self._lock:
            generation = int(generation)
            if (
                self.requested_generation is not None
                and generation < self.requested_generation
            ):
                return False
            self.requested_generation = generation
            if self.pending_generation is not None and self.pending_generation < generation:
                self.buffers.pop(self.pending_generation, None)
                self.pending_generation = None
        return True

    def write_hop(self, hop, generation=0):
        with self._lock:
            generation = int(generation)
            if (
                self.requested_generation is not None
                and generation < self.requested_generation
            ):
                return False
            self.writes.append((generation, np.asarray(hop).copy()))
            self.buffers[generation] = self.buffers.get(generation, 0) + hop.shape[1]
            if self.current_generation is None:
                self.current_generation = generation
            elif generation > self.current_generation:
                self.pending_generation = generation
        return True

    def generation_buffer_duration(self, generation):
        with self._lock:
            return self.buffers.get(int(generation), 0) / float(self.sr)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False
        self.stop_count += 1

    def reset_buffers(self):
        with self._lock:
            self.buffers.clear()
            self.current_generation = None
            self.pending_generation = None
            self.requested_generation = None
        self.reset_count += 1

    def fade_to_silence(self, _samples):
        self.fade_count += 1

    def close(self):
        self.closed = True

    def get_state(self):
        with self._lock:
            duration = self.buffers.get(self.current_generation, 0) / float(self.sr)
            pending = self.pending_generation
        return {
            "underruns": self.underruns,
            "buffer_duration": duration,
            "transition_status": "staging" if pending is not None else "idle",
        }


def _controller():
    decoder = FakeDecoder()
    player = FakePlayer()
    controller = OnnxTransportController(
        nav=FakeNavigation(),
        manual_engine=FakeManual(),
        manifold=FakeManifold(),
        player=player,
        latent_decoder=decoder,
        z_concat=np.zeros((12, 256), dtype=np.float32),
        file_offsets=np.asarray([0, 6, 12], dtype=np.int64),
        frame_file_ids=np.asarray([0] * 6 + [1] * 6, dtype=np.int32),
        z_mean=np.zeros(256, dtype=np.float32),
        z_std=np.ones(256, dtype=np.float32),
        initial_window=2,
    )
    return controller, decoder, player


def test_rapid_window_change_discards_in_flight_stale_decode_result():
    controller, decoder, player = _controller()
    ok, _ = controller.start()
    assert ok
    _wait_until(lambda: player.started)
    assert all(hop.shape == (2, 8) for generation, hop in player.writes if generation == 1)
    assert player.generation_buffer_duration(1) >= 0.120

    decoder.block_once()
    ok, _ = controller.set_decoder_window(4)
    assert ok
    assert decoder.decode_entered.wait(timeout=2.0)
    ok, _ = controller.set_decoder_window(2)
    assert ok
    decoder.decode_release.set()

    _wait_until(
        lambda: any(generation >= 3 for generation, _hop in player.writes)
    )
    assert not any(generation == 2 for generation, _hop in player.writes)
    assert controller.get_extra_state()["decoder"]["requested_window"] == 2
    controller.stop()


def test_web_onnx_navigation_uses_the_global_manifest_window():
    controller, _decoder, player = _controller()
    ok, _ = controller.start()
    assert ok
    _wait_until(lambda: len(controller.nav.fixed_retrieval_windows) >= 2)
    assert set(controller.nav.fixed_retrieval_windows) == {2}

    ok, _ = controller.set_decoder_window(4)
    assert ok
    _wait_until(lambda: 4 in controller.nav.fixed_retrieval_windows)
    assert None not in controller.nav.fixed_retrieval_windows
    assert player.requested_generation == 2
    controller.stop()


def test_manual_decode_request_queries_one_anchor_and_repeats_the_same_file_chunk():
    controller, _decoder, _player = _controller()
    controller.Z_concat[:, 0] = np.arange(12, dtype=np.float32)
    controller.manual.anchor = 5

    first = controller._manual_request(generation=1, window=4)
    second = controller._manual_request(generation=1, window=4)

    assert controller.manual.query_count == 2
    assert np.array_equal(first.raw_latents, second.raw_latents)
    assert first.raw_latents[:, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert np.all(first.raw_latents[:, 1:] == 0.0)


def test_stop_waits_for_in_flight_decode_before_resetting_pcm():
    controller, decoder, player = _controller()
    controller.start()
    _wait_until(lambda: player.started)
    decoder.block_once()
    controller.set_decoder_window(4)
    assert decoder.decode_entered.wait(timeout=2.0)
    resets_before_stop = player.reset_count

    result = []
    stop_thread = threading.Thread(target=lambda: result.append(controller.stop()))
    stop_thread.start()
    time.sleep(0.03)
    assert stop_thread.is_alive()
    assert player.reset_count == resets_before_stop

    decoder.decode_release.set()
    stop_thread.join(timeout=2.0)
    assert not stop_thread.is_alive()
    assert result[0][0]
    assert player.reset_count == resets_before_stop + 1


def test_runtime_failure_latches_visible_error_and_only_new_start_clears_it():
    controller, decoder, player = _controller()
    controller.start()
    _wait_until(lambda: player.started)
    decoder.fail_once()
    controller.set_decoder_window(4)

    _wait_until(lambda: controller.get_extra_state()["transport"]["error"] is not None)
    failed_state = controller.get_extra_state()
    assert not failed_state["transport"]["running"]
    assert "injected ORT failure" in failed_state["transport"]["error"]
    assert failed_state["decoder"]["transition_status"] == "error"
    assert player.fade_count == 1

    ok, _ = controller.start()
    assert ok
    _wait_until(lambda: player.started)
    restarted_state = controller.get_extra_state()
    assert restarted_state["transport"]["error"] is None
    assert restarted_state["transport"]["running"]
    controller.stop()
