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
        self.reset_policy_calls = []

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

    def get_random_controls(self):
        return {
            "phrase_scale": 0.4,
            "jump_rate": 0.55,
            "timbre_lock": 0.55,
            "drift": 0.5,
            "repeat_avoid": 0.75,
            "crossfile": 0.7,
        }

    def reset_policy(self, idx=None):
        self.reset_policy_calls.append(idx)
        if idx is not None:
            self.counter = int(idx)

    def set_random_controls(self, **_controls):
        pass

    def set_reorganized_controls(self, **_controls):
        pass


class FakeManifold:
    def __init__(self):
        self.calls = []

    def generate_batch(self, frames, exploration=0.0):
        self.calls.append((list(frames), float(exploration)))
        output = np.zeros((len(frames), 256), dtype=np.float32)
        output[:, 0] = np.asarray(frames, dtype=np.float32)
        return output


class FakeWanderPlanner:
    graph_available = False
    seed = 0x476F6F64
    last_diagnostics = None

    def __init__(self):
        self.calls = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def plan(self, anchor_frame, window_size, **controls):
        self.calls.append((int(anchor_frame), int(window_size), dict(controls)))
        raw = np.full(
            (int(window_size), 256),
            np.float32(anchor_frame),
            dtype=np.float32,
        )
        return SimpleNamespace(raw_latents=raw)


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


def test_wander_render_defaults_and_control_validation_are_visible_in_state():
    controller, _decoder, player = _controller()

    state = controller.get_extra_state()["wander_render"]
    assert state == {
        "frame_source": "k_nearest",
        "requested_frame_source": "k_nearest",
        "effective_frame_source": "k_nearest",
        "graph_available": False,
        "frame_order": 0.0,
        "latent_colour": 0.0,
        "seed": 0x476F6F64,
    }

    generation_before = controller._requested_generation
    ok, _ = controller.set_wander_render_controls(
        {
            "frame_source": "not_a_source",
            "frame_order": 4.5,
            "latent_colour": -2.0,
        }
    )
    assert ok
    state = controller.get_extra_state()["wander_render"]
    assert state["frame_source"] == "k_nearest"
    assert state["frame_order"] == 1.0
    assert state["latent_colour"] == 0.0
    assert controller._requested_generation == generation_before
    assert player.requested_generation is None

    ok, _ = controller.set_wander_render_controls(
        {"frame_source": "morphology_graph"}
    )
    assert ok
    fallback = controller.get_extra_state()["wander_render"]
    assert fallback["requested_frame_source"] == "morphology_graph"
    assert fallback["effective_frame_source"] == "contiguous"

    handled = controller.handle_ws_message(
        {
            "type": "wander_render",
            "controls": {
                "frame_source": "contiguous",
                "frame_order": 0.25,
                "latent_colour": 0.75,
            },
        }
    )
    assert handled
    state = controller.get_extra_state()["wander_render"]
    assert state["frame_source"] == "contiguous"
    assert state["frame_order"] == 0.25
    assert state["latent_colour"] == 0.75

    ok, _ = controller.set_wander_render_controls(
        {"frame_order": float("nan"), "latent_colour": "invalid"}
    )
    assert ok
    unchanged = controller.get_extra_state()["wander_render"]
    assert unchanged["frame_order"] == 0.25
    assert unchanged["latent_colour"] == 0.75


def test_live_wander_render_changes_stage_only_for_active_random_mode():
    controller, _decoder, player = _controller()

    with controller._lock:
        controller._generation_counter = 7
        controller._requested_generation = 7
        controller._active_mode = "reorganized"
        controller._running.set()
    ok, _ = controller.set_wander_render_controls({"frame_order": 0.2})
    assert ok
    assert controller._requested_generation == 7
    assert player.requested_generation is None

    with controller._lock:
        controller._active_mode = "random"
    ok, message = controller.set_wander_render_controls({"latent_colour": 0.4})
    assert ok
    assert "generation 8" in message
    assert controller._requested_generation == 8
    assert controller._generation_windows[8] == 2
    assert player.requested_generation == 8

    controller._running.clear()
    ok, _ = controller.set_wander_render_controls({"frame_order": 0.3})
    assert ok
    assert controller._requested_generation == 8
    assert player.requested_generation == 8


def test_wander_request_uses_navigation_cadence_and_final_anchor_without_tail_reuse():
    controller, _decoder, _player = _controller()
    planner = FakeWanderPlanner()
    controller.wander_planner = planner

    first, reset_serial = controller._wander_request(
        generation=1,
        window=4,
        previous_generation=None,
        planner_reset_serial=-1,
    )
    second, reset_serial = controller._wander_request(
        generation=1,
        window=4,
        previous_generation=1,
        planner_reset_serial=reset_serial,
    )
    third, reset_serial = controller._wander_request(
        generation=2,
        window=2,
        previous_generation=1,
        planner_reset_serial=reset_serial,
    )

    assert controller.nav.fixed_retrieval_windows == [4, 4, 4, 4, 4, 4, 2, 2]
    assert [call[:2] for call in planner.calls] == [(4, 4), (6, 4), (8, 2)]
    assert planner.reset_count == 1
    assert controller.manifold.calls == []
    assert np.all(first.raw_latents == 4.0)
    assert np.all(second.raw_latents == 6.0)
    assert np.all(third.raw_latents == 8.0)
    assert second.raw_latents.shape == (4, 256)

    controls = planner.calls[0][2]
    assert controls["frame_source"] == "k_nearest"
    assert controls["frame_order"] == 0.0
    assert controls["latent_colour"] == 0.0
    assert controls["jump_rate"] == 0.55


def test_reset_wander_resets_policy_stages_random_generation_and_resets_planner():
    controller, _decoder, player = _controller()
    planner = FakeWanderPlanner()
    controller.wander_planner = planner

    with controller._lock:
        controller._generation_counter = 3
        controller._requested_generation = 3
        controller._generation_windows = {3: 4}
        controller._requested_window = 4
        controller._active_mode = "random"
        controller._running.set()

    serial_before = controller._wander_reset_serial
    ok, message = controller.reset_wander(idx=5)
    assert ok
    assert message == "Wander reset"
    assert controller.nav.reset_policy_calls == [5]
    assert controller._wander_reset_serial == serial_before + 1
    assert controller._requested_generation == 4
    assert controller._generation_windows[4] == 4
    assert player.requested_generation == 4

    request, observed_serial = controller._wander_request(
        generation=4,
        window=4,
        previous_generation=3,
        planner_reset_serial=serial_before,
    )
    assert planner.reset_count == 1
    assert observed_serial == serial_before + 1
    assert planner.calls[-1][0] == 9
    assert np.all(request.raw_latents == 9.0)

    controller._running.clear()
    generation_before = controller._requested_generation
    ok, _ = controller.reset_wander()
    assert ok
    assert controller.nav.reset_policy_calls == [5, None]
    assert controller._requested_generation == generation_before


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
