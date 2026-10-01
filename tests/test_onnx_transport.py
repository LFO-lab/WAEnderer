from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

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

    def get_state(self):
        return {
            "policy_index": float(self.counter),
            "current_file_id": 0,
        }


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
        return SimpleNamespace(
            raw_latents=raw,
            input_frames=tuple(
                int(anchor_frame) + offset for offset in range(int(window_size))
            ),
        )


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
            "nearest_index": int(self.anchor),
            "position": [0.0, 0.0, 0.0],
            "wander_k": 1,
            "wander_speed": 0.0,
        }


class FakeDecoder:
    def __init__(self):
        self.supported_windows = (2, 4, 8, 16, 32)
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
        self.reported_samples_per_latent = 8
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
            samples_per_latent=self.reported_samples_per_latent,
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
        self.continuation_generation = None
        self.provenance_writes = []
        self.presentation_state = None
        self.capture_calls = []

    def request_generation(self, generation, *, continue_current=False):
        with self._lock:
            generation = int(generation)
            if (
                self.requested_generation is not None
                and generation < self.requested_generation
            ):
                return False
            self.requested_generation = generation
            self.continuation_generation = self.current_generation if continue_current else None
            if self.pending_generation is not None and self.pending_generation < generation:
                self.buffers.pop(self.pending_generation, None)
                self.pending_generation = None
        return True

    def write_hop(
        self,
        hop,
        generation=0,
        frame_indices=None,
        samples_per_frame=None,
    ):
        with self._lock:
            generation = int(generation)
            if (
                self.requested_generation is not None
                and generation < self.requested_generation
                and generation != self.continuation_generation
            ):
                return False
            self.writes.append((generation, np.asarray(hop).copy()))
            self.provenance_writes.append(
                {
                    "generation": generation,
                    "hop": np.asarray(hop).copy(),
                    "frame_indices": tuple(int(index) for index in frame_indices),
                    "samples_per_frame": int(samples_per_frame),
                }
            )
            self.buffers[generation] = self.buffers.get(generation, 0) + hop.shape[1]
            if self.current_generation is None:
                self.current_generation = generation
            elif generation > self.current_generation:
                self.pending_generation = generation
        return True

    def capture_presentation_hold(
        self,
        index,
        *,
        generation=None,
        samples_per_frame=None,
    ):
        with self._lock:
            call = (
                int(index),
                None if generation is None else int(generation),
                None if samples_per_frame is None else int(samples_per_frame),
            )
            self.capture_calls.append(call)
            if self.presentation_state is not None:
                return False
            self.presentation_state = {
                "index": int(index),
                "recent_indices": [int(index)],
                "generation": None if generation is None else int(generation),
                "samples_into_frame": 0,
                "samples_per_frame": (
                    None if samples_per_frame is None else int(samples_per_frame)
                ),
            }
        return True

    def get_presentation_state(self):
        with self._lock:
            if self.presentation_state is None:
                return None
            state = dict(self.presentation_state)
            state["recent_indices"] = list(state["recent_indices"])
            return state

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


def _controller(*, initial_mode="random", initial_window=2):
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
        initial_mode=initial_mode,
        initial_window=initial_window,
    )
    return controller, decoder, player


def test_start_captures_stationary_cursor_before_producer_prebuffer():
    controller, _decoder, player = _controller()
    controller.nav.counter = 3

    ok, _ = controller.start()
    assert ok
    _wait_until(lambda: controller.nav.counter > 3)

    assert player.capture_calls == [(3, 1, 8)]
    assert controller.get_extra_state()["presentation"] == {
        "index": 3,
        "recent_indices": [3],
        "generation": 1,
        "samples_into_frame": 0,
        "samples_per_frame": 8,
    }
    controller.stop()

    controller.nav.counter = 9
    ok, _ = controller.start()
    assert ok
    _wait_until(lambda: controller.nav.counter > 9)
    assert player.capture_calls[-1] == (9, 2, 8)
    assert controller.get_extra_state()["presentation"]["index"] == 3
    assert controller.get_extra_state()["presentation"]["generation"] == 1
    controller.stop()


def test_extra_state_omits_missing_presentation_and_forwards_atomic_snapshot():
    controller, _decoder, player = _controller()
    assert "presentation" not in controller.get_extra_state()

    player.presentation_state = {
        "index": 7,
        "recent_indices": [2, 7],
        "generation": 4,
        "samples_into_frame": 3,
        "samples_per_frame": 8,
    }
    state = controller.get_extra_state()
    assert state["presentation"] == player.presentation_state
    assert state["presentation"] is not player.presentation_state


@pytest.mark.parametrize(
    ("window", "expected_indices"),
    (
        (2, (4,)),
        (4, (2, 3)),
        (8, (0, 1, 2, 3)),
        (16, (0, 1, 2, 3, 4, 5, 5, 5)),
        (32, (0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    ),
)
def test_manual_accepted_hops_forward_leading_latent_indices(window, expected_indices):
    controller, _decoder, player = _controller(
        initial_mode="manual",
        initial_window=window,
    )
    controller.manual.anchor = 5

    ok, _ = controller.start()
    assert ok
    _wait_until(lambda: bool(player.provenance_writes))
    _wait_until(lambda: player.started)

    accepted = player.provenance_writes[0]
    assert accepted["generation"] == 1
    assert accepted["frame_indices"] == expected_indices
    assert accepted["samples_per_frame"] == 8
    assert accepted["hop"].shape == (2, len(expected_indices) * 8)
    controller.stop()


def test_decode_rejects_provenance_that_does_not_cover_the_pcm_hop():
    controller, decoder, player = _controller(initial_mode="manual")
    decoder.reported_samples_per_latent = 7

    ok, _ = controller.start()
    assert ok
    _wait_until(lambda: controller.get_extra_state()["transport"]["error"] is not None)

    error = controller.get_extra_state()["transport"]["error"]
    assert "1 frames * 7 samples != 8 hop samples" in error
    assert player.provenance_writes == []
    controller.stop()


def test_decode_rejects_out_of_range_provenance_before_player_acceptance():
    controller, _decoder, player = _controller(initial_mode="manual")
    original_request = controller._manual_request

    def out_of_range_request(generation, window):
        request = original_request(generation, window)
        return SimpleNamespace(
            generation=request.generation,
            window=request.window,
            raw_latents=request.raw_latents,
            frame_indices=(controller.Z_concat.shape[0],) + request.frame_indices[1:],
        )

    controller._manual_request = out_of_range_request

    ok, _ = controller.start()
    assert ok
    _wait_until(lambda: controller.get_extra_state()["transport"]["error"] is not None)

    error = controller.get_extra_state()["transport"]["error"]
    assert "out-of-range corpus index" in error
    assert player.provenance_writes == []
    controller.stop()


def test_rapid_window_change_finishes_candidate_then_pursues_latest_target():
    controller, decoder, player = _controller()
    controller.start()
    try:
        _wait_until(lambda: player.started)
        decoder.block_once()
        assert controller.set_decoder_window(4)[0]
        assert decoder.decode_entered.wait(timeout=2.0)
        assert controller.set_decoder_window(8)[0]
        assert controller.set_decoder_window(2)[0]
        decoder.decode_release.set()
        _wait_until(lambda: player.pending_generation == 2)
        assert any(generation == 2 for generation, _ in player.writes)
        assert not any(generation == 3 for generation, _ in player.writes)
        # FakePlayer has no audio callback; explicitly complete its handoff.
        player.current_generation, player.pending_generation = 2, None
        _wait_until(lambda: player.pending_generation == 4)
        assert controller.get_extra_state()["decoder"]["requested_window"] == 2
        assert not any(generation == 3 for generation, _ in player.writes)
    finally:
        decoder.decode_release.set()
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
    _wait_until(lambda: player.requested_generation == 2)
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
    assert first.frame_indices == (4, 5, 6, 7)
    assert second.frame_indices == (6, 7, 8, 9)
    assert third.frame_indices == (8, 9)

    controls = planner.calls[0][2]
    assert controls["frame_source"] == "k_nearest"
    assert controls["frame_order"] == 0.0
    assert controls["latent_colour"] == 0.0
    assert controls["jump_rate"] == 0.55


def test_reorganized_request_provenance_preserves_tail_reuse_order():
    controller, _decoder, _player = _controller()
    lane = controller._new_lane(1, 4)
    first = controller._lane_request(lane, "reorganized")
    second = controller._lane_request(lane, "reorganized")

    assert first.frame_indices == (1, 2, 3, 4)
    assert second.frame_indices == (3, 4, 5, 6)
    assert first.raw_latents[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert second.raw_latents[:, 0].tolist() == [3.0, 4.0, 5.0, 6.0]


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
    assert first.frame_indices == (2, 3, 4, 5)
    assert second.frame_indices == (2, 3, 4, 5)


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


def test_window_controls_validate_atomically_and_are_reported():
    controller, _, _ = _controller()
    assert controller.set_window_controls({'mode': 'adaptive', 'minimum': 4, 'maximum': 16})[0]
    assert not controller.set_window_controls({'minimum': 32, 'content': 'held'})[0]
    state = controller._decoder_state()['window_controls']
    assert state == dict(mode='adaptive', minimum=4, maximum=16, content='source', variation=.25)
    assert not controller.set_window_controls({'variation': float('nan')})[0]
    assert not controller.set_window_controls({'minimum': 3})[0]


def test_manual_held_texture_and_overlap_survive_window_change():
    controller, _, _ = _controller(initial_mode='manual')
    controller.Z_concat[:] = np.arange(12)[:, None]
    assert controller.set_window_controls({'content': 'held'})[0]
    first = controller._manual_request(0, 8)
    np.testing.assert_array_equal(first.raw_latents, np.tile(first.raw_latents[0], (8, 1)))
    second = controller._manual_request(1, 4)
    np.testing.assert_array_equal(second.raw_latents[:2], first.raw_latents[-2:])
    assert len(set(second.frame_indices)) == 1


def test_manual_content_controls_do_not_reset_wander():
    controller, _, player = _controller()
    serial = controller._wander_reset_serial
    assert controller.set_window_controls({'content': 'variation'})[0]
    assert controller._wander_reset_serial == serial
    assert player.requested_generation is None


def test_adaptive_change_preserves_wander_planner_state():
    controller, _, _ = _controller()
    serial = controller._wander_reset_serial
    assert controller.set_decoder_window(4, adaptive=True)[0]
    assert controller._wander_reset_serial == serial


def test_adaptive_waits_for_staged_generation_before_changing_again():
    controller, _, player = _controller(initial_window=8)
    controller.set_window_controls({'mode': 'adaptive'})
    controller._running.set()
    controller._requested_generation = 1
    controller._generation_counter = 1
    player.current_generation = 1
    controller._adaptive.motion = 10
    controller._adapt_window()
    assert controller._requested_window == 2
    requested = controller._requested_generation
    controller._adaptive.motion = 0
    controller._adaptive.changed_at = -100
    controller._adapt_window()
    assert controller._requested_generation == requested
    controller._running.clear()


def test_content_change_stages_manual_audio_but_does_not_change_navigation():
    controller, _, player = _controller(initial_mode='manual')
    controller._running.set()
    assert controller.set_window_controls({'content': 'held'})[0]
    assert player.requested_generation == 1
    assert controller.nav.counter == 0
    controller._running.clear()


def test_long_candidate_decode_keeps_rendering_and_latest_target_eventually_plays():
    from test_decoder_player import _bare_player
    controller, decoder, _ = _controller()
    player = _bare_player(sr=100)
    player.underruns = 0
    player.started = False
    player.start = lambda: setattr(player, 'started', True)
    player.stop = lambda: setattr(player, 'started', False)
    controller.decoder = player
    entered, release = threading.Event(), threading.Event()
    original_decode = decoder.decode
    lock = threading.Lock()
    concurrent = peak = 0
    nav_threads = set()
    original_step = controller.nav.step
    def step(**kw):
        nav_threads.add(threading.current_thread().name)
        return original_step(**kw)
    controller.nav.step = step
    def decode(latents):
        nonlocal concurrent, peak
        with lock:
            concurrent += 1
            peak = max(peak, concurrent)
        try:
            if len(latents) == 4 and not entered.is_set():
                entered.set()
                assert release.wait(3)
            return original_decode(latents)
        finally:
            with lock:
                concurrent -= 1
    decoder.decode = decode
    def render():
        with player._queue_lock:
            audio, underrun = player._pcm_buffer.render(1)
        assert not underrun
        assert np.all(np.isfinite(audio)) and np.all(audio > 0)
        return audio
    controller.start()
    try:
        _wait_until(lambda: player.started)
        controller.set_decoder_window(4)
        assert entered.wait(2)
        calls = decoder.calls.count(2)
        # More than the initial audio reserve is consumed while T4 is blocked.
        for i in range(80):
            if i % 4 == 0:
                controller.set_decoder_window((8, 16, 32)[(i//4) % 3])
            assert np.allclose(render(), 2)
            time.sleep(.005)
        assert len(controller._generation_windows) <= 4
        assert decoder.calls.count(2) > calls + 3
        assert peak == 2
        assert nav_threads == {'saw-onnx-decode'}
        controller.set_decoder_window(8)
        release.set()
        seen = set()
        deadline = time.monotonic()+4
        while time.monotonic() < deadline:
            render()
            seen.add(player.current_generation)
            if controller.get_erae_state()['active_window'] == 8:
                break
            time.sleep(.005)
        assert 2 in seen  # The completed intermediate T4 was actually audible.
        assert controller.get_erae_state()['active_window'] == 8
        assert controller.get_erae_state()['requested_window'] == 8
        assert peak == 2
    finally:
        release.set()
        controller.stop()


def test_hard_reset_during_candidate_decode_rejects_old_candidate():
    controller, decoder, player = _controller()
    controller.start()
    try:
        _wait_until(lambda: player.started)
        decoder.block_once()
        controller.set_decoder_window(4)
        assert decoder.decode_entered.wait(2)
        controller.reset_wander(idx=3)
        decoder.decode_release.set()
        _wait_until(lambda: any(g == 3 for g, _ in player.writes))
        assert not any(g == 2 for g, _ in player.writes)
    finally:
        decoder.decode_release.set()
        controller.stop()


def test_staged_planners_share_corpus_but_not_traversal_or_texture_state():
    controller, _, _ = _controller()
    active = controller._new_lane(1, 2)
    active.reset_serial = 7
    candidate = controller._new_lane(2, 4, active)
    assert candidate.reset_serial == 7
    assert candidate.planner._latents is active.planner._latents
    candidate.planner._recent_units.append(3)
    assert not active.planner._recent_units
    assert candidate.texture.rng is not active.texture.rng
