import json
import asyncio
import threading

import numpy as np
import pytest

pytest.importorskip("websockets")

from stable_audio_wanderer.runtime.ws_server import WSBroadcaster


def test_pipeline_preparation_does_not_block_websocket_event_loop():
    entered, release = threading.Event(), threading.Event()
    def prepare(_):
        entered.set()
        assert release.wait(3)
    broadcaster = WSBroadcaster(pipeline_message_handler=prepare)
    async def run():
        task = asyncio.create_task(broadcaster._handle_message(
            None, json.dumps({"type": "pipeline_start_perform"})))
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(.01)
            assert entered.is_set() and not task.done()
            assert json.loads(broadcaster._get_state_json())["type"] == "state"
        finally:
            release.set()
            await task
    asyncio.run(run())


def test_unbind_removes_old_corpus_and_control_references():
    broadcaster = WSBroadcaster()
    pushed = []
    broadcaster.broadcast_pipeline_message = pushed.append
    broadcaster.bind_nav_decoder(FakeNav(_make_nav_state()), FakeDecoder(),
                                 message_handler=lambda _: True,
                                 extra_state_provider=lambda: {"old": True},
                                 manual_points_3d=np.ones((3, 3), dtype=np.float32))
    broadcaster.unbind_nav_decoder()
    assert broadcaster.nav is None and broadcaster.decoder is None
    assert broadcaster._message_handler is None and broadcaster._extra_state_provider is None
    assert broadcaster._manual_points is None
    assert pushed[-1]["total_points"] == 0
    assert "old" not in json.loads(broadcaster._get_state_json())


class FakeNav:
    def __init__(self, state):
        self.ZZ = np.asarray(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._file_ids = np.asarray([0, 1, 1], dtype=np.int32)
        self.N = int(self.ZZ.shape[0])
        self._state = state
        self.cursor_index_calls = []

    def get_state(self):
        return self._state

    def set_cursor_index(self, idx):
        self.cursor_index_calls.append(int(idx))


class FakeDecoder:
    def get_state(self):
        return {"gain": 1.0, "underruns": 2, "transition_status": "idle"}


def _make_nav_state():
    return {
        "policy_index": 1.0,
        "policy_velocity": 0.25,
        "current_file_id": 1,
        "recent_indices": [0, 1, 2],
        "controls": {
            "random": {"jump_rate": 0.5},
            "reorganized": {"jump_rate": 0.4},
        },
        "fractional": {
            "idx_lower": 1,
            "idx_upper": 2,
            "frac": 0.25,
            "same_file": True,
            "file_id_lower": 1,
            "file_id_upper": 1,
        },
    }


def test_ws_broadcaster_exposes_manual_space_navigation_for_random_mode():
    nav = FakeNav(_make_nav_state())
    broadcaster = WSBroadcaster(
        nav,
        extra_state_provider=lambda: {
            "navigation_mode": "random",
            "transport": {"running": False, "selected_mode": "random"},
        },
        manual_points_3d=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        manual_fader_p01=np.zeros(3, dtype=np.float32),
        manual_fader_p99=np.ones(3, dtype=np.float32),
    )

    state = json.loads(broadcaster._get_state_json())

    assert state["navigation"]["mode"] == "random"
    assert state["navigation"]["position_3d"] == pytest.approx([0.625, 0.625, 0.625])
    assert np.allclose(
        np.asarray(state["navigation"]["trajectory_3d"], dtype=np.float32),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_ws_broadcaster_normalizes_manual_position_from_extra_state():
    nav = FakeNav(_make_nav_state())
    broadcaster = WSBroadcaster(
        nav,
        extra_state_provider=lambda: {
            "navigation_mode": "manual",
            "transport": {"running": False, "selected_mode": "manual"},
            "manual": {
                "position": [15.0, 30.0, 35.0],
                "nearest_index": 1,
            },
        },
        manual_points_3d=np.asarray(
            [
                [10.0, 20.0, 30.0],
                [20.0, 40.0, 50.0],
                [20.0, 40.0, 50.0],
            ],
            dtype=np.float32,
        ),
        manual_fader_p01=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        manual_fader_p99=np.asarray([20.0, 40.0, 50.0], dtype=np.float32),
    )

    state = json.loads(broadcaster._get_state_json())

    assert state["manual"]["position_3d"] == pytest.approx([0.5, 0.5, 0.25])


@pytest.mark.parametrize("navigation_mode", ["random", "manual", "reorganized"])
def test_ws_broadcaster_uses_audio_render_presentation_cursor(navigation_mode):
    nav = FakeNav(_make_nav_state())
    broadcaster = WSBroadcaster(
        nav,
        extra_state_provider=lambda: {
            "navigation_mode": navigation_mode,
            "transport": {"running": True, "selected_mode": navigation_mode},
            "manual": {
                "position": [0.5, 0.5, 0.5],
                "nearest_index": 1,
                "current_file_id": 1,
                "distance": 0.375,
                "faders": [0.2, 0.3, 0.4, 0.5],
            },
            "presentation": {
                "index": 0,
                "recent_indices": [2, 0, 2, 0],
                "generation": 7,
                "samples_into_frame": 2048,
                "samples_per_frame": 4096,
            },
        },
        manual_points_3d=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        manual_file_ids=np.asarray([0, 1, 1], dtype=np.int32),
        manual_fader_p01=np.zeros(3, dtype=np.float32),
        manual_fader_p99=np.ones(3, dtype=np.float32),
    )

    state = json.loads(broadcaster._get_state_json())

    assert state["navigation"] == {
        "index": 0,
        "index_normalized": 0.0,
        "velocity": 0.25,
        "file_id": 0,
        "fractional": None,
        "timbre_swap": {},
        "recompose": {},
        "policy_v2": {},
        "reorganized": {},
        "mode": navigation_mode,
        "position_3d": [0.0, 0.0, 0.0],
        "trajectory_3d": [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        "clock": "audio_render",
        "generation": 7,
    }
    assert state["controls"] == nav.get_state()["controls"]
    assert state["manual"] == {
        "position": [0.0, 0.0, 0.0],
        "nearest_index": 0,
        "current_file_id": 0,
        "distance": 0.375,
        "faders": [0.2, 0.3, 0.4, 0.5],
        "position_3d": [0.0, 0.0, 0.0],
    }


def test_ws_broadcaster_ignores_non_finite_presentation_index():
    nav = FakeNav(_make_nav_state())
    broadcaster = WSBroadcaster(
        nav,
        extra_state_provider=lambda: {
            "presentation": {
                "index": float("nan"),
                "recent_indices": [0],
                "generation": 1,
                "samples_into_frame": 0,
                "samples_per_frame": 4096,
            }
        },
        manual_points_3d=np.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
    )

    state = json.loads(broadcaster._get_state_json())

    assert state["navigation"]["index"] == 1
    assert state["navigation"]["fractional"]["frac"] == 0.25
    assert "clock" not in state["navigation"]


@pytest.mark.parametrize("presented_index", [99, -1, 1.5])
def test_ws_broadcaster_ignores_invalid_presentation_index(presented_index):
    nav = FakeNav(_make_nav_state())
    broadcaster = WSBroadcaster(
        nav,
        extra_state_provider=lambda: {
            "presentation": {
                "index": presented_index,
                "recent_indices": [0],
                "generation": 1,
                "samples_into_frame": 0,
                "samples_per_frame": 4096,
            }
        },
        manual_points_3d=np.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
    )

    state = json.loads(broadcaster._get_state_json())

    assert state["navigation"]["index"] == 1
    assert state["navigation"]["fractional"]["frac"] == 0.25
    assert "clock" not in state["navigation"]


def test_ws_broadcaster_preserves_full_manual_position_and_filters_bad_history():
    nav = FakeNav(_make_nav_state())
    broadcaster = WSBroadcaster(
        nav,
        extra_state_provider=lambda: {
            "navigation_mode": "manual",
            "presentation": {
                "index": 1,
                "recent_indices": [0, 99, -1, 1.25, 2],
                "generation": 3,
                "samples_into_frame": 0,
                "samples_per_frame": 4096,
            },
        },
        manual_points_3d=np.asarray(
            [
                [0.0, 0.0, 0.0, 0.25],
                [0.5, 0.5, 0.5, 0.50],
                [1.0, 1.0, 1.0, 0.75],
            ],
            dtype=np.float32,
        ),
        manual_file_ids=np.asarray([0, 1, 1], dtype=np.int32),
    )

    state = json.loads(broadcaster._get_state_json())

    assert state["manual"]["position"] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(
        np.asarray(state["navigation"]["trajectory_3d"], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
    )


def test_ws_broadcaster_uses_point_indices_and_cursor_index_messages():
    nav = FakeNav(_make_nav_state())
    broadcaster = WSBroadcaster(
        nav,
        manual_points_3d=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )

    corpus = json.loads(broadcaster._get_corpus_json())
    assert corpus["point_indices"] == [0, 1, 2]
    assert "positions_2d" not in corpus

    asyncio.run(
        broadcaster._handle_message(None, json.dumps({"type": "cursor_index", "index": 2}))
    )

    assert nav.cursor_index_calls == [2]


def test_ws_broadcaster_merges_onnx_transport_decoder_metadata():
    nav = FakeNav(_make_nav_state())
    broadcaster = WSBroadcaster(
        nav,
        decoder=FakeDecoder(),
        extra_state_provider=lambda: {
            "decoder": {
                "backend": "onnxruntime",
                "provider": "CPUExecutionProvider",
                "supported_windows": [2, 4],
                "selected_window": 2,
                "audio_hop_samples": 4096,
                "error": "latched failure",
                "underruns": 3,
            },
            "wander_render": {
                "requested_frame_source": "morphology_graph",
                "effective_frame_source": "contiguous",
                "graph_available": False,
                "frame_order": 0.25,
                "latent_colour": 0.5,
                "seed": 0x476F6F64,
            },
        },
    )

    state = json.loads(broadcaster._get_state_json())
    assert state["decoder"] == {
        "gain": 1.0,
        "underruns": 3,
        "transition_status": "idle",
        "backend": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "supported_windows": [2, 4],
        "selected_window": 2,
        "audio_hop_samples": 4096,
        "error": "latched failure",
    }
    assert state["wander_render"] == {
        "requested_frame_source": "morphology_graph",
        "effective_frame_source": "contiguous",
        "graph_available": False,
        "frame_order": 0.25,
        "latent_colour": 0.5,
        "seed": 0x476F6F64,
    }


def test_late_bind_pushes_the_new_corpus_to_existing_clients():
    broadcaster = WSBroadcaster()
    pushed = []
    broadcaster.broadcast_pipeline_message = lambda data: pushed.append(data)
    nav = FakeNav(_make_nav_state())

    broadcaster.bind_nav_decoder(
        nav,
        FakeDecoder(),
        manual_points_3d=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
            dtype=np.float32,
        ),
    )

    assert len(pushed) == 1
    assert pushed[0]["type"] == "corpus"
    assert pushed[0]["total_points"] == 3
    assert len(pushed[0]["manual_positions_3d"]) == 3
