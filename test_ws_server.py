import json
import asyncio

import numpy as np
import pytest

pytest.importorskip("websockets")

from stable_audio_wanderer.runtime.ws_server import WSBroadcaster


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
