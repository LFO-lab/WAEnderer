import threading

import numpy as np
import pytest

from stable_audio_wanderer.runtime.decoder_player import DecoderPlayer, GenerationPcmBuffer


def _stereo(value: float, samples: int) -> np.ndarray:
    return np.full((samples, 2), np.float32(value), dtype=np.float32)


def _bare_player(sr: int = 44100) -> DecoderPlayer:
    """Construct only the pure queue side without opening an audio device."""
    player = DecoderPlayer.__new__(DecoderPlayer)
    player.sr = sr
    player._gain = 1.0
    player.frame_samples = 0
    player.frame_duration = None
    player.underruns = player.buffer_underruns = player.device_underruns = 0
    player._crossfade_buffer = None
    player._pcm_buffer = GenerationPcmBuffer()
    player._queue_lock = threading.Lock()
    return player


def test_generation_handoff_crossfades_over_exactly_256_samples():
    buffer = GenerationPcmBuffer(transition_samples=256)
    assert buffer.enqueue(_stereo(1.0, 1024), generation=1)
    initial, underrun = buffer.render(128)
    assert not underrun
    assert np.all(initial == 1.0)

    assert buffer.enqueue(_stereo(3.0, 1024), generation=2)
    transition, underrun = buffer.render(256)
    assert not underrun
    expected = np.linspace(1.0, 3.0, 256, dtype=np.float32)
    assert np.allclose(transition[:, 0], expected, atol=1.0e-6)
    assert np.allclose(transition[:, 1], expected, atol=1.0e-6)
    assert buffer.current_generation == 2
    assert buffer.transition_status == "idle"

    replacement, underrun = buffer.render(128)
    assert not underrun
    assert np.all(replacement == 3.0)


def test_replacement_waits_without_silencing_old_pcm_until_a_full_transition_is_ready():
    buffer = GenerationPcmBuffer(transition_samples=256)
    buffer.enqueue(_stereo(1.0, 768), generation=4)
    buffer.enqueue(_stereo(2.0, 128), generation=5)

    old, underrun = buffer.render(256)
    assert not underrun
    assert np.all(old == 1.0)
    assert buffer.current_generation == 4
    assert buffer.transition_status == "staging"

    buffer.enqueue(_stereo(2.0, 256), generation=5)
    transition, underrun = buffer.render(256)
    assert not underrun
    assert transition[0, 0] == np.float32(1.0)
    assert transition[-1, 0] == np.float32(2.0)
    assert np.all(transition > 0.0)


def test_rapid_generation_changes_are_latest_wins_and_stale_pcm_is_rejected():
    buffer = GenerationPcmBuffer(transition_samples=256)
    buffer.enqueue(_stereo(1.0, 1024), generation=10)
    buffer.enqueue(_stereo(2.0, 1024), generation=11)
    buffer.enqueue(_stereo(4.0, 1024), generation=12)
    assert not buffer.enqueue(_stereo(9.0, 1024), generation=11)

    transition, underrun = buffer.render(256)
    assert not underrun
    assert transition[0, 0] == np.float32(1.0)
    assert transition[-1, 0] == np.float32(4.0)
    assert buffer.current_generation == 12
    after, underrun = buffer.render(128)
    assert not underrun
    assert np.all(after == 4.0)


def test_announcing_latest_generation_discards_staged_pcm_but_keeps_current_audio():
    buffer = GenerationPcmBuffer(transition_samples=256)
    buffer.enqueue(_stereo(1.0, 1024), generation=1)
    buffer.enqueue(_stereo(2.0, 1024), generation=2)

    assert buffer.pending_generation == 2
    assert buffer.request_generation(3)
    assert buffer.current_generation == 1
    assert buffer.pending_generation is None
    assert not buffer.enqueue(_stereo(2.0, 1024), generation=2)

    old, underrun = buffer.render(256)
    assert not underrun
    assert np.all(old == 1.0)

    assert buffer.enqueue(_stereo(3.0, 1024), generation=3)
    replacement, underrun = buffer.render(256)
    assert not underrun
    assert replacement[0, 0] == np.float32(1.0)
    assert replacement[-1, 0] == np.float32(3.0)


def test_transition_with_short_outgoing_tail_is_continuous_and_reports_shortfall():
    buffer = GenerationPcmBuffer(transition_samples=256)
    buffer.enqueue(_stereo(1.0, 100), generation=1)
    buffer.enqueue(_stereo(2.0, 512), generation=2)

    transition, underrun = buffer.render(256)

    assert underrun
    assert np.allclose(
        transition[:, 0],
        np.linspace(1.0, 2.0, 256, dtype=np.float32),
        atol=1.0e-6,
    )
    assert np.max(np.abs(np.diff(transition[:, 0]))) < 0.004
    assert buffer.current_generation == 2


def test_mid_transition_supersession_finishes_ramp_then_uses_latest_generation():
    buffer = GenerationPcmBuffer(transition_samples=256)
    buffer.enqueue(_stereo(1.0, 1024), generation=1)
    buffer.enqueue(_stereo(3.0, 1024), generation=2)

    first_half, underrun = buffer.render(128)
    assert not underrun
    assert buffer.current_generation == 1

    assert buffer.request_generation(3)
    assert buffer.enqueue(_stereo(5.0, 1024), generation=3)
    assert not buffer.enqueue(_stereo(4.0, 1024), generation=2)

    second_half, underrun = buffer.render(128)
    assert not underrun
    assert abs(float(second_half[0, 0] - first_half[-1, 0])) < 0.01
    assert second_half[-1, 0] == np.float32(3.0)
    assert buffer.current_generation == 2
    assert buffer.pending_generation == 3

    latest, underrun = buffer.render(256)
    assert not underrun
    assert latest[0, 0] == np.float32(3.0)
    assert latest[-1, 0] == np.float32(5.0)
    assert buffer.current_generation == 3


def test_reset_prevents_pcm_from_a_previous_run_leaking_into_restart():
    buffer = GenerationPcmBuffer()
    buffer.enqueue(_stereo(1.0, 512), generation=1)
    buffer.render(64)
    buffer.enqueue(_stereo(2.0, 512), generation=2)
    buffer.reset()

    buffer.enqueue(_stereo(7.0, 512), generation=9)
    output, underrun = buffer.render(256)
    assert not underrun
    assert np.all(output == 7.0)
    assert buffer.current_generation == 9
    assert buffer.pending_generation is None


def test_runtime_error_fade_reaches_and_holds_silence():
    buffer = GenerationPcmBuffer(transition_samples=256)
    buffer.enqueue(_stereo(1.0, 1024), generation=1)
    buffer.fade_to_silence(256)

    faded, underrun = buffer.render(256)
    assert not underrun
    assert faded[0, 0] == np.float32(1.0)
    assert faded[-1, 0] == np.float32(0.0)
    assert np.all(np.diff(faded[:, 0]) <= 0.0)
    silent, underrun = buffer.render(128)
    assert underrun
    assert np.all(silent == 0.0)


def test_presentation_cursor_follows_arbitrary_render_partitions_and_boundaries():
    buffer = GenerationPcmBuffer()
    assert buffer.enqueue(
        _stereo(0.0, 3 * 4096),
        generation=7,
        frame_indices=[12, 4, 99],
        samples_per_frame=4096,
    )
    assert buffer.get_presentation_state() is None

    _, underrun = buffer.render(4095)
    assert not underrun
    assert buffer.get_presentation_state() == {
        "index": 12,
        "recent_indices": [12],
        "generation": 7,
        "samples_into_frame": 4095,
        "samples_per_frame": 4096,
    }

    _, underrun = buffer.render(1)
    assert not underrun
    assert buffer.get_presentation_state() == {
        "index": 4,
        "recent_indices": [12, 4],
        "generation": 7,
        "samples_into_frame": 0,
        "samples_per_frame": 4096,
    }

    _, underrun = buffer.render(5000)
    assert not underrun
    assert buffer.get_presentation_state() == {
        "index": 99,
        "recent_indices": [12, 4, 99],
        "generation": 7,
        "samples_into_frame": 904,
        "samples_per_frame": 4096,
    }


def test_presentation_preserves_repeated_and_non_contiguous_provenance():
    buffer = GenerationPcmBuffer()
    assert buffer.enqueue(
        _stereo(0.0, 4 * 64),
        generation=3,
        frame_indices=[5, 5, 1, 9],
        samples_per_frame=64,
    )

    output, underrun = buffer.render(4 * 64)

    assert not underrun
    assert np.all(output == 0.0)
    assert buffer.get_presentation_state() == {
        "index": 9,
        "recent_indices": [5, 5, 1, 9],
        "generation": 3,
        "samples_into_frame": 64,
        "samples_per_frame": 64,
    }


def test_presentation_history_ring_keeps_the_latest_128_exact_indices():
    buffer = GenerationPcmBuffer()
    indices = list(range(200))
    assert buffer.capture_presentation_hold(
        0,
        generation=1,
        samples_per_frame=1,
    )
    assert buffer.enqueue(
        _stereo(0.0, len(indices)),
        generation=1,
        frame_indices=indices,
        samples_per_frame=1,
    )

    _, underrun = buffer.render(len(indices))

    assert not underrun
    state = buffer.get_presentation_state()
    assert state["index"] == 199
    assert state["recent_indices"] == list(range(72, 200))


def test_single_frame_hops_advance_at_exact_buffered_hop_boundaries():
    buffer = GenerationPcmBuffer()
    for index in (40, 41, 42):
        assert buffer.enqueue(
            _stereo(0.0, 4096),
            generation=1,
            frame_indices=[index],
            samples_per_frame=4096,
        )

    buffer.render(4096)
    assert buffer.get_presentation_state() == {
        "index": 41,
        "recent_indices": [40, 41],
        "generation": 1,
        "samples_into_frame": 0,
        "samples_per_frame": 4096,
    }
    buffer.render(4096)
    assert buffer.get_presentation_state() == {
        "index": 42,
        "recent_indices": [40, 41, 42],
        "generation": 1,
        "samples_into_frame": 0,
        "samples_per_frame": 4096,
    }


def test_zero_filled_underrun_does_not_advance_presentation_cursor():
    buffer = GenerationPcmBuffer()
    assert buffer.enqueue(
        _stereo(1.0, 4096),
        generation=2,
        frame_indices=[33],
        samples_per_frame=4096,
    )

    output, underrun = buffer.render(5000)

    assert underrun
    assert np.all(output[:4096] == 1.0)
    assert np.all(output[4096:] == 0.0)
    state = buffer.get_presentation_state()
    assert state["index"] == 33
    assert state["samples_into_frame"] == 4096
    buffer.render(1024)
    assert buffer.get_presentation_state() == state


def test_crossfade_holds_outgoing_cursor_then_promotes_at_incoming_offset():
    buffer = GenerationPcmBuffer(transition_samples=256)
    assert buffer.enqueue(
        _stereo(1.0, 8192),
        generation=1,
        frame_indices=[10, 11],
        samples_per_frame=4096,
    )
    buffer.render(1000)
    assert buffer.enqueue(
        _stereo(2.0, 8192),
        generation=2,
        frame_indices=[20, 21],
        samples_per_frame=4096,
    )

    buffer.render(128)
    assert buffer.get_presentation_state() == {
        "index": 10,
        "recent_indices": [10],
        "generation": 1,
        "samples_into_frame": 1128,
        "samples_per_frame": 4096,
    }

    buffer.render(128)
    assert buffer.get_presentation_state() == {
        "index": 20,
        "recent_indices": [10, 20],
        "generation": 2,
        "samples_into_frame": 256,
        "samples_per_frame": 4096,
    }


def test_crossfade_promotion_wins_an_exact_outgoing_frame_boundary():
    buffer = GenerationPcmBuffer(transition_samples=256)
    assert buffer.enqueue(
        _stereo(1.0, 8192),
        generation=1,
        frame_indices=[10, 11],
        samples_per_frame=4096,
    )
    buffer.render(4096 - 256)
    assert buffer.enqueue(
        _stereo(2.0, 8192),
        generation=2,
        frame_indices=[20, 21],
        samples_per_frame=4096,
    )

    buffer.render(256)

    assert buffer.get_presentation_state() == {
        "index": 20,
        "recent_indices": [10, 20],
        "generation": 2,
        "samples_into_frame": 256,
        "samples_per_frame": 4096,
    }


def test_stale_and_superseded_generations_cannot_enter_presentation_history():
    buffer = GenerationPcmBuffer(transition_samples=256)
    assert buffer.enqueue(
        _stereo(1.0, 4096),
        generation=1,
        frame_indices=[1],
        samples_per_frame=4096,
    )
    buffer.render(100)
    assert buffer.enqueue(
        _stereo(2.0, 4096),
        generation=2,
        frame_indices=[2],
        samples_per_frame=4096,
    )
    assert buffer.request_generation(3)
    assert not buffer.enqueue(
        _stereo(2.0, 4096),
        generation=2,
        frame_indices=[2],
        samples_per_frame=4096,
    )
    assert buffer.get_presentation_state()["recent_indices"] == [1]

    assert buffer.enqueue(
        _stereo(3.0, 4096),
        generation=3,
        frame_indices=[3],
        samples_per_frame=4096,
    )
    assert buffer.get_presentation_state()["recent_indices"] == [1]
    buffer.render(256)
    assert buffer.get_presentation_state() == {
        "index": 3,
        "recent_indices": [1, 3],
        "generation": 3,
        "samples_into_frame": 256,
        "samples_per_frame": 4096,
    }


def test_mid_crossfade_latest_wins_keeps_each_audible_generation_cursor():
    buffer = GenerationPcmBuffer(transition_samples=256)
    for generation, index in ((1, 10), (2, 20)):
        assert buffer.enqueue(
            _stereo(float(generation), 4096),
            generation=generation,
            frame_indices=[index],
            samples_per_frame=4096,
        )
    buffer.render(128)
    assert buffer.get_presentation_state()["generation"] == 1

    assert buffer.request_generation(3)
    assert buffer.enqueue(
        _stereo(3.0, 4096),
        generation=3,
        frame_indices=[30],
        samples_per_frame=4096,
    )
    buffer.render(128)
    assert buffer.get_presentation_state()["generation"] == 2
    assert buffer.get_presentation_state()["samples_into_frame"] == 256

    buffer.render(256)
    assert buffer.get_presentation_state() == {
        "index": 30,
        "recent_indices": [10, 20, 30],
        "generation": 3,
        "samples_into_frame": 256,
        "samples_per_frame": 4096,
    }


def test_reset_and_prebuffer_hold_the_last_audible_cursor():
    buffer = GenerationPcmBuffer()
    assert buffer.capture_presentation_hold(
        7, generation=1, samples_per_frame=4096
    )
    initial = buffer.get_presentation_state()
    assert initial == {
        "index": 7,
        "recent_indices": [7],
        "generation": 1,
        "samples_into_frame": 0,
        "samples_per_frame": 4096,
    }
    assert not buffer.capture_presentation_hold(
        99, generation=2, samples_per_frame=4096
    )

    assert buffer.enqueue(
        _stereo(1.0, 4096),
        generation=1,
        frame_indices=[7],
        samples_per_frame=4096,
    )
    assert buffer.get_presentation_state() == initial
    buffer.render(100)
    assert buffer.get_presentation_state()["recent_indices"] == [7]
    audible = buffer.get_presentation_state()

    buffer.reset()
    assert buffer.get_presentation_state() == audible
    assert buffer.enqueue(
        _stereo(2.0, 4096),
        generation=2,
        frame_indices=[22],
        samples_per_frame=4096,
    )
    assert buffer.get_presentation_state() == audible


def test_fade_post_zeroing_does_not_advance_presentation_cursor():
    buffer = GenerationPcmBuffer()
    assert buffer.enqueue(
        _stereo(1.0, 8192),
        generation=1,
        frame_indices=[8, 9],
        samples_per_frame=4096,
    )
    buffer.fade_to_silence(256)

    output, underrun = buffer.render(512)

    assert not underrun
    assert np.all(output[256:] == 0.0)
    assert buffer.get_presentation_state() == {
        "index": 8,
        "recent_indices": [8],
        "generation": 1,
        "samples_into_frame": 256,
        "samples_per_frame": 4096,
    }
    buffer.render(512)
    assert buffer.get_presentation_state()["samples_into_frame"] == 256


def test_fade_completion_wins_an_exact_frame_boundary():
    buffer = GenerationPcmBuffer()
    assert buffer.enqueue(
        _stereo(1.0, 8192),
        generation=1,
        frame_indices=[8, 9],
        samples_per_frame=4096,
    )
    buffer.render(4096 - 256)
    buffer.fade_to_silence(256)

    buffer.render(256)

    assert buffer.get_presentation_state() == {
        "index": 8,
        "recent_indices": [8],
        "generation": 1,
        "samples_into_frame": 4096,
        "samples_per_frame": 4096,
    }


def test_fade_completion_does_not_promote_an_inaudible_crossfade_tail():
    buffer = GenerationPcmBuffer(transition_samples=256)
    assert buffer.enqueue(
        _stereo(1.0, 4096),
        generation=1,
        frame_indices=[10],
        samples_per_frame=4096,
    )
    assert buffer.enqueue(
        _stereo(2.0, 4096),
        generation=2,
        frame_indices=[20],
        samples_per_frame=4096,
    )
    buffer.render(128)
    assert buffer.get_presentation_state()["generation"] == 1

    buffer.fade_to_silence(64)
    output, _ = buffer.render(128)

    assert np.all(output[64:] == 0.0)
    assert buffer.get_presentation_state() == {
        "index": 10,
        "recent_indices": [10],
        "generation": 1,
        "samples_into_frame": 192,
        "samples_per_frame": 4096,
    }


@pytest.mark.parametrize("fade_samples", [300, 512])
def test_fade_completion_promotes_a_crossfade_completed_before_silence(
    fade_samples,
):
    buffer = GenerationPcmBuffer(transition_samples=256)
    assert buffer.enqueue(
        _stereo(1.0, 4096),
        generation=1,
        frame_indices=[10],
        samples_per_frame=4096,
    )
    assert buffer.enqueue(
        _stereo(2.0, 4096),
        generation=2,
        frame_indices=[20],
        samples_per_frame=4096,
    )
    buffer.fade_to_silence(fade_samples)

    output, _ = buffer.render(512)

    assert np.all(output[fade_samples:] == 0.0)
    assert buffer.get_presentation_state() == {
        "index": 20,
        "recent_indices": [10, 20],
        "generation": 2,
        "samples_into_frame": fade_samples,
        "samples_per_frame": 4096,
    }


def test_write_hop_validates_and_accepts_pcm_and_provenance_atomically():
    player = _bare_player()
    player.frame_samples = 123
    assert player.request_generation(2)

    assert not player.write_hop(
        _stereo(1.0, 4096),
        generation=1,
        frame_indices=[1],
        samples_per_frame=4096,
    )
    assert player.frame_samples == 123
    assert player.generation_buffer_duration(1) == 0.0
    assert player.get_presentation_state() is None

    with pytest.raises(ValueError, match="PCM/provenance length mismatch"):
        player.write_hop(
            _stereo(2.0, 4096),
            generation=2,
            frame_indices=[2, 3],
            samples_per_frame=4096,
        )
    with pytest.raises(ValueError, match="provided together"):
        player.write_hop(
            _stereo(2.0, 4096),
            generation=2,
            frame_indices=[2],
        )
    assert player.generation_buffer_duration(2) == 0.0

    assert player.write_hop(
        _stereo(2.0, 4096),
        generation=2,
        frame_indices=[2],
        samples_per_frame=4096,
    )
    assert player.frame_samples == 4096
    assert player.frame_duration == pytest.approx(4096 / 44100)
    assert player.get_presentation_state() is None


def test_legacy_torch_write_frame_keeps_its_adaptive_crossfade_contract():
    player = _bare_player(sr=1000)

    player.write_frame(_stereo(1.0, 1000))
    assert player.frame_samples == 1000
    assert player.buffer_duration() == 0.75
    player.write_frame(_stereo(2.0, 1000))
    assert player.buffer_duration() == 1.5

    queued, underrun = player._pcm_buffer.render(1500)
    assert not underrun
    assert np.all(queued[:750] == 1.0)
    assert queued[750, 0] == np.float32(1.0)
    assert queued[999, 0] == np.float32(2.0)
    assert np.all(queued[1000:] == 2.0)
    assert player.get_presentation_state() is None


def test_soft_handoff_refills_only_the_current_generation():
    buffer = GenerationPcmBuffer(transition_samples=4)
    assert buffer.enqueue(_stereo(1, 8), generation=1)
    buffer.render(4)
    assert buffer.request_generation(2, continue_current=True)
    assert buffer.enqueue(_stereo(1, 8), generation=1)
    assert not buffer.enqueue(_stereo(9, 8), generation=0)
    out, underrun = buffer.render(8)
    assert not underrun and np.all(out == 1)
    assert buffer.enqueue(_stereo(2, 16), generation=2)
    assert buffer.enqueue(_stereo(1, 8), generation=1)
    buffer.render(4)
    assert buffer.current_generation == 2
    assert not buffer.enqueue(_stereo(1, 8), generation=1)


def test_hard_reset_revokes_soft_handoff_continuation():
    player = _bare_player()
    assert player.write_hop(_stereo(1, 1024), generation=1)
    assert player.request_generation(2, continue_current=True)
    assert player.write_hop(_stereo(1, 1024), generation=1)
    assert player.request_generation(3)
    assert not player.write_hop(_stereo(1, 1024), generation=1)
    assert not player.write_hop(_stereo(2, 1024), generation=2)
    assert player.write_hop(_stereo(3, 1024), generation=3)


def test_callback_distinguishes_device_underflow_from_pcm_starvation():
    from types import SimpleNamespace
    player = _bare_player()
    player.underruns = player.buffer_underruns = player.device_underruns = 0
    player.write_hop(_stereo(1, 512), generation=1)
    output = np.empty((512,2), dtype=np.float32)
    player._callback(output, 512, None, SimpleNamespace(output_underflow=True))
    assert (player.underruns, player.device_underruns, player.buffer_underruns) == (1,1,0)
    player._callback(output, 512, None, SimpleNamespace(output_underflow=False))
    assert (player.underruns, player.device_underruns, player.buffer_underruns) == (2,1,1)


def test_blocking_stream_reports_device_underflow():
    from types import SimpleNamespace
    player = _bare_player()
    player.underruns = player.buffer_underruns = player.device_underruns = 0
    player._running = threading.Event()
    player._running.set()
    player._blocksize = 512
    player.write_hop(_stereo(1,512), generation=1)
    def write(_):
        player._running.clear()
        return True
    player._stream = SimpleNamespace(write=write)
    player._blocking_writer_loop()
    assert (player.underruns, player.device_underruns, player.buffer_underruns) == (1,1,0)
