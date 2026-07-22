import threading

import numpy as np

from stable_audio_wanderer.runtime.decoder_player import DecoderPlayer, GenerationPcmBuffer


def _stereo(value: float, samples: int) -> np.ndarray:
    return np.full((samples, 2), np.float32(value), dtype=np.float32)


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


def test_legacy_torch_write_frame_keeps_its_adaptive_crossfade_contract():
    # Construct only the pure queue side so the test never opens an audio device.
    player = DecoderPlayer.__new__(DecoderPlayer)
    player.sr = 1000
    player._gain = 1.0
    player.frame_samples = 0
    player.frame_duration = None
    player._crossfade_buffer = None
    player._pcm_buffer = GenerationPcmBuffer()
    player._queue_lock = threading.Lock()

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
