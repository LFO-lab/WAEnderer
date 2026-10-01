import numpy as np
from stable_audio_wanderer.runtime.window_controls import AdaptiveWindow, ManualTexture


def test_adaptation_grows_slowly_and_shortens_on_movement():
    control = AdaptiveWindow()
    control.observe(np.zeros(4), 0)
    assert control.choose(2, [2, 4, 6, 8], 0) == 4
    assert control.choose(4, [2, 4, 6, 8], .1) == 4
    assert control.choose(4, [2, 4, 6, 8], .7) == 6
    control.observe(np.ones(4), .8)
    assert control.choose(8, [2, 4, 6, 8], 1) < 8


def test_sparse_bundle_and_range_remain_authoritative():
    control = AdaptiveWindow()
    assert control.choose(2, [8, 16], 0) in [8, 16]
    assert control.choose(8, [8], 1) == 8


def test_texture_is_continuous_independent_of_batch_size():
    latents = np.arange(48, dtype=np.float32).reshape(12, 4)
    def texture():
        return ManualTexture(latents, np.zeros(4), np.ones(4))
    whole = texture().render(3, 32, 'variation', .3)
    split = texture()
    pieces = np.concatenate([split.render(3, n, 'variation', .3) for n in [4, 12, 16]])
    np.testing.assert_array_equal(whole, pieces)
    assert np.ptp(whole[:, 0]) > 0


def test_held_and_zero_variation_repeat_centre():
    latents = np.arange(48, dtype=np.float32).reshape(12, 4)
    for content in ['held', 'variation']:
        texture = ManualTexture(latents, np.zeros(4), np.ones(4))
        np.testing.assert_array_equal(texture.render(3, 8, content, 0), np.tile(latents[3], (8, 1)))
