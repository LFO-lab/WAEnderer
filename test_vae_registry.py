from stable_audio_wanderer.vae import list_vaes


def test_same_s_adapter_is_registered_without_loading_model():
    vaes = {info.vae_id: info for info in list_vaes()}

    assert "same_s" in vaes
    assert vaes["same_s"].sample_rate == 44100
    assert vaes["same_s"].latent_dim == 256
    assert vaes["same_s"].requires_path is False
