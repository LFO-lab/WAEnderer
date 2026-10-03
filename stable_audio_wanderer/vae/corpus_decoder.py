"""Read corpus decoder requirements without unpickling corpus content."""
from pathlib import Path
import numpy as np


def corpus_decoder_spec(corpus_path, *, validate=True):
    path = Path(corpus_path).expanduser()
    if path.is_dir():
        path /= 'corpus.npz'
    with np.load(path, allow_pickle=False) as data:
        vae_id = str(data['vae_id'].item())
        if vae_id not in ('same_s', 'stable_audio_open', 'ear_vae_44k', 'ear_vae_48k'):
            raise ValueError(f'Unsupported corpus VAE {vae_id!r}')
        spec = dict(vae_id=vae_id, sample_rate=int(data['sr'].item()),
                    latent_hz=float(data['latent_hz'].item()))
        # Optional provenance in newer corpora. Older corpora retain geometry-only
        # matching; do not invent a checkpoint identity they never recorded.
        for source_key, target in (("vae_weights_sha256", "weights_sha256"),
                                   ("vae_config_sha256", "config_sha256"),
                                   ("vae_source_revision", "revision"),
                                   ("vae_effective_config_sha256", "effective_config_sha256"),
                                   ("vae_code_sha256", "code_sha256")):
            if source_key in data:
                spec[target] = str(data[source_key].item())
        if not validate:
            return spec
        if vae_id == 'same_s':
            from .onnx_decoder import validate_same_s_corpus
            validate_same_s_corpus(corpus_path)
            return spec
        z, mean, std = (data[k] for k in ('Z_concat', 'Z_mean', 'Z_std'))
        if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 1:
            raise ValueError('Corpus requires nonempty [frames, latent_dim] latents')
        dim = z.shape[1]
        for name, value in [('Z_concat', z), ('Z_mean', mean), ('Z_std', std)]:
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f'Corpus {name} must be finite float32')
        if mean.shape not in ((dim,), (1, dim)) or std.shape not in ((dim,), (1, dim)) or np.any(std <= 0):
            raise ValueError('Corpus normalization does not match latent dimensions')
        if not np.isfinite(spec['latent_hz']) or spec['latent_hz'] <= 0:
            raise ValueError('Corpus latent_hz must be positive and finite')
        expected_sr = 48000 if vae_id == 'ear_vae_48k' else 44100
        if spec['sample_rate'] != expected_sr:
            raise ValueError(f'{vae_id} requires sample rate {expected_sr}')
        if 'channels' in data and int(data['channels'].item()) != 2:
            raise ValueError('Decoder requires stereo corpus audio')
        spec['latent_dim'] = dim
        return spec
