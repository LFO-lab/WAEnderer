"""Fixed local RPC entrypoint; native inference owns a separate GPU process."""
import base64
from dataclasses import asdict
import json
from pathlib import Path
import sys


def send(stream, value):
    stream.write(json.dumps(value)+'\n')
    stream.flush()


def main():
    # Model libraries print progress to stdout; reserve original stdout for RPC.
    wire = sys.stdout
    sys.stdout = sys.stderr
    operation = sys.argv[1]
    if operation == 'availability':
        payload = json.load(sys.stdin)
        from .decoder_availability import decoder_availability
        send(wire, decoder_availability(**payload))
        return
    payload = json.load(sys.stdin) if operation == 'select' else json.loads(sys.stdin.readline())
    from .decoder_factory import select_decoder, create_decoder
    selection = select_decoder(payload['config'], corpus_spec=payload['corpus_spec'])
    if operation == 'select':
        fields = asdict(selection)
        fields.pop('artifact')
        send(wire, fields)
        return
    if operation != 'serve':
        raise ValueError('Unknown native worker operation')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    decoder = create_decoder(selection)
    try:
        if hasattr(decoder,'validate_corpus'):
            decoder.validate_corpus(payload['corpus_spec'])
        else:
            spec = payload['corpus_spec']
            meta = decoder.metadata_for(decoder.default_window)
            if spec['vae_id'] != decoder.info.vae_id or spec['sample_rate'] != meta.sample_rate or spec.get('latent_dim',meta.latent_dim) != meta.latent_dim:
                raise ValueError('Corpus does not match native decoder')
        info = dict(vars(decoder.info))
        # Only public transport/source identity fields cross the process boundary.
        info = {k:info[k] for k in ('backend','provider','device','vae_id','model_path',
            'config_sha256','model_sha256','source_revision') if k in info}
        info = {k:str(v) if isinstance(v,Path) else v for k,v in info.items()}
        send(wire, dict(info=info, default_window=decoder.default_window,
            windows={str(w):asdict(decoder.metadata_for(w)) for w in decoder.supported_windows}))
        for line in sys.stdin:
            request = json.loads(line)
            if request['operation'] == 'close':
                break
            try:
                raw = np.frombuffer(base64.b64decode(request['raw']),np.float32).reshape(request['shape']).copy()
                decoded = decoder.decode(raw)
                send(wire, dict(audio=base64.b64encode(decoded.audio.tobytes()).decode('ascii'),
                    shape=list(decoded.audio.shape), decode_time_ms=decoded.decode_time_ms))
            except Exception as exc:
                send(wire, dict(error=f'{type(exc).__name__}: {exc}'))
    finally:
        decoder.close()


if __name__ == '__main__':
    main()
