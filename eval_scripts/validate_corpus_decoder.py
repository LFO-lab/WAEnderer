"""Exercise a corpus-selected decoder through PipelineManager and real muted audio."""
import argparse
import functools
import json
from pathlib import Path
import time
from unittest.mock import patch
import sounddevice as sd
import torch
from bin.serve import _setup_perform_phase
from stable_audio_wanderer.runtime.pipeline_server import PipelineManager
from stable_audio_wanderer.runtime.ws_server import WSBroadcaster
from eval_scripts.benchmark_dual_inference import ObservedPlayer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--window',type=int,default=8)
    parser.add_argument('--seconds-per-mode',type=float,default=10)
    parser.add_argument('--audio-device',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manager = PipelineManager()
    manager.set_perform_setup_callback(lambda corpus,decoder,config:
        _setup_perform_phase(corpus,decoder,config,WSBroadcaster(),8765))
    factory = functools.partial(sd.OutputStream,device=args.audio_device)
    report = {'corpus':args.corpus,'torch':str(torch.__version__),'device':args.device,
              'window':args.window,'audio_device':args.audio_device,'muted':True,'modes':[]}
    try:
        def player(**kwargs):
            kwargs.update(gain=0.0,blocksize=1024)
            return ObservedPlayer(**kwargs)
        with patch('stable_audio_wanderer.runtime.decoder_player.sd.OutputStream',factory), \
             patch('stable_audio_wanderer.runtime.decoder_player.DecoderPlayer',player):
            manager.handle_message({'type':'pipeline_start_perform','config':{
                'corpus_dir':args.corpus,'decoder_backend':'pytorch','decoder_device':args.device,
                'decoder_window':args.window}})
        if manager.phase != 'perform':
            raise RuntimeError(manager._perform_error or 'Pipeline preparation failed')
        decoder = manager._app_decoder
        metadata = decoder.metadata_for(args.window)
        report.update(vae_id=decoder.info.vae_id,latent_dim=metadata.latent_dim,
            samples_per_latent=metadata.samples_per_latent,supported_windows=list(decoder.supported_windows))
        controller = manager._perform_controller
        for mode in ('wander','manual','reorganized'):
            controller.stop()
            assert controller.set_mode(mode)[0]
            assert controller.start()[0]
            deadline = time.monotonic()+args.seconds_per_mode
            while time.monotonic() < deadline:
                error = controller.get_extra_state()['transport']['error']
                if error: raise RuntimeError(error)
                time.sleep(.1)
            report['modes'].append({'mode':mode,'seconds':args.seconds_per_mode,
                'underruns_total':controller.decoder.underruns,'blocks':controller.decoder.blocks})
        report.update(underruns=controller.decoder.underruns,blocks=controller.decoder.blocks)
        report['passed'] = report['underruns']==0 and report['blocks']>0
    except Exception as exc:
        report.update(passed=False,error=str(exc))
    finally:
        manager.close()
    report['closed'] = manager._app_decoder is None and manager._perform_controller is None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__ == '__main__':main()
