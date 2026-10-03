"""Verify phase 1 compatibility against the unchanged packaged SAME-S graph."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch
from stable_audio_wanderer.vae.decoder_factory import select_decoder, create_decoder
from stable_audio_wanderer.vae.corpus_decoder import corpus_decoder_spec
from stable_audio_wanderer.vae.onnx_artifacts import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=corpus_decoder_spec(args.corpus)
    selection=select_decoder({'decoder_backend':'onnxruntime','decoder_device':'cpu'},corpus_spec=spec)
    graph=selection.artifact.root/selection.artifact.graphs[0].path
    before=sha256(graph)
    model=create_decoder(selection)
    try:
        windows=[]
        for window in model.supported_windows:
            raw=np.zeros((window,256),np.float32)
            result=model.decode(raw)
            assert result.audio.dtype==np.float32 and np.isfinite(result.audio).all()
            assert result.audio.shape==(window*4096,2)
            windows.append({'window':window,'shape':list(result.audio.shape),'decode_ms':result.decode_time_ms})
        options=model._session.get_session_options()
        report={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'compatibility_class':type(model).__name__,'artifact_identity':selection.artifact.identity,
            'legacy_manifest':selection.artifact.legacy,'graph_sha256':before,'source_identity':dict(selection.artifact.source),
            'providers':model._session.get_providers(),
            'intra_threads':options.intra_op_num_threads,'inter_threads':options.inter_op_num_threads,
            'optimization':str(options.graph_optimization_level),'windows':windows}
    finally:model.close()
    report['native_selections']=[]
    devices=['cpu']
    if torch.backends.mps.is_available():devices.append('mps')
    if torch.cuda.is_available():devices.append('cuda:0')
    for device in devices:
        native=select_decoder({'decoder_backend':'pytorch','decoder_device':device},corpus_spec=spec)
        report['native_selections'].append({'backend':native.backend,'device':native.device,'vae_id':native.vae_id})
    report.update(graph_unchanged=sha256(graph)==before,closed=model._session is None,
        scope='Offline real SAME-S ONNX decode and native selection only; no new export or native performance qualification.')
    report['passed']=report['graph_unchanged'] and report['closed'] and report['providers']==['CPUExecutionProvider']
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='windows'},indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
