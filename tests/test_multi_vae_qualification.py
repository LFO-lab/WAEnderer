"""Evidence can pass decoding while failing the physical performance gate."""
import json
from pathlib import Path

from eval_scripts.multi_vae_validation_common import PROTOCOL, digest
from stable_audio_wanderer.qualification import evaluate_multi_vae_campaign


def campaign(tmp_path):
    policy = json.loads(PROTOCOL.read_text())
    (tmp_path/'policy.json').write_text(PROTOCOL.read_text())
    context = dict(schema_version=1,vae_id='ear_vae_48k',artifact_identity='artifact',
        artifact_files={'decoder.onnx':'hash'},model_source={'weights_sha256':'weights'},
        corpus_sha256='corpus',policy_sha256=digest(PROTOCOL))
    metrics = dict(rmse=1e-8,max_abs_error=1e-7,snr_db=80)
    probe = dict(input_sha256='raw',cross=metrics,within_onnx=metrics,within_native=metrics)
    files = {name:dict(path=name+'.wav',sha256=name) for name in ('onnx','native')}
    numeric = dict(context,device='cpu',numeric_passed=True,error=None,
        comparisons=[dict(probe,window=w,file_index=0,start=s) for w in policy['windows'] for s in range(4)],
        synthetic=[dict(probe,window=w,seed=s) for w in policy['windows'] for s in range(8)],
        listening_pairs=[dict(probe,file_index=0,files=files,comparison_after_ola=metrics)])
    runtime = dict(context,backend='onnxruntime',device='cpu',audio_device='speakers',
        audio_device_info={'hostapi_name':'Core Audio'},clock='physical output (muted)',navigation='production',
        active_audio_seconds=601,rendered_samples=601*48000,sample_rate=48000,
        nonfinite_pcm=False,error=None,underruns=0,buffer_underruns=0,device_underruns=0,
        preparation_ms=12,process_cpu_percent=80,peak_rss_bytes=1024,
        declared_scenarios=[f'{m}:{s}' for m in ('random','manual','reorganized') for s in [*[f'T{w}' for w in policy['windows']],'adaptive']],
        windows={str(w):{'decode_ms':{'count':10,'median':2,'p95':3,'p99':4}} for w in policy['windows']},
        transitions=[dict(mode=m,step=i,settled_ms=20) for m in ('random','manual','reorganized') for i in range(17)])
    runtime['scenario_samples'] = {s:100 for s in runtime['declared_scenarios']}
    lifecycle = dict(context,passed=True,offline_reload=True,cycles=[dict(backend=b,device='cpu',stopped=True) for b in ('onnxruntime','pytorch')])
    manifest = dict(schema_version=1,protocol='policy.json',claims=[dict(vae_id='ear_vae_48k',backend='onnxruntime',device='cpu',
        claim_realtime=True,numerical_report='numeric.json',runtime_report='runtime.json',lifecycle_report='lifecycle.json',
        listening=[dict(file_index=0,files=files,reviewer='user',date='2026-10-03',observation='similar',new_engine_regression=False)])])
    for name,value in [('numeric',numeric),('runtime',runtime),('lifecycle',lifecycle),('manifest',manifest)]:
        (tmp_path/f'{name}.json').write_text(json.dumps(value))
    return tmp_path/'manifest.json',runtime,manifest


def test_cpu_underrun_preserves_functional_pass(tmp_path):
    path,runtime,_ = campaign(tmp_path)
    assert evaluate_multi_vae_campaign(path)['matrix'][0]['passed']
    runtime.update(underruns=1,buffer_underruns=1)
    (tmp_path/'runtime.json').write_text(json.dumps(runtime))
    row = evaluate_multi_vae_campaign(path)['matrix'][0]
    assert row['functional'] == 'passed' and row['realtime'] == 'failed' and not row['passed']


def test_physical_time_and_artifact_cannot_be_borrowed(tmp_path):
    path,runtime,_ = campaign(tmp_path)
    runtime.update(active_audio_seconds=599,artifact_identity='other-model')
    (tmp_path/'runtime.json').write_text(json.dumps(runtime))
    row = evaluate_multi_vae_campaign(path)['matrix'][0]
    assert row['realtime'] == 'failed' and len(row['reasons']) >= 2


def test_listening_must_identify_the_actual_pairs(tmp_path):
    path,_,manifest = campaign(tmp_path)
    manifest['claims'][0]['listening'][0]['files']['onnx']['sha256'] = 'another-vae'
    path.write_text(json.dumps(manifest))
    assert evaluate_multi_vae_campaign(path)['matrix'][0]['listening'] == 'failed'


def test_quick_diagnostic_cannot_qualify_a_full_campaign(tmp_path):
    path,_,_ = campaign(tmp_path)
    numeric_path = tmp_path/'numeric.json'
    report = json.loads(numeric_path.read_text())
    report['validation_scope'] = 'quick_diagnostic'
    numeric_path.write_text(json.dumps(report))
    row = evaluate_multi_vae_campaign(path)['matrix'][0]
    assert not row['passed']
    assert row['numerical'] == 'diagnostic_passed' and row['functional'] == 'observed'
    assert row['listening'] == 'passed'


def test_later_gpu_policy_preserves_original_cpu_evidence(tmp_path):
    path,_,manifest = campaign(tmp_path)
    later = json.loads((tmp_path/'policy.json').read_text())
    later['native_gpu']['stable_audio_open'] = {'rmse':2e-5,'max_abs_error':2e-4}
    (tmp_path/'later_policy.json').write_text(json.dumps(later))
    manifest['protocol'] = 'later_policy.json'
    manifest['claims'][0]['protocol'] = 'policy.json'
    path.write_text(json.dumps(manifest))
    assert evaluate_multi_vae_campaign(path)['matrix'][0]['numerical'] == 'passed'
