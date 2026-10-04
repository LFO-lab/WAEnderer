"""Preparation ownership, reuse, recovery and real tiny-graph publication."""
import io
import json
import shutil
import threading
from types import SimpleNamespace

import pytest
from test_onnx_artifacts import fixture, attest
from stable_audio_wanderer.vae import decoder_preparation as service
from stable_audio_wanderer.vae.onnx_artifacts import publish_artifact, resolve_artifact
from stable_audio_wanderer.runtime import decoder_preparation_jobs as jobs_module
from stable_audio_wanderer.runtime.decoder_preparation_jobs import PreparationJobs, store_lock
from stable_audio_wanderer.runtime.pipeline_server import PipelineManager


def coordinator(tmp_path, **kwargs):
    return PreparationJobs(store_dir=tmp_path/'store',journal_dir=tmp_path/'jobs',**kwargs)


def test_reuse_real_external_artifact_without_native_dependencies(tmp_path, monkeypatch):
    root=tmp_path/'prepared'; fixture(root,external=True)
    installed=publish_artifact(root,store_dir=tmp_path/'store')
    monkeypatch.setattr(jobs_module.subprocess,'Popen',lambda *a,**k:pytest.fail('Reuse must not start exporter'))
    jobs=coordinator(tmp_path)
    jobs.start({'vae_id':'stable_audio_open'})
    result=jobs.wait()
    assert result['status']=='already_valid' and result['artifact_dir']==str(installed)
    assert jobs.path.is_file()
    jobs.close()


def test_worker_failure_preserves_current_and_retry_has_new_identity(tmp_path):
    root=tmp_path/'prepared'; fixture(root)
    installed=publish_artifact(root,store_dir=tmp_path/'store')
    jobs=coordinator(tmp_path,interpreters={'stable_audio_open':'/nonexistent/python'})
    ids=[]
    for _ in range(2):
        jobs.start({'vae_id':'stable_audio_open','force':True})
        result=jobs.wait();ids.append(result['job_id'])
        assert result['status']=='failed'
        assert resolve_artifact('stable_audio_open',store_dir=tmp_path/'store').root==installed
    assert ids[0]!=ids[1] and result['generation']==2
    jobs.close()


def test_busy_and_store_lock_are_enforced(tmp_path,monkeypatch):
    entered=threading.Event(); release=threading.Event()
    def execute(self, request): entered.set();release.wait(5);self._update(status='already_valid')
    monkeypatch.setattr(PreparationJobs,'_execute',execute)
    jobs=coordinator(tmp_path);jobs.start({'vae_id':'same_s'});assert entered.wait(2)
    with pytest.raises(RuntimeError,match='busy'):jobs.start({'vae_id':'same_s'})
    with pytest.raises(BlockingIOError):
        with store_lock(tmp_path/'store','same_s'):pass
    release.set();jobs.wait();jobs.close()


def test_interrupted_journal_requires_explicit_retry(tmp_path):
    directory=tmp_path/'jobs';directory.mkdir()
    (directory/'latest.json').write_text(json.dumps({'job_id':'old','sequence':5,'generation':8,'status':'running'}))
    jobs=coordinator(tmp_path)
    assert jobs.snapshot()['status']=='failed' and 'interrupted' in jobs.snapshot()['detail']
    assert jobs.thread is None
    jobs.close()


@pytest.mark.parametrize('payload',[{'vae_id':'unknown'},{'vae_id':'same_s','interpreter':'sh'},
    {'vae_id':'ear_vae_48k','force':'true'},{'vae_id':'same_s','fixed_only':True}])
def test_request_contract_rejects_unsupported_inputs(payload):
    with pytest.raises(ValueError):service.validate_request(payload)


def test_stronger_evidence_is_not_satisfied_by_existing_artifact(tmp_path,monkeypatch):
    root=tmp_path/'prepared';fixture(root);publish_artifact(root,store_dir=tmp_path/'store')
    probe=tmp_path/'probe.npz';probe.write_bytes(b'new evidence')
    assert service.reusable({'vae_id':'stable_audio_open','store_dir':str(tmp_path/'store'),'fixture':str(probe)}) is None


def test_missing_inputs_reports_without_export(tmp_path):
    result=service.stage({'vae_id':'stable_audio_open'},tmp_path/'output')
    assert result['status']=='missing_inputs'
    assert not (tmp_path/'output').exists()


def test_staged_output_is_published_only_by_coordinator(tmp_path,monkeypatch):
    def popen(args,**kwargs):
        output=__import__('pathlib').Path(args[args.index('--output')+1])
        fixture(output,external=True)
        return SimpleNamespace(stdout=io.StringIO('Validated T2: 1 probes\nWAENDERER_RESULT '+json.dumps({'status':'staged'})+'\n'),
                               wait=lambda **kw:0,poll=lambda:0)
    monkeypatch.setattr(jobs_module.subprocess,'Popen',popen)
    events=[];jobs=coordinator(tmp_path,emit=events.append)
    jobs.start({'vae_id':'stable_audio_open','force':True});result=jobs.wait()
    assert result['status']=='prepared'
    assert resolve_artifact('stable_audio_open',store_dir=tmp_path/'store').identity==result['artifact_identity']
    assert any(e.get('stage')=='validating' for e in events)
    assert any(e.get('stage')=='publishing' for e in events)
    jobs.close()


def test_changed_inputs_cannot_publish(tmp_path,monkeypatch):
    snapshots=iter([{'source':{}},{'source':{'weights_sha256':'new'}}])
    monkeypatch.setattr(jobs_module,'input_snapshot',lambda _:next(snapshots))
    def popen(args,**kwargs):
        output=__import__('pathlib').Path(args[args.index('--output')+1]);fixture(output)
        return SimpleNamespace(stdout=io.StringIO('WAENDERER_RESULT {"status":"staged"}\n'),wait=lambda **k:0,poll=lambda:0)
    monkeypatch.setattr(jobs_module.subprocess,'Popen',popen)
    jobs=coordinator(tmp_path);jobs.start({'vae_id':'stable_audio_open','force':True})
    assert 'Inputs changed' in jobs.wait()['detail']
    assert not (tmp_path/'store/stable_audio_open/current.json').exists()
    jobs.close()


def test_pipeline_rejects_prepare_while_model_owned_and_mutation_during_job(tmp_path,monkeypatch):
    pipeline=PipelineManager(preparation_config={'store_dir':str(tmp_path/'store')})
    events=[];pipeline.set_broadcaster(SimpleNamespace(broadcast_pipeline_message=events.append))
    pipeline.phase='perform'
    pipeline.handle_message({'type':'pipeline_prepare_decoder'})
    assert 'Stop Perform' in events[-1]['error']
    pipeline.phase='idle'
    monkeypatch.setattr(type(pipeline._preparation),'busy',property(lambda _:True))
    for message in ('pipeline_start_perform','pipeline_start_train','pipeline_start_preprocess','pipeline_prepare_decoder'):
        pipeline.handle_message({'type':message})
        assert 'owns the pipeline' in events[-1]['error']
    pipeline.close()


def test_release_failure_prevents_preparation(tmp_path,monkeypatch):
    pipeline=PipelineManager(preparation_config={'store_dir':str(tmp_path/'store')})
    events=[];pipeline.set_broadcaster(SimpleNamespace(broadcast_pipeline_message=events.append))
    monkeypatch.setattr('stable_audio_wanderer.vae.corpus_decoder.corpus_decoder_spec',lambda *a,**k:{'vae_id':'same_s'})
    def fail():raise RuntimeError('cached model release failed')
    monkeypatch.setattr(pipeline,'_release_app_decoder',fail)
    pipeline.handle_message({'type':'pipeline_prepare_decoder','corpus_dir':'corpus'})
    assert 'release failed' in events[-1]['error'] and pipeline._preparation.thread is None
    monkeypatch.setattr(pipeline,'_release_app_decoder',lambda:None);pipeline.close()


def test_same_s_selected_local_corruption_does_not_fallback(tmp_path):
    local=tmp_path/'store/same_s';local.mkdir(parents=True)
    (local/'current.json').write_text(json.dumps({'artifact_id':'f'*64}))
    with pytest.raises(ValueError):resolve_artifact('same_s',store_dir=tmp_path/'store')


def test_shutdown_reaps_worker_without_publication(tmp_path,monkeypatch):
    import subprocess
    import sys
    import time
    original=subprocess.Popen
    entered=threading.Event()
    def popen(*args,**kwargs):
        process=original([sys.executable,'-u','-c','import time; print("Exporting fixture", flush=True); time.sleep(30)'],
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        entered.set()
        return process
    monkeypatch.setattr(jobs_module.subprocess,'Popen',popen)
    jobs=coordinator(tmp_path);jobs.start({'vae_id':'stable_audio_open','force':True})
    assert entered.wait(3)
    start=__import__('time').monotonic();jobs.close()
    assert __import__('time').monotonic()-start < 5
    assert not jobs.thread.is_alive() and jobs.process is None
    assert jobs.snapshot()['status']=='failed'
    assert not (tmp_path/'store/stable_audio_open/current.json').exists()


def test_same_s_staging_preserves_policy_and_complete_source(tmp_path,monkeypatch):
    from stable_audio_wanderer.vae.same_s_preparation import stage_same_s
    from stable_audio_wanderer.vae.same_s_weights import SOURCE_REVISION
    monkeypatch.setattr('stable_audio_wanderer.vae.same_s_weights.resolve_same_s_weights',lambda **k:
        SimpleNamespace(config_sha256='1'*64,model_sha256='2'*64))
    def exporter(output,**kwargs):
        output.mkdir()
        (output/'decoder.onnx').write_bytes(b'test graph; publication separately validates ONNX')
        report={'windows':{str(w):{'output_shape':[1,2,w*4096],'rmse':.001,'snr_db':30} for w in range(2,33,2)}}
        (output/'decoder_parity.json').write_text(json.dumps(report))
        return {'model':'decoder.onnx','source_license':'test license','conversion_description':'test conversion'}
    monkeypatch.setattr('bin.export_web_decoder.export_web_decoder',exporter)
    monkeypatch.setattr('stable_audio_wanderer.vae.same_s_preparation.version',lambda _: 'test-version')
    stage_same_s(tmp_path/'staged')
    manifest=json.loads((tmp_path/'staged/decoder.json').read_text())
    report=json.loads((tmp_path/'staged/parity.json').read_text())
    assert manifest['source']['revision']==SOURCE_REVISION
    assert manifest['supported_windows']==list(range(2,33,2))
    assert report['tolerances']=={'rmse_less_than':.005,'snr_db_greater_than':20}
    assert report['graphs']['decoder.onnx']==manifest['files']['decoder.onnx']
    with pytest.raises(ValueError,match='pinned'):stage_same_s(tmp_path/'bad',revision='bad')


def test_discovery_correlates_same_corpus_source_edits_off_thread(tmp_path,monkeypatch):
    # This protocol fixture must not consult a user's native interpreter registry.
    monkeypatch.setattr('stable_audio_wanderer.vae.native_runtime.native_config',lambda *a:{})
    events=[];done=threading.Event()
    pipeline=PipelineManager(preparation_config={'store_dir':str(tmp_path/'store')})
    pipeline.set_broadcaster(SimpleNamespace(broadcast_pipeline_message=lambda x:(events.append(x),done.set())))
    monkeypatch.setattr('stable_audio_wanderer.vae.corpus_decoder.corpus_decoder_spec',lambda *a,**k:{'vae_id':'ear_vae_44k'})
    monkeypatch.setattr('stable_audio_wanderer.vae.decoder_availability.decoder_availability',lambda *a,**k:[{'model_identity':'source'}])
    pipeline.handle_message({'type':'pipeline_list_decoders','corpus_dir':'corpus','request_id':'client:2','fingerprint':'source-v2'})
    assert done.wait(2)
    assert events[-1]['request_id']=='client:2' and events[-1]['fingerprint']=='source-v2'
    assert events[-1]['model_identity']=='source'
    pipeline.close()


def test_prepare_selects_requested_corpus_in_authoritative_state(tmp_path,monkeypatch):
    pipeline=PipelineManager(preparation_config={'store_dir':str(tmp_path/'store')})
    pipeline._corpus_dir='old'
    events=[];pipeline.set_broadcaster(SimpleNamespace(broadcast_pipeline_message=events.append))
    monkeypatch.setattr('stable_audio_wanderer.vae.corpus_decoder.corpus_decoder_spec',lambda *a,**k:{'vae_id':'same_s'})
    requests=[]
    monkeypatch.setattr(pipeline._preparation,'start',lambda request,**k:requests.append(request))
    pipeline.handle_message({'type':'pipeline_prepare_decoder','corpus_dir':'new','request':{'vae_id':'same_s'}})
    assert events[-1]['corpus_dir']=='new'
    assert requests[0]['compatibility_corpus'].endswith('/new')
    pipeline.close()


def test_batch_continues_after_invalid_model_request(tmp_path,monkeypatch,capsys):
    import bin.prepare_decoders as cli
    class FakeJobs:
        def __init__(self,**kwargs):pass
        def start(self,request):
            self.request=service.validate_request(request)
        def wait(self):return {'vae_id':self.request['vae_id'],'status':'already_valid'}
        def close(self):pass
    monkeypatch.setattr(cli,'PreparationJobs',FakeJobs)
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'models':{'same_s':{'bad_field':True}}}))
    assert cli.main(['--all-installed','--config-file',str(config),'--json'])==1
    results=json.loads(capsys.readouterr().out)
    assert len(results)==4
    assert next(r for r in results if r['vae_id']=='same_s')['status']=='failed'
    assert sum(r['status']=='already_valid' for r in results)==3
