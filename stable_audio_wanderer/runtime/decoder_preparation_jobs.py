"""Single-job offline subprocess coordinator shared by Web and CLI."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import uuid
from ..vae.decoder_preparation import validate_request, input_snapshot, reusable
from ..vae.onnx_artifacts import publish_artifact, read_artifact

TERMINAL = {'prepared','already_valid','missing_weights','missing_dependencies','missing_inputs','busy','failed'}


@contextlib.contextmanager
def store_lock(store, vae):
    directory=Path(store)/vae
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.prepare.lock').open('a+b') as stream:
        try:
            if os.name == 'nt':
                import msvcrt
                stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise BlockingIOError('Another preparation owns this VAE store') from exc
        try: yield
        finally:
            if os.name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream,fcntl.LOCK_UN)


class PreparationJobs:
    def __init__(self, *, interpreters=None, store_dir=None, emit=None, journal_dir=None):
        self.interpreters=dict(interpreters or {})
        self.store=Path(store_dir or Path.home()/'.cache/waenderer/decoders').expanduser().resolve()
        self.journal=Path(journal_dir) if journal_dir else self.store/'jobs'
        self.emit=emit or (lambda state: None)
        self.lock=threading.RLock()
        self.state=None
        self.thread=None
        self.process=None
        self.closed=False
        # Journals are per server instance; recover the latest status on restart.
        self.path=self.journal/'latest.json'
        try:
            self.state=json.loads(self.path.read_text())
            if self.state['status'] not in TERMINAL:
                self.state.update(status='failed',detail='Preparation interrupted by server restart; retry explicitly')
        except (OSError,ValueError,KeyError): self.state=None

    @property
    def busy(self):
        with self.lock: return self.thread is not None and self.thread.is_alive()

    def snapshot(self):
        with self.lock: return dict(self.state) if self.state else None

    def _update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state['sequence']+=1
            result=dict(self.state)
            self.journal.mkdir(parents=True,exist_ok=True)
            temp=self.path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            temp.write_text(json.dumps(result,indent=2)+'\n')
            os.replace(temp,self.path)
        self.emit(result)

    def start(self, request, *, context=None):
        request=validate_request(request)
        with self.lock:
            if self.closed or self.busy: raise RuntimeError('Preparation is busy or closed')
            # Web callers do not supply store/interpreter overrides. CLI can.
            request['store_dir']=str(self.store)
            generation=(self.state or {}).get('generation',0)+1
            self.state=dict(job_id=uuid.uuid4().hex,generation=generation,sequence=0,status='running',stage='checking_inputs',
                            vae_id=request['vae_id'],context=context or {},detail='Checking inputs')
            self.thread=threading.Thread(target=self._run,args=(request,),daemon=True)
            self.thread.start()
            return dict(self.state)

    def _run(self, request):
        try:
            self._update(stage='checking_inputs')
            with store_lock(self.store,request['vae_id']):
                self._execute(request)
        except BlockingIOError as exc: self._update(status='busy',detail=str(exc))
        except Exception as exc:
            self._update(status='failed',detail=f'{type(exc).__name__}: {exc}')

    def _execute(self, request):
        if request.get('weights') and not Path(request['weights']).is_file():
            self._update(status='missing_weights',detail='EAR checkpoint file does not exist'); return
        snapshot=input_snapshot(request)
        self._update(stage='checking_reuse',detail='Checking saved ONNX artifact and supported windows')
        try: artifact=reusable(request)
        except Exception as exc:
            artifact=None
            self._update(detail='Existing artifact cannot satisfy request: '+str(exc))
        if artifact is not None:
            if self.closed or input_snapshot(request)!=snapshot:
                raise ValueError('Preparation interrupted or inputs changed during reuse')
            self._update(status='already_valid',artifact_identity=artifact.identity,artifact_dir=str(artifact.root),detail='Existing artifact verified and runtime checked'); return
        if self.closed: raise RuntimeError('Preparation interrupted')
        interpreter=self.interpreters.get(request['vae_id'],sys.executable)
        project=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix='waenderer-job-') as temp:
            temp=Path(temp); request_path=temp/'request.json'; output=temp/'artifact'
            request_path.write_text(json.dumps(request))
            self._update(stage='loading',detail='Checking exporter environment and native source')
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
            with self.lock:
                if self.closed: raise RuntimeError('Preparation interrupted')
                self.process=subprocess.Popen([interpreter,'-m','bin.decoder_preparation_worker','--request',str(request_path),'--output',str(output)],
                    cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                process=self.process
            result=None
            try:
                for raw in process.stdout:
                    line=raw.strip()[:2000]
                    if line.startswith('WAENDERER_RESULT '): result=json.loads(line.removeprefix('WAENDERER_RESULT '))
                    elif line.startswith('Exporting'):
                        self._update(stage='exporting',detail=line)
                    elif line.startswith('Validated T'):
                        self._update(stage='validating',detail=line)
                    elif line:
                        with self.lock: logs=list(self.state.get('logs',[]))[-39:]+[line]
                        self._update(logs=logs)
                code=process.wait()
            finally:
                process.stdout.close()
                if process.poll() is None: process.kill(); process.wait()
                with self.lock: self.process=None
            if code or not result: raise RuntimeError(f'Exporter exited {code}; see job log')
            if result['status']!='staged': self._update(**result); return
            if input_snapshot(request)!=snapshot: raise ValueError('Inputs changed during export; retry with current source')
            artifact=read_artifact(output,expected_vae=request['vae_id'],expected_source=snapshot['source'])
            # Publication owns this lock. Shutdown cannot cross this boundary.
            with self.lock:
                if self.closed: raise RuntimeError('Preparation interrupted before publication')
                self._update(stage='publishing',detail='Validating and atomically publishing complete artifact')
                path=publish_artifact(output,store_dir=self.store)
            self._update(status='prepared',artifact_identity=artifact.identity,artifact_dir=str(path),detail='ONNX decoder prepared')

    def wait(self):
        with self.lock: thread=self.thread
        if thread: thread.join()
        return self.snapshot()

    def close(self):
        with self.lock:
            self.closed=True
            process=self.process
            if process and process.poll() is None: process.terminate()
            thread=self.thread
        if process:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        if thread and thread is not threading.current_thread(): thread.join()
