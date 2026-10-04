"""CPU ONNX decoder for versioned dynamic or fixed-window artifacts."""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import threading
import os
import time
from types import MappingProxyType
import numpy as np
from .decoder_contract import DecodedAudioWindow, DecoderRuntimeError, DecoderWindowMetadata
from .onnx_artifacts import ArtifactError, PROVIDER, inspect_graph, local_path, read_artifact


@dataclass(frozen=True)
class ArtifactDecoderInfo:
    resource_path: Path
    model_path: Path
    backend: str
    provider: str
    device: str
    vae_id: str
    artifact_identity: str
    intra_op_num_threads: int
    graph_optimization: str


class ArtifactOnnxDecoder:
    """Calls and close are serialized; at most two fixed graph sessions are retained."""
    def __init__(self, artifact):
        if artifact.legacy: raise ArtifactError('Use the SAME-S compatibility loader for legacy resources')
        import onnxruntime as ort
        self._ort=ort
        self.artifact=artifact
        artifact.verify_files()
        # Inspect every graph's external references before any ORT session is made.
        for graph in artifact.graphs: inspect_graph(artifact,graph)
        self._lock=threading.RLock()
        self._closed=False
        self._sessions=OrderedDict()
        self._intra_op_threads=min(8, os.cpu_count() or 1)
        self._graphs={w:g for g in artifact.graphs for w in g.windows}
        self._windows=MappingProxyType({w:DecoderWindowMetadata(w,artifact.latent_dim,
            artifact.sample_rate,artifact.channels,artifact.samples_per_latent,w*artifact.samples_per_latent,
            w//2,w//2*artifact.samples_per_latent,'full_overlap_add') for w in artifact.supported_windows})
        self.info=ArtifactDecoderInfo(artifact.root,local_path(artifact.root,artifact.graphs[0].path),
            'onnxruntime',PROVIDER,'cpu',artifact.vae_id,artifact.identity,self._intra_op_threads,'all')
        try:
            # Validate all I/O contracts on load, without a full audio warm-up.
            for graph in artifact.graphs: self._session(graph)
        except Exception:
            self.close()
            raise

    @property
    def supported_windows(self): return self.artifact.supported_windows

    @property
    def default_window(self): return self.artifact.default_window

    def metadata_for(self, window):
        if isinstance(window,(bool,np.bool_)) or not isinstance(window,(int,np.integer)) or window not in self._windows:
            raise DecoderRuntimeError(f'Unsupported ONNX decoder window {window!r}')
        return self._windows[window]

    def validate_corpus(self,spec): self.artifact.validate_corpus(spec)

    def _session(self,graph):
        if graph.path in self._sessions:
            self._sessions.move_to_end(graph.path)
            return self._sessions[graph.path]
        # Immutable exports are rechecked on lazy reload; do not silently use a
        # replacement graph/external weight file after session eviction.
        self.artifact.verify_files()
        inspect_graph(self.artifact,graph)
        if len(self._sessions)>=2: self._sessions.popitem(last=False)
        options=self._ort.SessionOptions()
        options.intra_op_num_threads=self._intra_op_threads
        options.inter_op_num_threads=1
        options.graph_optimization_level=self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Avoid idle spin-wait stealing CPU from navigation and the callback.
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        session=self._ort.InferenceSession(str(local_path(self.artifact.root,graph.path)),
            sess_options=options,providers=[PROVIDER])
        a=self.artifact
        inputs,outputs=session.get_inputs(),session.get_outputs()
        if session.get_providers()!=[PROVIDER] or len(inputs)!=1 or len(outputs)!=1:
            raise ArtifactError('ONNX session requires CPU-only execution and one input/output')
        for tensor,name,width,layout in ((inputs[0],a.input_name,a.latent_dim,a.input_layout),
                                        (outputs[0],a.output_name,a.channels,a.output_layout)):
            axis=2 if layout in ('BDT','BCT') else 1
            shape=tensor.shape
            if tensor.name!=name or tensor.type!='tensor(float)' or len(shape)!=3 or shape[0]!=1 or shape[3-axis]!=width:
                raise ArtifactError(f'ONNX I/O contract mismatch: {name}')
            dim=shape[axis]
            expected=graph.windows[0]*(1 if tensor is inputs[0] else a.samples_per_latent)
            if (graph.dynamic and isinstance(dim,int)) or (not graph.dynamic and dim!=expected):
                raise ArtifactError(f'ONNX time dimension mismatch: {name}')
        self._sessions[graph.path]=session
        return session

    def decode(self,raw_latents):
        raw=np.asarray(raw_latents)
        a=self.artifact
        if raw.dtype!=np.float32 or raw.ndim!=2 or raw.shape[1]!=a.latent_dim or not np.isfinite(raw).all():
            raise DecoderRuntimeError(f'Expected finite float32 latents [T,{a.latent_dim}]')
        meta=self.metadata_for(len(raw))
        start=time.perf_counter()
        with self._lock:
            if self._closed: raise DecoderRuntimeError('ONNX decoder is closed')
            try:
                session=self._session(self._graphs[len(raw)])
                model_input=np.ascontiguousarray(raw.T[None] if a.input_layout=='BDT' else raw[None])
                result=session.run([a.output_name],{a.input_name:model_input})
                if len(result)!=1: raise DecoderRuntimeError('Invalid ONNX result count')
                output=np.asarray(result[0])
                shape=(1,a.channels,meta.audio_window_samples) if a.output_layout=='BCT' else (1,meta.audio_window_samples,a.channels)
                if output.dtype!=np.float32 or output.shape!=shape or not np.isfinite(output).all():
                    raise DecoderRuntimeError(f'Invalid ONNX PCM; expected finite float32 {shape}')
                audio=np.array(output[0].T if a.output_layout=='BCT' else output[0],copy=True,order='C')
            except DecoderRuntimeError: raise
            except Exception as exc: raise DecoderRuntimeError(f'ONNX decode failed for T{len(raw)}: {exc}') from exc
        return DecodedAudioWindow(audio,meta,(time.perf_counter()-start)*1000)

    def close(self):
        with self._lock:
            self._sessions.clear()
            self._closed=True


def load_artifact_decoder(artifact):
    # Bind a prepared selection to the exact bytes checked before teardown.
    current=read_artifact(artifact.root,expected_vae=artifact.vae_id)
    if current.cache_key!=artifact.cache_key: raise ArtifactError('Artifact changed after decoder selection')
    if artifact.legacy:
        from .onnx_decoder import load_same_s_app_decoder
        return load_same_s_app_decoder(resource_dir=artifact.root)
    return ArtifactOnnxDecoder(current)
