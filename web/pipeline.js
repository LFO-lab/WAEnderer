/**
 * Pipeline frontend logic for Stable Audio Wanderer.
 * Manages preprocess/train/perform tab UI and WebSocket pipeline messages.
 */

// Pipeline state
let pipelinePhase = 'idle';
let pipelineCorpusDir = null;
let selectedAudioDir = '';
const corpusMetadata = new Map();
let selectedTab = 'preprocess';
let encoderAvailability = {};
let encoderListSeen = false;

function encoderReady() {
    const id = document.getElementById('pp-vae-select')?.value;
    return encoderAvailability[id]?.ready === true;
}
let pipelineServerSeen = false;
const APP_DECODER_WINDOWS = [2]; // Backend metadata supplies validated lengths after loading.
let decoderWindowControlsAvailable = false;
let decoderWindowMode = 'fixed';
const editingDecoderControls = new Set();
const pendingDecoderControls = new Map();

let decoderChoices = null;
let decoderChoiceCorpus = null;
let decoderInitialWindow = 2;
let decoderCommandPending = false;
let pipelineConnectionReady = true;
let activePipelineDecoder = null;
let decoderRequestNumber = 0;
let decoderLatestRequest = null;
let decoderFingerprint = null;
let decoderModelIdentity = null;
let decoderServerNamespace = 'legacy';
let decoderVae = null;
let preparationSupported = false;
let preparationJob = null;
let preparationRequestPending = false;
let decoderSourceTimer = null;
const decoderClientId = Math.random().toString(36).slice(2);
const decoderPreferences = new Map();

function decoderSources() {
    const value = id => document.getElementById(id)?.value.trim() || '';
    return {vae_weight_path:value('perform-vae-weight-path'), vae_repo_path:value('perform-vae-repo-path'),
        vae_config_path:value('perform-vae-config-path')};
}
function currentDecoderFingerprint() {
    return JSON.stringify({corpus_dir:pipelineCorpusDir,...decoderSources()});
}
function decoderPreferenceKey() {
    return decoderModelIdentity ? `waenderer.decoder.v1:${decoderServerNamespace}:${pipelineCorpusDir}:${decoderModelIdentity}` : null;
}
function savedDecoderPreference() {
    const key = decoderPreferenceKey();
    if (!key) return null;
    try { return localStorage.getItem(key) || decoderPreferences.get(key); }
    catch (_) { return decoderPreferences.get(key); }
}
function saveDecoderPreference() {
    const key=decoderPreferenceKey();
    const value=document.getElementById('perform-decoder-engine')?.value;
    if (!key || !value) return;
    decoderPreferences.set(key,value);
    try { localStorage.setItem(key,value); } catch (_) {}
}
function preparationBusy() {
    return preparationRequestPending || preparationJob?.status === 'running';
}
function onPreparationJob(job) {
    if (!job) return;
    if ((job.generation || 0) < (preparationJob?.generation || 0)) return;
    if (preparationJob?.job_id === job.job_id && job.sequence <= preparationJob.sequence) return;
    const changed=preparationJob?.job_id !== job.job_id || preparationJob?.status !== job.status;
    preparationJob=job;
    preparationRequestPending=false;
    const element=document.getElementById('perform-prepare-status');
    const statusLabels={running:'Preparing',prepared:'Ready',already_valid:'Ready',failed:'Failed',missing_weights:'Weights required',missing_dependencies:'Export dependencies required',missing_inputs:'Inputs required',busy:'Busy'};
    if (element) element.textContent=`${job.vae_id}: ${statusLabels[job.status] || job.status} · ${job.detail || ''}`;
    if (changed && job.status !== 'running' && pipelineCorpusDir) requestDecoderAvailability();
    updateDecoderControls();
}


function selectedDecoderChoice() {
    const value = document.getElementById('perform-decoder-engine')?.value ?? '';
    const [backend, device] = value.split('|');
    return {backend, device};
}

function updateDecoderAvailability() {
    const element = document.getElementById('perform-decoder-availability');
    const selected = selectedDecoderChoice();
    const choice = decoderChoices?.find(item => item.backend === selected.backend && item.device === selected.device);
    if (element) element.textContent = choice
        ? `Native source: ${choice.native_source_present ?? choice.weights ? 'present' : 'missing'} · ONNX artifact: ${choice.artifact_verified ? 'verified' : 'unavailable'}. ${choice.detail}`
        : 'Choose a decoder after checking availability.';
}

function onDecoderList(data) {
    if (data.corpus_dir && data.corpus_dir !== pipelineCorpusDir) return;
    if (data.request_id && (data.request_id !== decoderLatestRequest || data.fingerprint !== currentDecoderFingerprint())) return;
    if (data.server_namespace) decoderServerNamespace=data.server_namespace;
    if ('preparation_supported' in data) preparationSupported=data.preparation_supported;
    const identityChanged=decoderModelIdentity !== (data.model_identity || null);
    decoderModelIdentity=data.model_identity || null;
    decoderVae=data.vae_id || 'same_s';
    const modelLabel=document.getElementById('perform-decoder-model');
    if (modelLabel) modelLabel.textContent=formatCorpusVae(data.vae_id || corpusMetadata.get(pipelineCorpusDir)?.vae_id);
    const summaryLabel = document.getElementById('perform-corpus-summary');
    if (summaryLabel) summaryLabel.textContent = pipelineCorpusDir
        ? formatCorpusAudioSummary(data.corpus_audio || corpusMetadata.get(pipelineCorpusDir)?.corpus_audio)
            || 'Corpus details unavailable. Restart the server and refresh the corpus list.'
        : '';
    if (data.corpus_dir && data.corpus_dir !== pipelineCorpusDir) return;
    if (pipelineCorpusDir && !data.corpus_dir && decoderChoiceCorpus) return;
    const corpusChanged = Boolean(data.corpus_dir && data.corpus_dir !== decoderChoiceCorpus);
    if (data.corpus_dir) decoderChoiceCorpus = data.corpus_dir;
    if (corpusChanged) {
        decoderInitialWindow = data.default_window || 2;
        if (pipelinePhase === 'idle') populateDecoderWindowOptions([decoderInitialWindow], decoderInitialWindow);
    }
    const weightGroup = document.getElementById('perform-vae-weight-group');
    if (weightGroup) weightGroup.hidden = !String(data.vae_id || '').startsWith('ear_');
    decoderChoices = data.decoders || [];
    const select = document.getElementById('perform-decoder-engine');
    if (select) {
        let previous = select.value || 'onnxruntime|cpu';
        if (pipelinePhase !== 'perform') {
            const saved=savedDecoderPreference();
            if (saved) previous=saved;
            else if (corpusChanged || identityChanged || !previous) {
                const preferred=decoderChoices.find(c => c.backend === 'onnxruntime' && c.device === 'cpu' && c.selectable && c.artifact_verified !== false);
                previous=preferred ? 'onnxruntime|cpu' : '';
            }
        }
        select.innerHTML = '';
        const placeholder=document.createElement('option');
        placeholder.value=''; placeholder.textContent='Choose a decoder'; placeholder.disabled=true;
        select.appendChild(placeholder);
        decoderChoices.forEach(choice => {
            const option = document.createElement('option');
            option.value = `${choice.backend}|${choice.device}`;
            option.textContent = choice.label + (choice.selectable ? '' : ' — unavailable');
            option.disabled = !choice.selectable;
            option.title=choice.detail || '';
            select.appendChild(option);
        });
        // Never silently replace a requested or active device with another one.
        if (!Array.from(select.options).some(option => option.value === previous)) {
            const option = document.createElement('option');
            option.value = previous; option.textContent = previous + ' — unavailable'; option.disabled = true;
            select.appendChild(option);
        }
        select.value = previous;
    }
    const unavailable=document.getElementById('perform-decoder-unavailable');
    if (unavailable) {
        unavailable.innerHTML='';
        decoderChoices.filter(c=>!c.selectable).forEach(choice=>{
            const row=document.createElement('li');
            row.textContent=`${choice.label}: ${choice.detail}`;
            unavailable.appendChild(row);
        });
    }
    try { if (pipelineCorpusDir) localStorage.setItem(`waenderer.corpus.v1:${decoderServerNamespace}`,pipelineCorpusDir); } catch (_) {}
    updateDecoderAvailability();
    if (data.error) setDecoderStatus(`Error: ${data.error}`, 'error');
    updateDecoderControls();
}

function pipelineDisconnected() {
    pipelineConnectionReady = false;
    decoderCommandPending = false;
    decoderWindowControlsAvailable = false;
    activePipelineDecoder = null;
    preparationRequestPending=false;
    if (typeof updateDecoderRuntimeDisplay === 'function') updateDecoderRuntimeDisplay({});
    setDecoderStatus('Disconnected — active decoder unknown. Reconnect before changing it.', 'error');
    updateDecoderControls();
}

function applyDecoderPhase(data) {
    pipelineConnectionReady = true;
    decoderCommandPending = false;
    activePipelineDecoder = pipelinePhase === 'perform' ? (data.decoder || activePipelineDecoder) : null;
    if (activePipelineDecoder) {
        const decoder = activePipelineDecoder;
        const select = document.getElementById('perform-decoder-engine');
        const device = decoder.device || (decoder.backend === 'onnxruntime' ? 'cpu' : decoder.provider);
        const value = `${decoder.backend}|${device === 'mps:0' ? 'mps' : device}`;
        if (select) {
            if (!Array.from(select.options).some(option => option.value === value)) {
                const option = document.createElement('option'); option.value = value; option.textContent = value;
                select.appendChild(option);
            }
            select.value = value;
        }
        setDecoderStatus(`Active: ${decoder.vae_id || decoderVae || ''} · ${decoder.backend} · ${device} · runtime validated`, 'ok');
    } else if (pipelinePhase === 'preparing') {
        setDecoderStatus('Loading and warming decoder… No active performance.');
    } else if (pipelinePhase === 'stopping') {
        setDecoderStatus('Stopping audio and waiting for pending decoding…');
    } else {
        setDecoderStatus('No active decoder. Choose a decoder and load corpus.');
        if (pipelinePhase === 'idle') populateDecoderWindowOptions([decoderInitialWindow], decoderInitialWindow);
        pendingDecoderControls.clear();
    }
    if (!activePipelineDecoder && typeof updateDecoderRuntimeDisplay === 'function') updateDecoderRuntimeDisplay({});
    if (data.error) setDecoderStatus(`Error: ${data.error}`, 'error');
    updateDecoderAvailability();
}

function syncDecoderControl(element, value) {
    if (!element) return;
    const incoming = String(value);
    const pending = pendingDecoderControls.get(element.id);
    if (pending && pending.value === incoming) pendingDecoderControls.delete(element.id);
    else if (pending && Date.now() < pending.until) return;
    else pendingDecoderControls.delete(element.id);
    // Native select menus and keyboard selection must survive streaming state updates.
    if (editingDecoderControls.has(element.id)) return;
    if (element.value !== incoming) element.value = incoming;
}

function noteDecoderControlChange(element) {
    pendingDecoderControls.set(element.id, {value: element.value, until: Date.now() + 1500});
}


// Training history for chart rendering
let trainingHistory = {
    random: { train_loss: [], val_loss: [], window_mae: [] },
    reorganized: { train_loss: [], val_loss: [], accuracy: [] },
};

// ---- Tab Navigation ----

function selectTab(tab) {
    selectedTab = tab;
    updatePipelinePhaseUI();
}

function setPipelineCorpusDir(corpusDir) {
    if (!corpusDir || corpusDir === pipelineCorpusDir) return;
    pipelineCorpusDir = corpusDir;
    const modelLabel = document.getElementById('perform-decoder-model');
    if (modelLabel) modelLabel.textContent = corpusMetadata.get(corpusDir)?.vae_id
        ? formatCorpusVae(corpusMetadata.get(corpusDir).vae_id) : 'Checking corpus…';
    const summaryLabel = document.getElementById('perform-corpus-summary');
    if (summaryLabel) summaryLabel.textContent = formatCorpusAudioSummary(corpusMetadata.get(corpusDir)?.corpus_audio)
        || 'Reading corpus details…';
    const select = document.getElementById('perform-corpus-select');
    if (select) select.value = corpusDir;
    decoderChoices = [];
    requestDecoderAvailability();
    updateDecoderControls();
}

function formatCorpusVae(vaeId) {
    return vaeId ? `VAE: ${vaeId === 'same_s' ? 'same-s' : vaeId}` : 'VAE: unavailable';
}

function formatCorpusAudioSummary(summary) {
    if (!summary) return '';
    const parts = [];
    const duration = summary.encoded_duration_seconds;
    if (typeof duration === 'number' && Number.isFinite(duration) && duration >= 0) {
        const seconds = Math.round(duration);
        const hh = String(Math.floor(seconds / 3600)).padStart(2, '0');
        const mm = String(Math.floor(seconds / 60) % 60).padStart(2, '0');
        const ss = String(seconds % 60).padStart(2, '0');
        parts.push(`Encoded audio: ≈ ${hh}:${mm}:${ss}`);
    }
    if (Number.isInteger(summary.audio_file_count)) parts.push(`${summary.audio_file_count} files`);
    if (summary.sample_rate) parts.push(`${summary.sample_rate / 1000} kHz`);
    return parts.join(' · ');
}

function normalizeDecoderWindows(values) {
    if (!Array.isArray(values)) return [];
    return [...new Set(values.map(value => {
        const parsed = Number(String(value).replace(/^T/i, ''));
        return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
    }).filter(value => value !== null))].sort((a, b) => a - b);
}

function setDecoderStatus(message, kind = 'neutral') {
    const status = document.getElementById('perform-decoder-status');
    if (!status) return;
    status.textContent = message;
    status.style.color = kind === 'ok' ? '#4aff6a' : (kind === 'error' ? '#ff6b6b' : '#888');
}

function populateDecoderWindowOptions(windows, selectedWindow = null) {
    const select = document.getElementById('perform-decoder-window');
    const display = document.getElementById('val-decoder-window');
    if (!select) return;

    const normalized = normalizeDecoderWindows(windows);
    select.innerHTML = '';
    if (normalized.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'Decoder unavailable';
        select.appendChild(option);
        if (display) display.textContent = '--';
        return;
    }

    normalized.forEach(windowSize => {
        const option = document.createElement('option');
        option.value = String(windowSize);
        option.textContent = `T${windowSize}`;
        select.appendChild(option);
    });

    const requested = Number(selectedWindow);
    const selected = normalized.includes(requested)
        ? requested
        : (normalized.includes(2) ? 2 : normalized[0]);
    select.value = String(selected);
    if (display) display.textContent = `T${selected}`;
}

function pipelineCanStartTransport() {
    // The shared page must remain usable with standalone perform.py, which
    // never emits pipeline_* messages.  Once the unified server identifies
    // itself, decoding is unavailable until its validated handoff completes.
    return !pipelineServerSeen || (pipelineConnectionReady && !decoderCommandPending && pipelinePhase === 'perform');
}

function updateDecoderControls() {
    const windowSelect = document.getElementById('perform-decoder-window');
    const corpusToggle = document.getElementById('perform-corpus-toggle');
    const performanceLoaded = pipelinePhase === 'perform' || decoderWindowControlsAvailable;

    if (windowSelect) {
        windowSelect.disabled = !pipelineConnectionReady || decoderCommandPending || (!performanceLoaded && (!pipelineCorpusDir || pipelinePhase !== 'idle'))
            || (decoderWindowControlsAvailable && decoderWindowMode === 'adaptive');
    }
    ['window-mode', 'window-minimum', 'window-maximum', 'manual-content', 'manual-variation'].forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.disabled = !pipelineConnectionReady || decoderCommandPending || !decoderWindowControlsAvailable;
            element.title = decoderWindowControlsAvailable ? '' : 'Load corpus with the updated server to enable this control';
        }
    });
    const requested = selectedDecoderChoice();
    const available = decoderChoices?.find(item => item.backend === requested.backend && item.device === requested.device);
    const ready = pipelineConnectionReady && !decoderCommandPending && !preparationBusy();
    for (const id of ['pp-start-btn','train-start-btn','perform-corpus-select']) {
        const button=document.getElementById(id);
        if (button) button.disabled=!ready || pipelinePhase !== 'idle' || (id === 'train-start-btn' && !pipelineCorpusDir) || (id === 'pp-start-btn' && !encoderReady());
    }
    const engine = document.getElementById('perform-decoder-engine');
    if (engine) engine.disabled = !ready || pipelinePhase !== 'idle';
    for (const id of ['perform-vae-repo-path','perform-vae-config-path']) {
        const field=document.getElementById(id);
        if (field) field.disabled=!ready || pipelinePhase !== 'idle';
    }
    const prepare=document.getElementById('perform-prepare-btn');
    if (prepare) {
        prepare.disabled=!ready || pipelinePhase !== 'idle' || !pipelineCorpusDir || !decoderVae || !decoderChoices?.length || !preparationSupported;
        prepare.textContent=preparationJob && !['running','prepared','already_valid'].includes(preparationJob.status) ? 'Retry ONNX preparation' : 'Prepare ONNX decoder';
        prepare.title=preparationSupported ? '' : 'Reconnect to an updated server to prepare decoders';
    }
    const weights = document.getElementById('perform-vae-weight-path');
    if (weights) weights.disabled = !ready || pipelinePhase !== 'idle';
    const refresh = document.getElementById('perform-decoder-refresh');
    if (refresh) refresh.disabled = !ready || pipelinePhase !== 'idle';
    if (corpusToggle) {
        const canUnload = ['perform', 'error'].includes(pipelinePhase);
        corpusToggle.textContent = canUnload ? 'Unload corpus' : 'Load corpus';
        corpusToggle.setAttribute('aria-pressed', String(canUnload));
        corpusToggle.disabled = !ready || (!canUnload && (pipelinePhase !== 'idle' || !pipelineCorpusDir
            || (decoderChoices !== null && !available?.selectable)));
    }
    if (typeof updateDecodeToggle === 'function') updateDecodeToggle();
}

function updateDecoderWindowState(decoder) {
    if (!decoder || typeof decoder !== 'object') return;
    const windows = normalizeDecoderWindows(decoder.supported_windows);
    const selected =
        decoder.requested_window ??
        decoder.pending_window ??
        decoder.transition?.target_window ??
        decoder.selected_window ??
        decoder.decoder_window ??
        decoder.window_size;
    const select = document.getElementById('perform-decoder-window');
    const existing = select
        ? normalizeDecoderWindows(Array.from(select.options, option => option.value))
        : [];
    if (windows.length > 0 && windows.join(',') !== existing.join(',')) {
        populateDecoderWindowOptions(windows, selected);
    } else if (select && Number.isInteger(Number(selected)) && existing.includes(Number(selected))) {
        syncDecoderControl(select, Number(selected));
        const display = document.getElementById('val-decoder-window');
        if (display) display.textContent = `T${Number(selected)}`;
    }
    if (pipelinePhase === 'perform' && !decoderCommandPending && pipelineConnectionReady && decoder.backend) {
        const offered = windows.length > 0
            ? ` · ${windows.map(value => `T${value}`).join(', ')}`
            : '';
        setDecoderStatus(
            `Active: ${decoder.backend} · ${decoder.device || decoder.provider || 'unknown device'}${offered}`,
            'ok'
        );
    }
    const controls = decoder.window_controls;
    decoderWindowControlsAvailable = Boolean(controls);
    if (controls) {
        decoderWindowMode = controls.mode;
        const mapping = {mode: 'window-mode', minimum: 'window-minimum', maximum: 'window-maximum',
                         content: 'manual-content', variation: 'manual-variation'};
        for (const key of ['minimum', 'maximum']) {
            const element = document.getElementById(mapping[key]);
            if (element && !editingDecoderControls.has(element.id) && Array.from(element.options, o => Number(o.value)).join(',') !== windows.join(',')) {
                element.innerHTML = '';
                windows.forEach(size => {
                    const option = document.createElement('option');
                    option.value = String(size); option.textContent = `T${size}`;
                    element.appendChild(option);
                });
            }
        }
        Object.entries(mapping).forEach(([key, id]) => {
            const element = document.getElementById(id);
            syncDecoderControl(element, controls[key]);
        });
        const range = document.getElementById('adaptive-window-range');
        if (range) range.hidden = document.getElementById('window-mode')?.value !== 'adaptive';
        const content = document.getElementById('manual-content-controls');
        if (content) content.hidden = false;
        const variation = document.getElementById('manual-variation-control');
        if (variation) variation.hidden = document.getElementById('manual-content')?.value !== 'variation';
    }
    if (decoder.error) setDecoderStatus(`Error: ${decoder.error}`, 'error');
    updateDecoderControls();

}

function handlePipelineMessage(data) {
    const type = data.type;
    pipelineServerSeen = true;

    if (type === 'pipeline_preparation') {
        onPreparationJob(data.job);
    } else if (type === 'pipeline_decoder_list') {
        onDecoderList(data);
    } else if (type === 'pipeline_state') {
        if ('preparation_supported' in data) preparationSupported=data.preparation_supported;
        if (data.server_namespace) decoderServerNamespace=data.server_namespace;
        if (data.preparation) onPreparationJob(data.preparation);
        if (data.error) preparationRequestPending=false;
        pipelinePhase = data.phase || 'idle';
        if (pipelinePhase !== 'perform') decoderWindowControlsAvailable = false;
        let restoredCorpus=data.corpus_dir;
        if (!restoredCorpus && !pipelineCorpusDir && pipelinePhase === 'idle') {
            try { restoredCorpus=localStorage.getItem(`waenderer.corpus.v1:${decoderServerNamespace}`); } catch (_) {}
        }
        setPipelineCorpusDir(restoredCorpus);
        applyDecoderPhase(data);
        if (data.error) {
            console.error('[pipeline] Error:', data.error);
            if (selectedTab === 'preprocess') showEncoderError(data.error);
            setDecoderStatus(`Error: ${data.error}`, 'error');
        }
        // If server is already in a phase, switch to that tab
        if (pipelinePhase !== 'idle') selectTab(['preprocess', 'train'].includes(pipelinePhase) ? pipelinePhase : 'perform');
        updatePipelinePhaseUI();
    } else if (type === 'pipeline_phase_change') {
        pipelinePhase = data.phase || 'idle';
        if (pipelinePhase !== 'perform') decoderWindowControlsAvailable = false;
        setPipelineCorpusDir(data.corpus_dir);
        applyDecoderPhase(data);

        if (data.completed === 'preprocess') {
            onPreprocessComplete(data);
            selectTab('train');
        } else if (data.completed === 'train') {
            onTrainComplete(data);
            selectTab('perform');
        }

        // Auto-switch to tab when a phase starts
        if (pipelinePhase !== 'idle') selectTab(['preprocess', 'train'].includes(pipelinePhase) ? pipelinePhase : 'perform');

        if (data.error) {
            console.error('[pipeline] Error:', data.error);
            if (selectedTab === 'preprocess') showEncoderError(data.error);
            setDecoderStatus(`Error: ${data.error}`, 'error');
        }
        updatePipelinePhaseUI();
    } else if (type === 'pipeline_stats') {
        if (data.phase === 'preprocess') {
            onPreprocessProgress(data);
        } else if (data.phase === 'train') {
            onTrainProgress(data);
        }
    } else if (type === 'pipeline_file_list') {
        if (data.audio_dir === selectedAudioDir) {
            document.getElementById('pp-source-status').textContent = data.error || `${data.files.length} WAV files found.`;
            onFileList(data);
        }
    } else if (type === 'pipeline_audio_directory') {
        document.getElementById('pp-browse-btn').disabled = false;
        document.getElementById('pp-source-status').textContent = data.error || '';
        if (data.audio_dir) {
            selectedAudioDir = data.audio_dir;
            const pathText = document.getElementById('pp-audio-dir');
            pathText.textContent = selectedAudioDir;
            pathText.classList.remove('panel-hidden');
            scanAudioSource();
        }
    } else if (type === 'pipeline_corpus_list') {
        onCorpusList(data);
    } else if (type === 'pipeline_vae_list') {
        onVAEList(data);
    }
}

function updatePipelinePhaseUI() {
    const phases = ['preprocess', 'train', 'perform'];

    // Update tab appearance: selected tab + running indicator
    phases.forEach(p => {
        const el = document.getElementById(`phase-${p}`);
        if (!el) return;
        el.classList.toggle('selected', p === selectedTab);
        el.classList.toggle('running', p === pipelinePhase && pipelinePhase !== 'idle');
    });

    // Show/hide panels based on selected tab
    const ppPanel = document.getElementById('preprocess-panel');
    const trainPanel = document.getElementById('train-panel');
    const performPanel = document.getElementById('perform-panel');

    if (ppPanel) ppPanel.classList.toggle('panel-hidden', selectedTab !== 'preprocess');
    if (trainPanel) trainPanel.classList.toggle('panel-hidden', selectedTab !== 'train');
    if (performPanel) performPanel.classList.toggle('panel-hidden', selectedTab !== 'perform');

    // Disable start buttons during active phase
    const ppStartBtn = document.getElementById('pp-start-btn');
    const trainStartBtn = document.getElementById('train-start-btn');
    if (ppStartBtn) ppStartBtn.disabled = pipelinePhase !== 'idle' || !encoderReady();
    if (trainStartBtn) trainStartBtn.disabled = pipelinePhase !== 'idle' || !pipelineCorpusDir;
    updateDecoderControls();
}

// ---- Preprocess ----

function onFileList(data) {
    const container = document.getElementById('pp-file-list-container');
    const tbody = document.querySelector('#pp-file-list tbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    if (data.error) {
        tbody.innerHTML = `<tr><td colspan="2" style="color:#ff6b6b">${data.error}</td></tr>`;
        container.classList.remove('panel-hidden');
        return;
    }

    const files = data.files || [];
    files.forEach(f => {
        const tr = document.createElement('tr');
        const nameCell = document.createElement('td');
        nameCell.textContent = f.name;
        const durationCell = document.createElement('td');
        durationCell.textContent = `${f.approx_duration.toFixed(1)}s`;
        tr.appendChild(nameCell);
        tr.appendChild(durationCell);
        tbody.appendChild(tr);
    });
    container.classList.toggle('panel-hidden', files.length === 0);
}

function onCorpusList(data) {
    const select = document.getElementById('perform-corpus-select');
    if (!select) return;

    // Keep the first option
    while (select.options.length > 1) select.remove(1);
    corpusMetadata.clear();

    (data.corpora || []).forEach(c => {
        corpusMetadata.set(c.path, c);
        const opt = document.createElement('option');
        opt.value = c.path;
        opt.textContent = c.name;
        select.appendChild(opt);
    });
    select.value = pipelineCorpusDir || '';
    if (pipelineCorpusDir) {
        const metadata = corpusMetadata.get(pipelineCorpusDir);
        const model = document.getElementById('perform-decoder-model');
        if (model && metadata?.vae_id) model.textContent = formatCorpusVae(metadata.vae_id);
        const summary = document.getElementById('perform-corpus-summary');
        if (summary && metadata?.corpus_audio) summary.textContent = formatCorpusAudioSummary(metadata.corpus_audio);
    }
}

function onVAEList(data) {
    const select = document.getElementById('pp-vae-select');
    if (!select) return;

    const previous = encoderListSeen ? select.value : 'same_s';
    encoderListSeen = true;
    encoderAvailability = {};
    select.innerHTML = '';
    (data.vaes || []).forEach(v => {
        encoderAvailability[v.vae_id] = v.availability;
        const opt = document.createElement('option');
        opt.value = v.vae_id;
        opt.textContent = v.display_name + (v.availability?.install_required ? ' — install required' : v.availability?.optional ? ' — optional' : ' — default');
        opt.dataset.requiresPath = v.requires_path ? '1' : '0';
        opt.dataset.pathLabel = v.path_label || 'Path to model weights';
        select.appendChild(opt);
    });
    if (Array.from(select.options).some(option => option.value === 'same_s')) {
        select.value = Object.prototype.hasOwnProperty.call(encoderAvailability, previous) ? previous : 'same_s';
    }
    // Trigger change to update path visibility
    updateVAEPathVisibility();
}

function updateVAEPathVisibility() {
    const select = document.getElementById('pp-vae-select');
    const pathGroup = document.getElementById('pp-vae-path-group');
    const pathLabel = document.getElementById('pp-vae-path-label');
    if (!select || !pathGroup) return;

    const opt = select.options[select.selectedIndex];
    const status = document.getElementById('pp-model-status');
    if (status) status.textContent = encoderAvailability[select.value]?.detail || 'Checking model availability…';
    const setup = document.getElementById('pp-model-setup');
    if (setup) setup.textContent = encoderAvailability[select.value]?.setup || '';
    updateDecoderControls();
    const needsPath = opt && opt.dataset.requiresPath === '1';
    pathGroup.classList.toggle('panel-hidden', !needsPath);
    if (pathLabel && opt) {
        pathLabel.textContent = opt.dataset.pathLabel || 'Path to model weights';
    }
}

function showEncoderError(error) {
    const text = document.getElementById('pp-progress-text');
    if (text) text.textContent = `Error: ${error}`;
}

function onPreprocessProgress(data) {
    const fill = document.getElementById('pp-progress-fill');
    const text = document.getElementById('pp-progress-text');
    const statsEl = document.getElementById('pp-stats');
    const event = data.event;

    if (event === 'model_download' || event === 'model_download_done') {
        if (text) text.textContent = data.detail;
    } else if (event === 'vae_load_start') {
        if (text) text.textContent = 'Loading and validating encoder…';
    } else if (event === 'vae_load_done') {
        if (text) text.textContent = 'Encoder ready';
    } else if (event === 'encode_file_start' || event === 'encode_file_done') {
        const pct = ((data.file_index + (event === 'encode_file_done' ? 1 : 0)) / data.total_files) * 100;
        if (fill) fill.style.width = pct + '%';
        if (text) text.textContent = `Encoding ${data.file_name} (${data.file_index + 1}/${data.total_files})`;
    } else if (event === 'scan_done') {
        if (text) text.textContent = `Found ${data.file_count} WAV files`;
    } else if (event === 'geometry_start') {
        if (text) text.textContent = 'Computing geometry...';
        if (fill) fill.style.width = '85%';
    } else if (event === 'embedding_start') {
        if (text) text.textContent = `Computing embedding (${data.reducer})...`;
        if (fill) fill.style.width = '75%';
    } else if (event === 'units_start') {
        if (text) text.textContent = 'Building reorganized units...';
        if (fill) fill.style.width = '90%';
    } else if (event === 'save_start') {
        if (text) text.textContent = 'Saving corpus...';
        if (fill) fill.style.width = '95%';
    } else if (event === 'complete') {
        if (fill) fill.style.width = '100%';
        if (text) text.textContent = 'Complete!';
    }
}

function onPreprocessComplete(data) {
    const statsEl = document.getElementById('pp-stats');
    if (statsEl) {
        statsEl.classList.remove('panel-hidden');
        statsEl.textContent =
            `Files: ${data.total_files || 0}\n` +
            `Frames: ${data.total_frames || 0}\n` +
            `Silence removed: ${(data.silence_removed_pct || 0).toFixed(1)}%\n` +
            `Corpus: ${data.corpus_dir || ''}`;
    }
    setPipelineCorpusDir(data.corpus_dir);

    // Update train panel corpus info
    const trainInfo = document.getElementById('train-corpus-info');
    if (trainInfo && pipelineCorpusDir) {
        trainInfo.textContent = `Corpus: ${pipelineCorpusDir}\nFrames: ${data.total_frames || '?'}`;
    }
}

// ---- Train ----

function onTrainProgress(data) {
    const fill = document.getElementById('train-progress-fill');
    const text = document.getElementById('train-progress-text');
    const statsEl = document.getElementById('train-stats');
    const mode = data.mode || 'random';

    if (data.epoch != null && data.total_epochs != null) {
        const pct = (data.epoch / data.total_epochs) * 100;
        if (fill) fill.style.width = pct + '%';

        const trainLoss = data.train ? (data.train.loss ?? '?').toFixed ? data.train.loss.toFixed(4) : data.train.loss : '?';
        const valLoss = data.val ? (data.val.loss ?? '?').toFixed ? data.val.loss.toFixed(4) : data.val.loss : '?';
        if (text) text.textContent = `[${mode}] Epoch ${data.epoch}/${data.total_epochs} | train=${trainLoss} val=${valLoss}`;

        // Accumulate history for chart
        if (data.train && data.val) {
            if (mode === 'random' || mode === 'all') {
                trainingHistory.random.train_loss.push(data.train.loss || 0);
                trainingHistory.random.val_loss.push(data.val.loss || 0);
                if (data.val.window_mae_frames != null) {
                    trainingHistory.random.window_mae.push(data.val.window_mae_frames);
                }
            }
            if (mode === 'reorganized') {
                trainingHistory.reorganized.train_loss.push(data.train.loss || 0);
                trainingHistory.reorganized.val_loss.push(data.val.loss || 0);
                if (data.val.acc != null) {
                    trainingHistory.reorganized.accuracy.push(data.val.acc);
                }
            }
        }

        // Notify sketch.js to render training curves
        if (typeof setDrawMode === 'function') {
            setDrawMode('training');
        }
    }

    if (data.event === 'complete') {
        if (statsEl) {
            statsEl.classList.remove('panel-hidden');
            statsEl.textContent = `[${mode}] Training complete: ${data.path || ''}`;
        }
    }
}

function onTrainComplete(data) {
    const statsEl = document.getElementById('train-stats');
    if (statsEl) {
        statsEl.classList.remove('panel-hidden');
        let info = 'Training complete.';
        if (data.best_val_loss != null) info += ` Best val loss: ${data.best_val_loss.toFixed(4)}`;
        statsEl.textContent = info;
    }
    const fill = document.getElementById('train-progress-fill');
    if (fill) fill.style.width = '100%';

    // Switch canvas back to 3D navigation view
    if (typeof setDrawMode === 'function') {
        setDrawMode('perform');
    }
}

// ---- Setup Controls ----

function scanAudioSource() {
    const dir = selectedAudioDir;
    document.getElementById('pp-file-list-container').classList.add('panel-hidden');
    const sent = dir && sendPipelineMessage({ type: 'pipeline_list_files', audio_dir: dir });
    document.getElementById('pp-source-status').textContent = !dir ? '' :
        sent ? 'Scanning…' : 'Connect to the server to scan this directory.';
}

function setupPipelineControls() {
    populateDecoderWindowOptions(APP_DECODER_WINDOWS, 2);
    setDecoderStatus('No active decoder. Select a corpus and choose a decoder.');
    // Tab click handlers
    ['preprocess', 'train', 'perform'].forEach(tab => {
        const el = document.getElementById(`phase-${tab}`);
        if (el) {
            el.addEventListener('click', () => selectTab(tab));
        }
    });

    // Set initial tab state
    updatePipelinePhaseUI();

    // VAE dropdown change handler
    const vaeSelect = document.getElementById('pp-vae-select');
    if (vaeSelect) {
        vaeSelect.addEventListener('change', updateVAEPathVisibility);
    }

    // Native picker opens on the server host and selection triggers a scan.
    const browseBtn = document.getElementById('pp-browse-btn');
    browseBtn?.addEventListener('click', () => {
        if (sendPipelineMessage({type: 'pipeline_choose_audio_dir',
            audio_dir: selectedAudioDir})) {
            browseBtn.disabled = true;
            document.getElementById('pp-source-status').textContent = 'Choose a folder in the system window…';
        } else {
            document.getElementById('pp-source-status').textContent = 'Connect to the server to choose a directory.';
        }
    });

    // Apply corpus selection immediately; loading remains an explicit action.
    const corpusSelect = document.getElementById('perform-corpus-select');
    if (corpusSelect) {
        corpusSelect.addEventListener('change', () => {
            if (corpusSelect.value) {
                setPipelineCorpusDir(corpusSelect.value);
                const trainInfo = document.getElementById('train-corpus-info');
                if (trainInfo) trainInfo.textContent = `Corpus: ${pipelineCorpusDir}`;
                updatePipelinePhaseUI();
            }
        });
    }

    document.getElementById('pp-check-models')?.addEventListener('click', () => {
        encoderAvailability = {};
        updateVAEPathVisibility();
        sendPipelineMessage({type: 'pipeline_list_vaes'});
    });

    // Preprocess start
    const ppStartBtn = document.getElementById('pp-start-btn');
    if (ppStartBtn) {
        ppStartBtn.addEventListener('click', () => {
            const audioDir = selectedAudioDir;
            if (!audioDir || !encoderReady()) return;

            const config = {
                audio_dir: audioDir,
                out_prefix: audioDir.split('/').pop() || 'corpus',
                vae_id: document.getElementById('pp-vae-select')?.value || 'same_s',
                vae_weight_path: document.getElementById('pp-vae-path')?.value || '',
                trim_silence: document.getElementById('pp-trim-silence')?.checked ?? true,
                silence_threshold_db: parseFloat(document.getElementById('pp-threshold')?.value || '-45'),
                manual_reducer: document.getElementById('pp-reducer')?.value || 'umap',
                manual_embed_dim: parseInt(document.getElementById('pp-embed-dim')?.value || '4'),
                latent_nav_k: parseInt(document.getElementById('pp-latent-nav-k')?.value || '32'),
                reorg_target_sec: parseFloat(document.getElementById('pp-reorg-target-sec')?.value || '5.0'),
                reorg_min_sec: parseFloat(document.getElementById('pp-reorg-min-sec')?.value || '2.0'),
                reorg_max_sec: parseFloat(document.getElementById('pp-reorg-max-sec')?.value || '10.0'),
            };

            // Reset progress
            const fill = document.getElementById('pp-progress-fill');
            if (fill) fill.style.width = '0%';
            const text = document.getElementById('pp-progress-text');
            if (text) text.textContent = 'Starting...';
            const stats = document.getElementById('pp-stats');
            if (stats) stats.classList.add('panel-hidden');

            sendPipelineMessage({ type: 'pipeline_start_preprocess', config });
        });
    }

    // Preprocess cancel
    const ppCancelBtn = document.getElementById('pp-cancel-btn');
    if (ppCancelBtn) {
        ppCancelBtn.addEventListener('click', () => {
            sendPipelineMessage({ type: 'pipeline_cancel' });
        });
    }

    // Train start
    const trainStartBtn = document.getElementById('train-start-btn');
    if (trainStartBtn) {
        trainStartBtn.addEventListener('click', () => {
            if (!pipelineCorpusDir) return;

            // Reset training history
            trainingHistory = {
                random: { train_loss: [], val_loss: [], window_mae: [] },
                reorganized: { train_loss: [], val_loss: [], accuracy: [] },
            };

            const config = {
                corpus_dir: pipelineCorpusDir,
                navigation_mode: document.getElementById('train-mode')?.value || 'all',
                epochs: parseInt(document.getElementById('train-epochs')?.value || '2000'),
                batch_size: parseInt(document.getElementById('train-batch-size')?.value || '64'),
                lr: parseFloat(document.getElementById('train-lr')?.value || '0.001'),
                hidden: parseInt(document.getElementById('train-hidden')?.value || '256'),
                lambda_recon: parseFloat(document.getElementById('train-recon')?.value || '1.0'),
                lambda_smooth: parseFloat(document.getElementById('train-smooth')?.value || '0.1'),
                lambda_manifold: parseFloat(document.getElementById('train-manifold')?.value || '0.1'),
                lambda_window: parseFloat(document.getElementById('train-window')?.value || '0.5'),
                reorganized_epochs: parseInt(document.getElementById('train-reorg-epochs')?.value || '120'),
                reorganized_hidden_dim: parseInt(document.getElementById('train-reorg-hidden')?.value || '192'),
            };

            const fill = document.getElementById('train-progress-fill');
            if (fill) fill.style.width = '0%';
            const text = document.getElementById('train-progress-text');
            if (text) text.textContent = 'Starting training...';
            const stats = document.getElementById('train-stats');
            if (stats) stats.classList.add('panel-hidden');

            sendPipelineMessage({ type: 'pipeline_start_train', config });
        });
    }

    // Train cancel
    const trainCancelBtn = document.getElementById('train-cancel-btn');
    if (trainCancelBtn) {
        trainCancelBtn.addEventListener('click', () => {
            sendPipelineMessage({ type: 'pipeline_cancel' });
        });
    }

    document.getElementById('perform-decoder-engine')?.addEventListener('change', () => {
        saveDecoderPreference();
        updateDecoderAvailability();
        updateDecoderControls();
    });
    for (const id of ['perform-vae-weight-path','perform-vae-repo-path','perform-vae-config-path']) {
        document.getElementById(id)?.addEventListener('input', () => {
            decoderChoices=[];
            decoderModelIdentity=null;
            updateDecoderControls();
            if (decoderSourceTimer) clearTimeout(decoderSourceTimer);
            decoderSourceTimer=setTimeout(requestDecoderAvailability,200);
        });
    }
    document.getElementById('perform-prepare-btn')?.addEventListener('click', () => {
        if (!preparationSupported || preparationBusy() || pipelinePhase !== 'idle') return;
        const source=decoderSources();
        const request={vae_id:decoderVae};
        if (decoderVae?.startsWith('ear_')) Object.assign(request, {weights:source.vae_weight_path,repo:source.vae_repo_path,config:source.vae_config_path});
        preparationRequestPending=sendPipelineMessage({type:'pipeline_prepare_decoder',corpus_dir:pipelineCorpusDir,
            fingerprint:currentDecoderFingerprint(),request});
        updateDecoderControls();
    });
    document.getElementById('perform-decoder-refresh')?.addEventListener('click', () => {
        requestDecoderAvailability(true);
    });
    // Load/unload corpus using the server's confirmed pipeline state.
    const corpusToggle = document.getElementById('perform-corpus-toggle');
    if (corpusToggle) {
        corpusToggle.addEventListener('click', () => {
            if (corpusToggle.disabled || decoderCommandPending) return;
            if (['perform', 'error'].includes(pipelinePhase)) {
                if (sendPipelineMessage({type: 'pipeline_stop_perform'})) {
                    decoderCommandPending = true;
                    setDecoderStatus('Unloading corpus…');
                    updateDecoderControls();
                } else setDecoderStatus('Not connected. Reconnect and try again.', 'error');
                return;
            }
            const choice = selectedDecoderChoice();
            const decoderWindow = Number(document.getElementById('perform-decoder-window')?.value);
            if (!Number.isInteger(decoderWindow)) return;
            const sent = sendPipelineMessage({
                type: 'pipeline_start_perform',
                config: {
                    corpus_dir: pipelineCorpusDir,
                    decoder_window: decoderWindow,
                    decoder_backend: choice.backend,
                    decoder_device: choice.device,
                    ...decoderSources(),
                    decoder_model_identity: decoderModelIdentity,
                },
            });
            if (sent) {
                decoderCommandPending = true;
                setDecoderStatus('Loading and warming decoder…');
                updateDecoderControls();
            } else setDecoderStatus('Not connected. Reconnect and try again.', 'error');
        });
    }

    const decoderWindowSelect = document.getElementById('perform-decoder-window');
    if (decoderWindowSelect) {
        decoderWindowSelect.addEventListener('change', () => {
            noteDecoderControlChange(decoderWindowSelect);
            const size = Number(decoderWindowSelect.value);
            if (!Number.isInteger(size)) return;
            const display = document.getElementById('val-decoder-window');
            if (display) display.textContent = `T${size}`;
            if (pipelinePhase === 'perform' || decoderWindowControlsAvailable) {
                sendPipelineMessage({ type: 'decoder_window', size });
            }
        });
    }

    const windowControlFields = { 'window-mode': 'mode', 'window-minimum': 'minimum',
        'window-maximum': 'maximum', 'manual-content': 'content', 'manual-variation': 'variation' };
    ['perform-decoder-window', ...Object.keys(windowControlFields)].forEach(id => {
        const element = document.getElementById(id);
        if (!element) return;
        element.addEventListener('focus', () => editingDecoderControls.add(id));
        element.addEventListener('blur', () => editingDecoderControls.delete(id));
    });
    Object.entries(windowControlFields).forEach(([id, key]) => {
        const element = document.getElementById(id);
        if (!element) return;
        element.addEventListener('change', () => {
            noteDecoderControlChange(element);
            const value = ['minimum', 'maximum', 'variation'].includes(key) ? Number(element.value) : element.value;
            sendPipelineMessage({type: 'window_controls', controls: {[key]: value}});
        });
    });

    // Threshold slider display
    const thresholdSlider = document.getElementById('pp-threshold');
    const thresholdDisplay = document.getElementById('val-pp-threshold');
    if (thresholdSlider && thresholdDisplay) {
        thresholdSlider.addEventListener('input', (e) => {
            thresholdDisplay.textContent = e.target.value;
        });
    }

    // Loss weight slider displays
    ['recon', 'smooth', 'manifold', 'window'].forEach(name => {
        const slider = document.getElementById(`train-${name}`);
        const display = document.getElementById(`val-train-${name}`);
        if (slider && display) {
            slider.addEventListener('input', (e) => {
                display.textContent = parseFloat(e.target.value).toFixed(2);
            });
        }
    });

    // Collapsible sections
    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.addEventListener('click', () => {
            header.classList.toggle('open');
            const bodyId = header.id.replace('-header', '-body');
            const body = document.getElementById(bodyId);
            if (body) body.classList.toggle('open');
        });
    });
}

function sendPipelineMessage(data) {
    if (typeof ws !== 'undefined' && ws && wsConnected) {
        try {
            ws.send(JSON.stringify(data));
            return true;
        } catch (error) {
            console.error('[pipeline] Failed to send WebSocket message:', error);
        }
    }
    return false;
}

function requestCorpusList() {
    sendPipelineMessage({ type: 'pipeline_list_corpora' });
    sendPipelineMessage({ type: 'pipeline_list_vaes' });
    sendPipelineMessage({ type: 'pipeline_get_state' });
    requestDecoderAvailability();
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupPipelineControls);
} else {
    setupPipelineControls();
}


function requestDecoderAvailability(retryValidation=false) {
    decoderLatestRequest=`${decoderClientId}:${++decoderRequestNumber}`;
    decoderFingerprint=currentDecoderFingerprint();
    decoderChoices=[];
    sendPipelineMessage({type:'pipeline_list_decoders',corpus_dir:pipelineCorpusDir,
        request_id:decoderLatestRequest,fingerprint:decoderFingerprint,retry_validation:retryValidation,...decoderSources()});
    updateDecoderControls();
}
