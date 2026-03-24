/**
 * Pipeline frontend logic for Stable Audio Wanderer.
 * Manages preprocess/train/perform tab UI and WebSocket pipeline messages.
 */

// Pipeline state
let pipelinePhase = 'idle';
let pipelineCorpusDir = null;
let selectedTab = 'preprocess';

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

function handlePipelineMessage(data) {
    const type = data.type;

    if (type === 'pipeline_state') {
        pipelinePhase = data.phase || 'idle';
        pipelineCorpusDir = data.corpus_dir || pipelineCorpusDir;
        // If server is already in a phase, switch to that tab
        if (pipelinePhase !== 'idle') selectTab(pipelinePhase);
        updatePipelinePhaseUI();
    } else if (type === 'pipeline_phase_change') {
        pipelinePhase = data.phase || 'idle';
        pipelineCorpusDir = data.corpus_dir || pipelineCorpusDir;

        if (data.completed === 'preprocess') {
            onPreprocessComplete(data);
            selectTab('train');
        } else if (data.completed === 'train') {
            onTrainComplete(data);
            selectTab('perform');
        }

        // Auto-switch to tab when a phase starts
        if (pipelinePhase !== 'idle') selectTab(pipelinePhase);

        if (data.error) {
            console.error('[pipeline] Error:', data.error);
        }
        updatePipelinePhaseUI();
    } else if (type === 'pipeline_stats') {
        if (data.phase === 'preprocess') {
            onPreprocessProgress(data);
        } else if (data.phase === 'train') {
            onTrainProgress(data);
        }
    } else if (type === 'pipeline_file_list') {
        onFileList(data);
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
    const performStartBtn = document.getElementById('perform-start-btn');

    if (ppStartBtn) ppStartBtn.disabled = pipelinePhase !== 'idle';
    if (trainStartBtn) trainStartBtn.disabled = pipelinePhase !== 'idle' || !pipelineCorpusDir;
    if (performStartBtn) performStartBtn.disabled = pipelinePhase !== 'idle' || !pipelineCorpusDir;
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
        tr.innerHTML = `<td>${f.name}</td><td>${f.approx_duration.toFixed(1)}s</td>`;
        tbody.appendChild(tr);
    });
    container.classList.toggle('panel-hidden', files.length === 0);
}

function onCorpusList(data) {
    const select = document.getElementById('pp-corpus-select');
    if (!select) return;

    // Keep the first option
    while (select.options.length > 1) select.remove(1);

    (data.corpora || []).forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.path;
        opt.textContent = c.name;
        select.appendChild(opt);
    });
}

function onVAEList(data) {
    const select = document.getElementById('pp-vae-select');
    if (!select) return;

    select.innerHTML = '';
    (data.vaes || []).forEach(v => {
        const opt = document.createElement('option');
        opt.value = v.vae_id;
        opt.textContent = v.display_name;
        opt.dataset.requiresPath = v.requires_path ? '1' : '0';
        opt.dataset.pathLabel = v.path_label || 'Path to model weights';
        select.appendChild(opt);
    });
    // Trigger change to update path visibility
    updateVAEPathVisibility();
}

function updateVAEPathVisibility() {
    const select = document.getElementById('pp-vae-select');
    const pathGroup = document.getElementById('pp-vae-path-group');
    const pathLabel = document.getElementById('pp-vae-path-label');
    if (!select || !pathGroup) return;

    const opt = select.options[select.selectedIndex];
    const needsPath = opt && opt.dataset.requiresPath === '1';
    pathGroup.classList.toggle('panel-hidden', !needsPath);
    if (pathLabel && opt) {
        pathLabel.textContent = opt.dataset.pathLabel || 'Path to model weights';
    }
}

function onPreprocessProgress(data) {
    const fill = document.getElementById('pp-progress-fill');
    const text = document.getElementById('pp-progress-text');
    const statsEl = document.getElementById('pp-stats');
    const event = data.event;

    if (event === 'encode_file_start' || event === 'encode_file_done') {
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
    pipelineCorpusDir = data.corpus_dir || pipelineCorpusDir;

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

function setupPipelineControls() {
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

    // Scan button
    const scanBtn = document.getElementById('pp-scan-btn');
    if (scanBtn) {
        scanBtn.addEventListener('click', () => {
            const dir = document.getElementById('pp-audio-dir').value.trim();
            if (!dir) return;
            sendPipelineMessage({ type: 'pipeline_list_files', audio_dir: dir });
        });
    }

    // Use existing corpus
    const useCorpusBtn = document.getElementById('pp-use-corpus-btn');
    if (useCorpusBtn) {
        useCorpusBtn.addEventListener('click', () => {
            const select = document.getElementById('pp-corpus-select');
            if (select && select.value) {
                pipelineCorpusDir = select.value;
                const trainInfo = document.getElementById('train-corpus-info');
                if (trainInfo) trainInfo.textContent = `Corpus: ${pipelineCorpusDir}`;
                updatePipelinePhaseUI();
            }
        });
    }

    // Preprocess start
    const ppStartBtn = document.getElementById('pp-start-btn');
    if (ppStartBtn) {
        ppStartBtn.addEventListener('click', () => {
            const audioDir = document.getElementById('pp-audio-dir').value.trim();
            if (!audioDir) return;

            const config = {
                audio_dir: audioDir,
                out_prefix: audioDir.split('/').pop() || 'corpus',
                vae_id: document.getElementById('pp-vae-select')?.value || 'stable_audio_open',
                vae_weight_path: document.getElementById('pp-vae-path')?.value || '',
                trim_silence: document.getElementById('pp-trim-silence')?.checked ?? true,
                silence_threshold_db: parseFloat(document.getElementById('pp-threshold')?.value || '-45'),
                manual_reducer: document.getElementById('pp-reducer')?.value || 'pca',
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

    // Perform start
    const performStartBtn = document.getElementById('perform-start-btn');
    if (performStartBtn) {
        performStartBtn.addEventListener('click', () => {
            if (!pipelineCorpusDir) return;
            sendPipelineMessage({
                type: 'pipeline_start_perform',
                config: { corpus_dir: pipelineCorpusDir },
            });
        });
    }

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
        ws.send(JSON.stringify(data));
    }
}

function requestCorpusList() {
    sendPipelineMessage({ type: 'pipeline_list_corpora' });
    sendPipelineMessage({ type: 'pipeline_list_vaes' });
    sendPipelineMessage({ type: 'pipeline_get_state' });
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupPipelineControls);
} else {
    setupPipelineControls();
}
