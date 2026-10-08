/* Experimental backend microphone navigation; text-only diagnostic rendering. */
function sendAudioInput(action, values = {}) {
    if (ws && wsConnected) ws.send(JSON.stringify({type: 'audio_input', action, ...values}));
}

function setupAudioInputControls() {
    document.getElementById('mode-audio-input').addEventListener('click', () => {
        if (transportRunning) return;
        selectedNavigationMode = 'audio_input';
        applyNavigationModeUI();
        sendTransportSetMode('audio_input');
        sendAudioInput('devices');
    });
    document.getElementById('audio-input-devices').addEventListener('click', () => sendAudioInput('devices'));
    document.getElementById('audio-input-start').addEventListener('click', () => {
        const value = document.getElementById('audio-input-device').value;
        sendAudioInput('start', {device: value === '' ? null : Number(value)});
    });
    document.getElementById('audio-input-stop').addEventListener('click', () => sendAudioInput('stop'));
    document.getElementById('audio-input-path').addEventListener('change', event =>
        sendAudioInput('path', {path: event.target.value}));
}

function updateAudioInputState(state) {
    const button = document.getElementById('mode-audio-input');
    if (state.available) button.dataset.available = 'true';
    else delete button.dataset.available;
    const device = document.getElementById('audio-input-device');
    const entries = state.devices || [];
    const signature = JSON.stringify(entries);
    if (device.dataset.signature !== signature) {
        const previous = device.value;
        device.replaceChildren(new Option('System default', ''));
        entries.forEach(entry => device.add(new Option(entry.name, String(entry.id))));
        if ([...device.options].some(option => option.value === previous)) device.value = previous;
        device.dataset.signature = signature;
    }
    device.disabled = Boolean(state.running);
    document.getElementById('audio-input-start').disabled = !state.available || Boolean(state.running);
    document.getElementById('audio-input-stop').disabled = !state.running;
    document.getElementById('audio-input-path').disabled = !state.available;
    if (state.path) document.getElementById('audio-input-path').value = state.path;
    document.getElementById('audio-input-level').value = Math.max(-90, state.level_db ?? -90);
    const settings = state.settings || {};
    const active = (state.paths || {})[state.path] || {};
    const silent = state.running && state.level_db < (settings.silence_db ?? -45);
    const stale = active.age_ms != null && active.age_ms > (settings.freshness_seconds || 2) * 1000;
    document.getElementById('audio-input-status').textContent = state.message || state.capture_error ||
        (!state.available ? 'Audio Input unavailable for this corpus.' :
         `${state.status || 'stopped'} · ${state.device || 'default input'} · ${state.path || 'descriptors'}` +
         (silent ? ' · silence: holding' : stale ? ' · stale result: holding' : ''));
    const number = value => value == null ? '—' : Number(value).toFixed(1);
    const lines = ['descriptors', 'latents'].map(path => {
        const result = (state.paths || {})[path] || {};
        return `${path}: ${result.status || 'waiting'} · index ${result.index ?? '—'} · ` +
            `distance ${number(result.distance)} · analysis ${number(result.analysis_ms)} ms · ` +
            `input age ${number(result.age_ms)} ms` +
            (result.source ? ` · ${result.source} @ ${number(result.source_seconds)} s` : '') +
            (result.error ? `\n${result.error}` : '');
    });
    lines.push(`Render anchor: ${state.render_anchor ?? '—'}. Distances use different spaces.`);
    document.getElementById('audio-input-diagnostics').textContent = lines.join('\n');
}
