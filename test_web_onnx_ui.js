#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class FakeElement {
    constructor(id = "") {
        this.id = id;
        this.value = "";
        this.textContent = "";
        this.disabled = false;
        this.options = [];
        this.style = {};
        this.dataset = {};
        this.listeners = {};
        this.classList = { add() {}, remove() {}, toggle() {} };
        this._innerHTML = "";
    }
    addEventListener(type, callback) { this.listeners[type] = callback; }
    appendChild(child) { this.options.push(child); }
    set innerHTML(value) {
        this._innerHTML = String(value);
        if (value === "") this.options = [];
    }
    get innerHTML() { return this._innerHTML; }
}

const ids = [
    "perform-vae-weight-group", "perform-vae-weight-path", "perform-decoder-status", "perform-decoder-window", "val-decoder-window",
    "perform-start-btn", "perform-stop-btn", "perform-decoder-engine", "perform-decoder-refresh", "perform-decoder-availability", "btn-start", "pp-vae-select", "pp-vae-path-group",
    "window-mode", "window-minimum", "window-maximum", "manual-content", "manual-variation",
    "adaptive-window-range", "manual-content-controls", "manual-variation-control",
];
const elements = new Map(ids.map(id => [id, new FakeElement(id)]));
const sent = [];
const document = {
    readyState: "loading",
    getElementById(id) { return elements.get(id) || null; },
    createElement() { return new FakeElement(); },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    addEventListener() {},
};
const context = vm.createContext({
    console,
    document,
    wsConnected: true,
    ws: { send(payload) { sent.push(JSON.parse(payload)); } },
});
vm.runInContext(fs.readFileSync("web/pipeline.js", "utf8"), context, {
    filename: "web/pipeline.js",
});
vm.runInContext("setupPipelineControls();", context);

const windows = elements.get("perform-decoder-window");
const start = elements.get("perform-start-btn");
const transportStart = elements.get("btn-start");
assert.deepEqual(windows.options.map(option => option.value), ["2"]);
assert.equal(windows.value, "2");

vm.runInContext(
    "handlePipelineMessage({type:'pipeline_state',phase:'idle',corpus_dir:'/corpus/demo'});",
    context,
);
assert.equal(start.disabled, true, "corpus-specific availability is required before Start");
assert.equal(sent.at(-1).corpus_dir, '/corpus/demo');
vm.runInContext("onDecoderList({corpus_dir:'/corpus/demo',vae_id:'same_s',decoders:[{backend:'onnxruntime',device:'cpu',selectable:true,label:'ONNX'}]});", context);
assert.equal(start.disabled, false, "compatible decoder enables Start");
assert.equal(transportStart.disabled, true, "transport waits for perform handoff");

start.listeners.click();
assert.deepEqual(sent.pop(), {
    type: "pipeline_start_perform",
    config: { corpus_dir: "/corpus/demo", decoder_window: 2, decoder_backend: "onnxruntime", decoder_device: "cpu", vae_weight_path: "" },
});

vm.runInContext(
    "handlePipelineMessage({type:'pipeline_phase_change',phase:'perform',corpus_dir:'/corpus/demo'});",
    context,
);
assert.equal(transportStart.disabled, false);
assert.equal(windows.disabled, false);
windows.value = "4";
windows.listeners.change();
assert.deepEqual(sent.pop(), { type: "decoder_window", size: 4 });

vm.runInContext(
    "handlePipelineMessage({type:'pipeline_state',phase:'idle',error:'not a SAME-S corpus'});",
    context,
);
assert.match(elements.get("perform-decoder-status").textContent, /not a SAME-S corpus/);

vm.runInContext(
    "onVAEList({vaes:[{vae_id:'stable_audio_open',display_name:'SAO'}," +
    "{vae_id:'same_s',display_name:'SAME-S'}]});",
    context,
);
assert.equal(elements.get("pp-vae-select").value, "same_s");
assert.equal(elements.get("pp-vae-select").options.length, 2, "other VAEs remain available");

const html = fs.readFileSync("web/index.html", "utf8");
assert.doesNotMatch(html, /perform-bundle-path|Validate Bundle/);
const manualWindowTag = html.match(/<input(?=[^>]*id="manual-window")[^>]*>/);
assert.ok(manualWindowTag);
assert.doesNotMatch(manualWindowTag[0], /\bdisabled\b/);

const sketch = fs.readFileSync("web/sketch.js", "utf8");
assert.match(sketch, /decoder\.backend === 'onnxruntime'/);
assert.match(sketch, /legacyManualWindow\.disabled = webOnnxMode/);
console.log("Web app-owned ONNX workflow contract passed");

vm.runInContext("pipelinePhase = 'perform'; updateDecoderWindowState({supported_windows:[2,4,6,8], requested_window:6, window_controls:{mode:'adaptive',minimum:2,maximum:8,content:'variation',variation:0.4}});", context);
assert.equal(windows.disabled, true);
assert.deepEqual(elements.get('window-minimum').options.map(o => o.value), ['2','4','6','8']);
assert.equal(elements.get('manual-content').value, 'variation');
assert.equal(elements.get('manual-variation-control').hidden, false);
elements.get('manual-content').value = 'held';
elements.get('manual-content').listeners.change();
assert.deepEqual(sent.pop(), {type:'window_controls', controls:{content:'held'}});
vm.runInContext("updateDecoderWindowState({supported_windows:[2,4], requested_window:2, window_controls:{mode:'fixed',minimum:2,maximum:4,content:'held',variation:0.4}});", context);
assert.equal(windows.disabled, false);
assert.equal(elements.get('manual-variation-control').hidden, true);

// Frequent decoder broadcasts must not overwrite an open native dropdown.
const contentSelect = elements.get('manual-content');
contentSelect.listeners.focus();
contentSelect.value = 'variation';
for (let i = 0; i < 30; i++) {
    vm.runInContext("updateDecoderWindowState({supported_windows:[2,4], requested_window:2, window_controls:{mode:'fixed',minimum:2,maximum:4,content:'held',variation:0.4}});", context);
}
assert.equal(contentSelect.value, 'variation', 'streaming state preserves keyboard/menu selection');
contentSelect.listeners.change();
contentSelect.listeners.blur();
vm.runInContext("updateDecoderWindowState({supported_windows:[2,4], requested_window:2, window_controls:{mode:'fixed',minimum:2,maximum:4,content:'held',variation:0.4}});", context);
assert.equal(contentSelect.value, 'variation', 'stale state cannot undo an unacknowledged change');
vm.runInContext("updateDecoderWindowState({supported_windows:[2,4], requested_window:2, window_controls:{mode:'fixed',minimum:2,maximum:4,content:'variation',variation:0.4}});", context);
assert.equal(contentSelect.value, 'variation');
vm.runInContext("updateDecoderWindowState({supported_windows:[2,4], requested_window:2, window_controls:{mode:'fixed',minimum:2,maximum:4,content:'source',variation:0.4}});", context);
assert.equal(contentSelect.value, 'source', 'server state resumes after acknowledgement');

const lengthSelect = elements.get('window-mode');
lengthSelect.listeners.focus();
lengthSelect.value = 'adaptive';
vm.runInContext("updateDecoderWindowState({supported_windows:[2,4], requested_window:2, window_controls:{mode:'fixed',minimum:2,maximum:4,content:'source',variation:0.4}});", context);
assert.equal(lengthSelect.value, 'adaptive');
lengthSelect.listeners.change();
assert.deepEqual(sent.pop(), {type:'window_controls', controls:{mode:'adaptive'}});
lengthSelect.listeners.blur();

// A reconnect can receive live decoder state before the pipeline phase message.
vm.runInContext("pipelinePhase = 'idle'; updateDecoderWindowState({supported_windows:[2,4], requested_window:2, window_controls:{mode:'adaptive',minimum:2,maximum:4,content:'source',variation:0.4}});", context);
assert.equal(lengthSelect.disabled, false);
assert.equal(contentSelect.disabled, false);
assert.equal(windows.disabled, true, 'fixed T remains disabled in adaptive mode');
vm.runInContext("handlePipelineMessage({type:'pipeline_state',phase:'idle'});", context);
assert.equal(lengthSelect.disabled, true, 'unloaded transport disables runtime controls');
console.log('Streaming dropdown interaction and reconnect checks passed');

// Phase 4: explicit selection, lifecycle lock, errors and reconnect.
const engine = elements.get('perform-decoder-engine');
const stopPerform = elements.get('perform-stop-btn');
vm.runInContext(`handlePipelineMessage({type:'pipeline_decoder_list',corpus_dir:'/corpus/demo',vae_id:'same_s',decoders:[
    {backend:'onnxruntime',device:'cpu',label:'ONNX · CPU',hardware:true,dependencies:true,weights:true,selectable:true,detail:'Not yet validated'},
    {backend:'pytorch',device:'mps',label:'PyTorch · GPU · MPS',hardware:true,dependencies:true,weights:true,selectable:true,detail:'Not yet validated'},
    {backend:'pytorch',device:'cuda:0',label:'PyTorch · GPU · CUDA:0',hardware:false,dependencies:true,weights:true,selectable:false,detail:'GPU unavailable'}
]});`, context);
assert.equal(engine.value, 'onnxruntime|cpu');
assert.equal(engine.options[2].disabled, true);
engine.value = 'pytorch|mps';
engine.listeners.change();
start.listeners.click();
assert.equal(sent.at(-1).config.decoder_backend, 'pytorch');
assert.equal(sent.at(-1).config.decoder_device, 'mps');
assert.equal(start.disabled, true);
assert.equal(engine.disabled, true);
assert.equal(transportStart.disabled, true);
const sentCount = sent.length;
start.listeners.click();
assert.equal(sent.length, sentCount, 'duplicate start is blocked');
vm.runInContext("handlePipelineMessage({type:'pipeline_phase_change',phase:'preparing'});", context);
assert.equal(vm.runInContext('selectedTab', context), 'perform');
assert.match(elements.get('perform-decoder-status').textContent, /warming/);
vm.runInContext("handlePipelineMessage({type:'pipeline_phase_change',phase:'perform',decoder:{backend:'pytorch',device:'mps:0'}});", context);
assert.match(elements.get('perform-decoder-status').textContent, /Active: pytorch · mps:0/);
assert.equal(engine.disabled, true, 'even paused audio requires leaving Perform');
assert.equal(stopPerform.disabled, false);
stopPerform.listeners.click();
assert.equal(sent.at(-1).type, 'pipeline_stop_perform');
assert.equal(transportStart.disabled, true);
vm.runInContext("handlePipelineMessage({type:'pipeline_phase_change',phase:'stopping',decoder:null});", context);
assert.equal(vm.runInContext('selectedTab', context), 'perform');
vm.runInContext("handlePipelineMessage({type:'pipeline_phase_change',phase:'idle',decoder:null});", context);
assert.equal(engine.disabled, false);
assert.doesNotMatch(elements.get('perform-decoder-status').textContent, /Active:/);
engine.value = 'onnxruntime|cpu'; engine.listeners.change(); start.listeners.click();
assert.equal(sent.at(-1).config.decoder_backend, 'onnxruntime');
vm.runInContext("handlePipelineMessage({type:'pipeline_phase_change',phase:'idle',decoder:null,error:'warm-up failed; retry'});", context);
assert.equal(start.disabled, false);
assert.match(elements.get('perform-decoder-status').textContent, /warm-up failed/);
vm.runInContext('pipelineDisconnected();', context);
assert.equal(start.disabled, true);
assert.equal(engine.disabled, true);
assert.match(elements.get('perform-decoder-status').textContent, /unknown/);
vm.runInContext("handlePipelineMessage({type:'pipeline_state',phase:'perform',decoder:{backend:'pytorch',device:'mps:0'}});", context);
assert.equal(engine.value, 'pytorch|mps', 'reconnect displays server identity');
assert.equal(stopPerform.disabled, false);
assert.equal(transportStart.disabled, false);
vm.runInContext("handlePipelineMessage({type:'pipeline_phase_change',phase:'error',error:'drain failed',decoder:null});", context);
assert.equal(start.disabled, true);
assert.equal(stopPerform.disabled, false, 'cleanup can be retried');
console.log('Dual decoder selection, lifecycle and reconnect checks passed');

// A different corpus changes the offered model, not the model used to decode its latents.
vm.runInContext("pipelinePhase='idle'; decoderCommandPending=false; setPipelineCorpusDir('/corpus/rack');", context);
assert.equal(start.disabled, true);
vm.runInContext("onDecoderList({corpus_dir:'/corpus/old',vae_id:'same_s',decoders:[]});", context);
assert.equal(start.disabled, true, 'stale availability must be ignored');
vm.runInContext("onDecoderList({corpus_dir:'/corpus/rack',vae_id:'stable_audio_open',default_window:8,decoders:[{backend:'pytorch',device:'cpu',selectable:true,label:'SAO CPU'},{backend:'pytorch',device:'mps',selectable:true,label:'SAO MPS'}]});", context);
assert.equal(elements.get('perform-decoder-engine').value, 'pytorch|mps');
assert.equal(elements.get('perform-vae-weight-group').hidden, true);
assert.ok(elements.get('perform-decoder-engine').options.every(o => !o.value.startsWith('onnxruntime')));
start.listeners.click();
assert.equal(sent.at(-1).config.corpus_dir, '/corpus/rack');
assert.equal(sent.at(-1).config.decoder_window, 8);
assert.equal(sent.at(-1).config.decoder_backend, 'pytorch');
vm.runInContext("pipelinePhase='idle'; decoderCommandPending=false; setPipelineCorpusDir('/corpus/ear'); onDecoderList({corpus_dir:'/corpus/ear',vae_id:'ear_vae_48k',decoders:[{backend:'pytorch',device:'cpu',selectable:true,label:'EAR CPU'}]});", context);
assert.equal(elements.get('perform-vae-weight-group').hidden, false);
elements.get('perform-vae-weight-path').value = '/models/ear/weights.pyt';
start.listeners.click();
assert.equal(sent.at(-1).config.vae_weight_path, '/models/ear/weights.pyt');
console.log('Corpus-specific VAE selection and EAR weight path passed');

// Phase 0: replay recorded production discovery through fresh clients (reload)
// and reconnect state. This is a DOM harness, not a live-browser observation.
const inventory = JSON.parse(fs.readFileSync('docs/multi_vae_phase0_inventory.json', 'utf8'));
function inventoryClient() {
    const nodes = new Map(ids.map(id => [id, new FakeElement(id)]));
    const client = vm.createContext({console, wsConnected:true, ws:{send() {}}, document:{
        readyState:'loading', getElementById:id => nodes.get(id) || null,
        createElement:() => new FakeElement(), querySelector:() => null,
        querySelectorAll:() => [], addEventListener() {}
    }});
    vm.runInContext(fs.readFileSync('web/pipeline.js','utf8'),client);
    vm.runInContext('setupPipelineControls();',client);
    return {client,nodes};
}
function loadDiscovery(target, key) {
    target.client.response = inventory.discovery[key];
    vm.runInContext("handlePipelineMessage({type:'pipeline_state',phase:'idle',corpus_dir:response.corpus_dir}); handlePipelineMessage(response);",target.client);
}
for (let reload=0; reload<2; reload++) {
    const target = inventoryClient();
    loadDiscovery(target,'same_s');
    const menu = target.nodes.get('perform-decoder-engine');
    assert.ok(menu.options.some(o=>o.value==='onnxruntime|cpu' && !o.disabled), 'SAME-S ONNX survives reload');
    assert.ok(menu.options.some(o=>o.value==='pytorch|cpu'), 'SAME-S native CPU is visible');
    menu.value='pytorch|cpu';
    vm.runInContext("pipelineDisconnected(); handlePipelineMessage({type:'pipeline_state',phase:'idle',corpus_dir:response.corpus_dir}); handlePipelineMessage(response);",target.client);
    assert.equal(menu.value,'pytorch|cpu','refresh/reconnect preserves native CPU selection');
    loadDiscovery(target,'stable_audio_open');
    assert.ok(menu.options.some(o=>o.value==='pytorch|cpu'));
    assert.ok(!menu.options.some(o=>o.value==='onnxruntime|cpu'),'SAME-S ONNX must not decode Rack');
    loadDiscovery(target,'same_s');
    assert.ok(menu.options.some(o=>o.value==='onnxruntime|cpu' && !o.disabled),'SAME-S ONNX returns after switching back');
    const unavailable = JSON.parse(JSON.stringify(inventory.discovery.same_s));
    unavailable.decoders.find(c=>c.backend==='onnxruntime').selectable=false;
    unavailable.decoders.find(c=>c.backend==='onnxruntime').detail='Packaged graph missing';
    target.client.response=unavailable;
    vm.runInContext('handlePipelineMessage(response);',target.client);
    assert.ok(menu.options.some(o=>o.value==='onnxruntime|cpu' && o.disabled),'missing graph remains visible as unavailable');
}
console.log('Phase 0 recorded discovery: SAME-S/Rack, reload, reconnect and missing-artifact states passed');

// Phase 2 uses real recorded production responses from the published Rack artifact.
const phase2 = JSON.parse(fs.readFileSync('docs/multi_vae_phase2_web.json', 'utf8'));
for (const run of phase2.runs) {
    const target = inventoryClient();
    target.client.response = run.discovery;
    vm.runInContext("handlePipelineMessage({type:'pipeline_state',phase:'idle',corpus_dir:response.corpus_dir}); handlePipelineMessage(response);",target.client);
    const menu = target.nodes.get('perform-decoder-engine');
    for (const value of ['onnxruntime|cpu','pytorch|cpu','pytorch|mps']) {
        assert.ok(menu.options.some(o=>o.value===value && !o.disabled), `Rack ${value} available`);
    }
    menu.value='onnxruntime|cpu';
    vm.runInContext("pipelineDisconnected(); handlePipelineMessage({type:'pipeline_state',phase:'idle',corpus_dir:response.corpus_dir}); handlePipelineMessage(response);",target.client);
    assert.equal(menu.value,'onnxruntime|cpu','Rack ONNX survives reconnect');
}
console.log('Phase 2 recorded Rack ONNX/native choices and offline server restart passed');

// Phase 3 production responses come from the default environment without DAC.
for (const variant of ['44k','48k']) {
    const evidence = JSON.parse(fs.readFileSync(`docs/multi_vae_phase3_web_${variant}.json`, 'utf8'));
    for (const run of evidence.runs) {
        const target = inventoryClient();
        target.client.response = run.discovery;
        vm.runInContext("handlePipelineMessage({type:'pipeline_state',phase:'idle',corpus_dir:response.corpus_dir}); handlePipelineMessage(response);", target.client);
        const menu = target.nodes.get('perform-decoder-engine');
        assert.ok(menu.options.some(o=>o.value==='onnxruntime|cpu' && !o.disabled), `EAR ${variant} ONNX works without DAC`);
        assert.ok(menu.options.some(o=>o.value==='pytorch|cpu'), `EAR ${variant} native CPU remains visible`);
        menu.value='onnxruntime|cpu';
        vm.runInContext("pipelineDisconnected(); handlePipelineMessage({type:'pipeline_state',phase:'idle',corpus_dir:response.corpus_dir}); handlePipelineMessage(response);", target.client);
        assert.equal(menu.value,'onnxruntime|cpu', `EAR ${variant} selection survives reconnect`);
    }
}
console.log('Phase 3 recorded EAR 44k/48k discovery, offline restart and reconnect passed');
