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
    "perform-decoder-status", "perform-decoder-window", "val-decoder-window",
    "perform-start-btn", "btn-start", "pp-vae-select", "pp-vae-path-group",
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
assert.deepEqual(windows.options.map(option => option.value), ["2", "4", "8", "16", "32"]);
assert.equal(windows.value, "2");

vm.runInContext(
    "handlePipelineMessage({type:'pipeline_state',phase:'idle',corpus_dir:'/corpus/demo'});",
    context,
);
assert.equal(start.disabled, false, "selected corpus enables Start without validation");
assert.equal(transportStart.disabled, true, "transport waits for perform handoff");

start.listeners.click();
assert.deepEqual(sent.pop(), {
    type: "pipeline_start_perform",
    config: { corpus_dir: "/corpus/demo", decoder_window: 2 },
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
