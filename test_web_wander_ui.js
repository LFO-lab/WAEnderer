#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class FakeClassList {
    constructor() { this.values = new Set(); }
    add(value) { this.values.add(value); }
    remove(value) { this.values.delete(value); }
    toggle(value, force) {
        if (force === undefined ? !this.values.has(value) : force) {
            this.values.add(value);
            return true;
        }
        this.values.delete(value);
        return false;
    }
    contains(value) { return this.values.has(value); }
}

class FakeElement {
    constructor(id = "") {
        this.id = id;
        this.value = "";
        this.textContent = "";
        this.disabled = false;
        this.checked = false;
        this.title = "";
        this.type = id === "wander-frame-source" ? "select-one" : "range";
        this.listeners = {};
        this.classList = new FakeClassList();
    }
    addEventListener(type, callback) { this.listeners[type] = callback; }
}

const elements = new Map();
const element = id => {
    if (!elements.has(id)) elements.set(id, new FakeElement(id));
    return elements.get(id);
};
const document = {
    getElementById(id) { return element(id); },
};
const sent = [];
const context = vm.createContext({
    console,
    document,
    constrain(value, minimum, maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    },
    setTimeout(callback) { callback(); },
    window: { confirm() { return false; } },
    capture(payload) { sent.push(JSON.parse(payload)); },
});

vm.runInContext(fs.readFileSync("web/sketch.js", "utf8"), context, {
    filename: "web/sketch.js",
});
vm.runInContext("ws = {send: capture}; wsConnected = true; setupControls();", context);

const frameSource = element("wander-frame-source");
const frameOrder = element("wander-frame-order");
const latentColour = element("wander-latent-colour");

frameSource.value = "morphology_graph";
frameSource.listeners.change({ target: frameSource });
assert.deepEqual(sent.pop(), {
    type: "wander_render",
    controls: { frame_source: "morphology_graph" },
});

frameOrder.value = "0.37";
frameOrder.listeners.input({ target: frameOrder });
assert.equal(element("val-wander-frame-order").textContent, "0.37");
assert.deepEqual(sent.pop(), {
    type: "wander_render",
    controls: { frame_order: 0.37 },
});

latentColour.value = "0.62";
latentColour.listeners.input({ target: latentColour });
assert.equal(element("val-wander-latent-colour").textContent, "0.62");
assert.deepEqual(sent.pop(), {
    type: "wander_render",
    controls: { latent_colour: 0.62 },
});

vm.runInContext(`handleMessage({
    type: 'state',
    navigation: {mode: 'random', index: 3, velocity: 0, file_id: 1},
    transport: {running: false},
    wander_render: {
        requested_frame_source: 'morphology_graph',
        effective_frame_source: 'contiguous',
        graph_available: false,
        frame_order: 0.25,
        latent_colour: 0.5,
        seed: 1198485348
    },
    decoder: {backend: 'onnxruntime', selected_window: 2}
});`, context);
assert.equal(frameSource.value, "morphology_graph");
assert.equal(frameOrder.value, "0.25");
assert.equal(element("val-wander-frame-order").textContent, "0.25");
assert.equal(latentColour.value, "0.5");
assert.equal(element("val-wander-latent-colour").textContent, "0.50");
assert.match(element("wander-render-status").textContent, /Effective: Contiguous/);
assert.match(element("wander-render-status").textContent, /requested Morphology Graph/);
assert.match(element("wander-render-status").textContent, /graph unavailable/);
assert.equal(frameOrder.disabled, false, "Frame Order remains enabled at T2");
assert.match(element("status-text").textContent, /Wander/);

element("mode-random").listeners.click();
assert.deepEqual(sent.pop(), {
    type: "transport",
    action: "set_mode",
    mode: "random",
});

element("btn-random-reset").listeners.click();
assert.deepEqual(sent.pop(), { type: "reset" });

const html = fs.readFileSync("web/index.html", "utf8");
assert.match(html, /id="mode-random"[^>]*>Wander<\/button>/);
assert.match(html, /<h2>Wander<\/h2>/);
assert.match(html, /id="btn-random-reset">Reset Wander<\/button>/);
assert.match(html, /id="wander-frame-source"/);
assert.match(html, /value="k_nearest" selected>K Nearest<\/option>/);
assert.match(html, /value="contiguous">Contiguous<\/option>/);
assert.match(html, /value="morphology_graph">Morphology Graph<\/option>/);

for (const id of ["wander-frame-order", "wander-latent-colour"]) {
    const tag = html.match(new RegExp(`<input(?=[^>]*id="${id}")[^>]*>`));
    assert.ok(tag, `${id} input exists`);
    assert.match(tag[0], /min="0"/);
    assert.match(tag[0], /max="1"/);
    assert.match(tag[0], /step="0\.01"/);
    assert.match(tag[0], /value="0"/);
    assert.doesNotMatch(tag[0], /\bdisabled\b/);
}

assert.match(html, /Latent Colour \(Dither\)/);
assert.match(html, /id="manual-dither"/, "legacy Manual Dither remains present");
assert.match(html, /id="perform-decoder-window"/, "Decoder T remains global");
assert.match(html, /<option value="random">Random Only<\/option>/, "training terminology stays compatible");

console.log("Web Wander rendering controls contract passed");
