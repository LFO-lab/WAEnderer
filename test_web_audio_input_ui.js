"use strict";
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
    constructor() { this.dataset = {}; this.listeners = {}; this.value = ''; this.options = []; }
    addEventListener(type, callback) { this.listeners[type] = callback; }
    replaceChildren(...items) { this.options = items; }
    add(item) { this.options.push(item); }
}
const elements = new Map();
const element = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
};
const sent = [];
const context = vm.createContext({
    document: {getElementById: element},
    Option: class { constructor(text, value) { this.text = text; this.value = value; } },
    wsConnected: true, ws: {send: message => sent.push(JSON.parse(message))},
    transportRunning: false, selectedNavigationMode: 'random',
    applyNavigationModeUI() {}, sendTransportSetMode(mode) { sent.push({mode}); },
});
vm.runInContext(fs.readFileSync('web/audio_input.js', 'utf8'), context);
vm.runInContext('setupAudioInputControls()', context);
element('mode-audio-input').listeners.click();
assert.equal(context.selectedNavigationMode, 'audio_input');
assert.deepEqual(sent.at(-1), {type: 'audio_input', action: 'devices'});
element('audio-input-device').value = '3';
element('audio-input-start').listeners.click();
assert.equal(sent.at(-1).device, 3);
context.transportRunning = true;
element('audio-input-path').listeners.change({target: {value: 'latents'}});
assert.deepEqual(sent.at(-1), {type: 'audio_input', action: 'path', path: 'latents'});
const count = sent.length;
element('mode-audio-input').listeners.click();
assert.equal(sent.length, count, 'navigation mode stays fixed during playback');
vm.runInContext(`updateAudioInputState({available:true, running:true, path:'latents',
    status:'running', level_db:-80, devices:[{id:3,name:'Microphone'}],
    paths:{latents:{status:'error',error:'<model unavailable>',age_ms:2500}}})`, context);
assert.equal(element('audio-input-device').disabled, true);
assert.match(element('audio-input-status').textContent, /silence: holding/);
assert.match(element('audio-input-diagnostics').textContent, /<model unavailable>/);
vm.runInContext('updateAudioInputState({})', context);
assert.equal(element('audio-input-start').disabled, true, 'unload disables capture');
assert.equal(element('mode-audio-input').dataset.available, undefined);
console.log('Audio Input UI checks passed');
