const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

test('OSC camera updates move existing controls once and preserve local changes', () => {
    const elements = new Map();
    const context = vm.createContext({
        document: {getElementById(id) {
            if (!elements.has(id)) elements.set(id, {classList: {toggle() {}}});
            return elements.get(id);
        }},
        eraeHandleMessage: () => true,
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/sketch.js'), 'utf8'), context);
    const send = updates => {
        context.message = {type: 'state', osc_input: {camera_updates: updates}};
        vm.runInContext('handleMessage(message)', context);
    };
    const updates = {
        left: {revision: 1, value: true}, right: {revision: 1, value: false},
        speed: {revision: 2, value: 1.5},
    };
    send(updates);
    assert.equal(vm.runInContext('manualCameraToggles.left', context), true);
    assert.equal(vm.runInContext('manualCameraSpeed', context), 1.5);
    assert.equal(elements.get('val-cam-speed').textContent, '1.50');
    vm.runInContext('setManualCameraToggle("right", true); manualCameraSpeed = 0.5', context);
    send(updates);
    assert.equal(vm.runInContext('manualCameraToggles.right', context), true);
    assert.equal(vm.runInContext('manualCameraSpeed', context), 0.5);
    send({left: {revision: 3, value: false}, right: {revision: 3, value: false}});
    assert.equal(vm.runInContext('manualCameraToggles.right', context), false);
});
