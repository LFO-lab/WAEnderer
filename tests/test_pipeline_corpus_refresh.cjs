const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

test('completion refreshes corpus options and retains the newly selected corpus', () => {
    const sent = [];
    const select = {
        options: [{value: ''}],
        selected: '',
        set value(value) {
            this.selected = this.options.some(option => option.value === value) ? value : '';
        },
        get value() { return this.selected; },
        appendChild(option) { this.options.push(option); },
        remove(index) { this.options.splice(index, 1); },
    };
    const context = vm.createContext({
        console,
        document: {
            readyState: 'loading', addEventListener() {},
            getElementById: id => id === 'perform-corpus-select' ? select : null,
            createElement: () => ({}),
        },
        capture: message => sent.push(message),
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/pipeline.js'), 'utf8'), context);
    // Keep real completion routing, corpus selection, and list rebuilding;
    // unrelated decoder and presentation work is outside this regression.
    vm.runInContext(`
        sendPipelineMessage = capture;
        requestDecoderAvailability = () => {};
        updateDecoderControls = () => {};
        applyDecoderPhase = () => {};
        updatePipelinePhaseUI = () => {};
        onPreprocessComplete = () => {};
    `, context);
    const message = data => {
        context.message = data;
        vm.runInContext('handlePipelineMessage(message)', context);
    };
    for (const completed of ['preprocess', 'train']) {
        message({type: 'pipeline_phase_change', phase: 'idle', completed, corpus_dir: '/corpus/new'});
        assert.equal(sent.at(-1).type, 'pipeline_list_corpora');
        message({type: 'pipeline_corpus_list', corpora: [
            {path: '/corpus/old', name: 'old'}, {path: '/corpus/new', name: 'new'},
        ]});
        assert.equal(select.value, '/corpus/new');
        assert.deepEqual(select.options.map(option => option.value), ['', '/corpus/old', '/corpus/new']);
    }
    assert.equal(vm.runInContext('selectedTab', context), 'perform');
    const count = sent.length;
    message({type: 'pipeline_phase_change', phase: 'idle', reason: 'cancelled'});
    message({type: 'pipeline_phase_change', phase: 'idle', error: 'Training failed'});
    assert.equal(sent.length, count);
});
