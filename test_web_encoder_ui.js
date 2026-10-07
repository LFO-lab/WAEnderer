const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
for (const id of ['pp-vae-select', 'pp-vae-path-group', 'pp-model-status', 'pp-start-btn', 'pp-progress-text']) {
    elements.set(id, {value:'same_s', options:[], selectedIndex:0, textContent:'', disabled:false,
        addEventListener(){}, classList:{toggle(){}}, appendChild(option){this.options.push(option);},
        set innerHTML(value){this.options=[];}});
}
const context = vm.createContext({document:{readyState:'loading', addEventListener(){}, getElementById:id=>elements.get(id),
    createElement:()=>({dataset:{}})}, console});
vm.runInContext(fs.readFileSync('web/pipeline.js','utf8'),context);
vm.runInContext("pipelineConnectionReady=true; onVAEList({vaes:[{vae_id:'same_s',display_name:'SAME-S',availability:{ready:false,detail:'Weights missing'}}]});",context);
assert.equal(elements.get('pp-start-btn').disabled,true);
assert.equal(elements.get('pp-model-status').textContent,'Weights missing');
vm.runInContext("onVAEList({vaes:[{vae_id:'same_s',display_name:'SAME-S',availability:{ready:true,detail:'Cached'}}]});",context);
assert.equal(elements.get('pp-start-btn').disabled,false);
vm.runInContext("pipelinePhase='preprocess'; updateDecoderControls();",context);
assert.equal(elements.get('pp-start-btn').disabled,true);
console.log('Encoder availability UI passed');

vm.runInContext("pipelinePhase='idle'; onVAEList({vaes:[{vae_id:'same_s',availability:{ready:true,download_required:true,detail:'Will download'}}]});",context);
assert.equal(elements.get('pp-start-btn').disabled,false);
vm.runInContext("onPreprocessProgress({event:'model_download',detail:'Downloading model.safetensors'});",context);
assert.equal(elements.get('pp-progress-text').textContent,'Downloading model.safetensors');

vm.runInContext("showEncoderError('Authentication failed. Run hf auth login');",context);
assert.match(elements.get('pp-progress-text').textContent,/Error: Authentication failed/);
