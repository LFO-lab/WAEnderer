#!/usr/bin/env node
'use strict';
const assert=require('node:assert/strict'), fs=require('node:fs'), vm=require('node:vm');
const sent=[], elements=new Map();
function element(id) {
    if(!elements.has(id)) elements.set(id,{textContent:'',style:{},listeners:{},
        addEventListener(k,f){this.listeners[k]=f;},
        getContext(){return {createImageData(){return {data:new Uint8ClampedArray(32*24*4)};},putImageData(i){this.image=i;}};}});
    return elements.get(id);
}
const c=vm.createContext({console,Date,document:{getElementById:element},capture:p=>sent.push(JSON.parse(p))});
vm.runInContext(fs.readFileSync('web/erae_math.js','utf8'),c);
vm.runInContext(fs.readFileSync('web/sketch.js','utf8'),c);
vm.runInContext(fs.readFileSync('web/erae_visual.js','utf8'),c);
vm.runInContext('ws={send:capture}; wsConnected=true; width=800; height=600; eraeSubscribe();',c);
assert.equal(sent.pop().type,'erae_subscribe');
element('erae-claim').listeners.click(); assert.deepEqual(sent.pop(),{type:'erae_claim',takeover:true});
const rows=Array.from({length:2101},(_,i)=>[i,.1,.1,.1,null,0]); rows[2100]=[2100,.5,.5,.5,null,0];
function feed(data){c.packet=data;vm.runInContext('handleMessage(packet)',c);}
feed({type:'erae_geometry_begin',session:'engine',corpus:'corpus',count:rows.length});
for(let i=0;i<rows.length;i+=512) feed({type:'erae_geometry_chunk',offset:i,rows:rows.slice(i,i+512)});
assert.equal(vm.runInContext('manualCorpusPoints3D.length',c),0,'partial geometry must not be activated');
feed({type:'erae_geometry_end'});
assert.equal(vm.runInContext('findClosestManualPointIndex(400,300)',c),2100);
vm.runInContext('sendCursorFor3DPoint(2100)',c); assert.deepEqual(sent.pop(),{type:'cursor_index',index:2100,visual_session:'engine',visual_corpus:'corpus'});
feed({type:'erae_owner',owner:2,available:true,mine:true});vm.runInContext('eraePublishView()',c);
assert.equal(sent.at(-1).type,'erae_view'); const revision=sent.at(-1).revision;
feed({type:'erae_owner',owner:2,available:true,mine:true});
assert.equal(vm.runInContext('eraeRevision',c),revision,'another subscriber must not roll back the owner revision');
feed({type:'corpus',manual_positions_3d:[],point_indices:[]});
assert.equal(vm.runInContext('manualCorpusPoints3D.length',c),2101,'legacy snapshots must not replace full geometry');
feed({type:'erae_preview',pixels:Array.from({length:768},()=>[0,0,0]),status:'live'});
assert.equal(element('erae-status').textContent,'live');
feed({type:'erae_owner',owner:3,available:false,mine:false});
assert.match(element('erae-owner').textContent,/Select a view/);
feed({type:'erae_unavailable'});
assert.equal(vm.runInContext('findClosestManualPointIndex(400,300)',c),-1);
console.log('Erae browser ownership, full geometry, picker and preview passed');
