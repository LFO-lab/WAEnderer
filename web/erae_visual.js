/* Optional view ownership and bridge-rendered preview; no musical commands. */
let eraeIdentity=null, eraeAssembly=null, eraeOwner=0, eraeMine=false;
let eraeFullGeometry=false;
let eraeRevision=0, eraeLastView='', eraeLastSend=0, eraeLastPreview=0;
function eraePickingReady() { return !eraeFullGeometry || eraeIdentity!==null; }
function eraeSelectionIdentity() { return eraeIdentity ? {visual_session:eraeIdentity.session,visual_corpus:eraeIdentity.corpus} : {}; }
function eraeSend(message) { if(ws && wsConnected) ws.send(JSON.stringify(message)); }
function eraeSubscribe() {
    eraeDisconnected();
    eraeSend({type:'erae_subscribe',version:1,role:'browser'});
}
function eraeDisconnected() {
    eraeIdentity=null; eraeAssembly=null; eraeMine=false;
    document.getElementById('erae-status').textContent='Visual connection unavailable.';
}
function eraePublishView() {
    const now=Date.now();
    if(eraeLastPreview && now-eraeLastPreview>2000) {
        document.getElementById('erae-status').textContent='Bridge preview stale.';
        document.getElementById('erae-preview').style.opacity='.3';
    }
    if(!eraeMine || !eraeIdentity || drawMode!=='perform' || now-eraeLastSend<1000/30) return;
    const view=manualCameraSnapshot(), encoded=JSON.stringify(view);
    if(encoded===eraeLastView && now-eraeLastSend<500) return;
    if(encoded!==eraeLastView) eraeRevision++;
    eraeLastView=encoded; eraeLastSend=now;
    eraeSend({type:'erae_view',...eraeIdentity,owner:eraeOwner,revision:eraeRevision,view});
}
function eraeHandleMessage(data) {
    if(data.type==='corpus' && eraeIdentity) return true;
    if(!String(data.type||'').startsWith('erae_')) return false;
    switch(data.type) {
    case 'erae_unavailable':
        eraeIdentity=null; eraeAssembly=null;
        document.getElementById('erae-status').textContent='No synchronized corpus. Enable --erae-osc.';
        break;
    case 'erae_geometry_begin':
        eraeIdentity=null;
        if(!Number.isInteger(data.count) || data.count<0 || data.count>1000000) break;
        eraeAssembly={session:data.session,corpus:data.corpus,count:data.count,rows:[],started:Date.now()};
        break;
    case 'erae_geometry_chunk': {
        const a=eraeAssembly;
        if(!a || Date.now()-a.started>30000 || data.offset!==a.rows.length || !Array.isArray(data.rows)
            || data.rows.length>512 || a.rows.length+data.rows.length>a.count) {eraeAssembly=null;break;}
        if(data.rows.some(r=>!Array.isArray(r)||r.length!==6||!Number.isInteger(r[0])||r[0]<0||r[0]>=a.count
            ||!r.slice(1,4).every(Number.isFinite)||!(r[4]===null||Number.isFinite(r[4]))||!Number.isInteger(r[5])||r[5]<0)) {
            eraeAssembly=null;break;
        }
        a.rows.push(...data.rows); break;
    }
    case 'erae_geometry_end': {
        const a=eraeAssembly; eraeAssembly=null;
        if(!a || a.rows.length!==a.count || new Set(a.rows.map(r=>r[0])).size!==a.count) break;
        manualCorpusPoints3D=a.rows.map(r=>r.slice(1,4));
        manualCorpusIndices=a.rows.map(r=>r[0]);
        manualCorpusFileIds=a.rows.map(r=>r[5]);
        manualCorpusColorValues=a.rows.every(r=>r[4]!==null)?a.rows.map(r=>r[4]):[];
        eraeIdentity={session:a.session,corpus:a.corpus}; eraeFullGeometry=true;
        eraeLastView=''; eraeLastSend=0;
        break;
    }
    case 'erae_owner':
        if (eraeOwner!==data.owner) {eraeRevision=0; eraeLastView=''; eraeLastSend=0;}
        eraeOwner=data.owner; eraeMine=data.mine;
        document.getElementById('erae-owner').textContent=data.mine?'This view controls Erae.':(data.available?'Another browser owns the Erae view.':'Select a view to enable zone 0.');
        break;
    case 'erae_preview': {
        if(!Array.isArray(data.pixels)||data.pixels.length!==768) break;
        const canvas=document.getElementById('erae-preview'), ctx=canvas.getContext('2d');
        const image=ctx.createImageData(32,24);
        data.pixels.forEach((p,i)=>{image.data.set([...p,255],i*4);});
        ctx.putImageData(image,0,0); canvas.style.opacity='1';
        document.getElementById('erae-status').textContent=data.status;
        eraeLastPreview=Date.now();break;
    }
    }
    return true;
}
document.getElementById('erae-claim').addEventListener('click',()=>eraeSend({type:'erae_claim',takeover:true}));
document.getElementById('erae-release').addEventListener('click',()=>eraeSend({type:'erae_release'}));
