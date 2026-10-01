/* Pure camera/picker helpers shared by the browser and parity fixtures. */
(function(root) {
    function world(v, c) {
        const yaw=c.yaw*Math.PI/180, pitch=c.pitch*Math.PI/180;
        const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
        const y=v[1]*cp+v[2]*sp, z=-v[1]*sp+v[2]*cp;
        return [v[0]*cy-z*sy, y, v[0]*sy+z*cy];
    }
    function project(p,c) {
        const [x,y,z]=p.map(v=>(v-.5)*2);
        const yaw=c.yaw*Math.PI/180, pitch=c.pitch*Math.PI/180;
        const x1=x*Math.cos(yaw)+z*Math.sin(yaw), z1=-x*Math.sin(yaw)+z*Math.cos(yaw);
        const y2=y*Math.cos(pitch)-z1*Math.sin(pitch), depth=y*Math.sin(pitch)+z1*Math.cos(pitch)+c.distance;
        const perspective=1/Math.max(.25,depth);
        return {x:c.width*.5+x1*perspective*c.width*.9,y:c.height*.5-y2*perspective*c.height*.9,depth,perspective};
    }
    function ray(mx,my,c) {
        const origin=world([0,0,-c.distance],c);
        const raw=world([(mx-c.width*.5)/(c.width*.9),-(my-c.height*.5)/(c.height*.9),1],c);
        const mag=Math.sqrt(raw.reduce((n,v)=>n+v*v,0));
        return {origin,dir:raw.map(v=>v/mag)};
    }
    function pick(points,mx,my,c) {
        const r=ray(mx,my,c); let best=-1, score=Infinity;
        points.forEach((p,i)=>{
            const d=p.map((v,j)=>(v-.5)*2-r.origin[j]);
            const t=d.reduce((n,v,j)=>n+v*r.dir[j],0);
            if(t<=0) return;
            const s=d.reduce((n,v,j)=>n+(v-t*r.dir[j])**2,0);
            if(s<score) {score=s;best=i;}
        });
        return best;
    }
    const api={world,project,ray,pick};
    if(typeof module!=='undefined') module.exports=api;
    root.EraeMath=api;
})(globalThis);
