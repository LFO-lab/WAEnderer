import asyncio
import json
import time

from stable_audio_wanderer.runtime.erae_visual import VisualRelay, geometry_packets

VIEW=dict(yaw=0,pitch=0,distance=2.6,width=800,height=600,colour_a=[0,0,0],colour_b=[255,255,255],files=[[255,0,0]])

class Socket:
    def __init__(self): self.sent=[]
    async def send(self,packet): self.sent.append(json.loads(packet))
    async def close(self): pass


def test_visual_owner_snapshot_revision_and_rebind():
    async def run():
        relay=VisualRelay(); a,b,bridge=Socket(),Socket(),Socket()
        packets=geometry_packets('engine','corpus',[[.5,.5,.5]]*2100,None,[0]*2100)
        relay.bind(('engine','corpus'),packets)
        for ws,role in ((a,'browser'),(b,'browser'),(bridge,'bridge')):
            await relay.handle(ws,dict(type='erae_subscribe',version=1,role=role))
        await relay.handle(a,dict(type='erae_claim'))
        owner=relay.generation
        await relay.handle(b,dict(type='erae_claim'))
        assert relay.owner is a
        packet=dict(type='erae_view',session='engine',corpus='corpus',owner=owner,revision=1,view=VIEW)
        await relay.handle(a,packet)
        assert relay.view is not None
        await relay.handle(a,{**packet,'revision':0,'view':{**VIEW,'yaw':40}})
        assert relay.view['view']['yaw']==0
        await asyncio.sleep(.1)
        assert any(m['type']=='erae_geometry_end' for m in bridge.sent)
        assert sum(len(m['rows']) for m in bridge.sent if m['type']=='erae_geometry_chunk')==2100
        await relay.handle(b,dict(type='erae_claim',takeover=True))
        assert relay.owner is b and relay.view is None
        await relay.handle(a,packet)
        assert relay.view is None
        relay.bind(None)
        assert relay.view is None and relay.identity is None
        for ws in (a,b,bridge): relay.remove(ws)
        await asyncio.sleep(0)
    asyncio.run(run())


def test_bounded_coalescing_and_invalid_camera():
    async def run():
        relay=VisualRelay(); ws=Socket()
        relay.bind(('engine','corpus'),())
        await relay.handle(ws,dict(type='erae_subscribe',version=1,role='browser'))
        await relay.handle(ws,dict(type='erae_claim'))
        for i in range(100):
            await relay.handle(ws,dict(type='erae_view',session='engine',corpus='corpus',owner=relay.generation,revision=i,view=VIEW))
        assert len(relay.peers[ws]['queue'])<=3
        await relay.handle(ws,dict(type='erae_view',session='engine',corpus='corpus',owner=relay.generation,revision=101,view={**VIEW,'width':0}))
        assert relay.view_sequence==99
        relay.remove(ws)
        await asyncio.sleep(0)
    asyncio.run(run())


def test_real_websocket_osc_bridge_preview_selection_and_disconnect():
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    from websockets import connect
    from websockets.server import serve
    from stable_audio_wanderer.runtime.ws_server import WSBroadcaster
    from stable_audio_wanderer.runtime.erae_osc import EraeOscServer
    from test_erae_osc import Controller
    bridge_path=Path(__file__).resolve().parents[2]/'WAEnderer_erae'
    if not bridge_path.exists():
        import pytest
        pytest.skip('companion bridge checkout required')
    sys.path.insert(0,str(bridge_path))
    from waenderer_erae.config import Config
    from waenderer_erae.osc_client import OscClient
    from waenderer_erae.performance import Performance
    from waenderer_erae.runtime import HardwareRuntime
    from waenderer_erae.erae_device import FakeMidiBackend
    from waenderer_erae.touch import TouchEvent

    async def until(predicate, timeout=3):
        deadline=time.monotonic()+timeout
        while not predicate():
            assert time.monotonic()<deadline, 'loopback condition timed out'
            await asyncio.sleep(.01)

    async def run():
        engine=EraeOscServer(port=0).start(); controller=Controller(); engine.bind(controller)
        broadcaster=WSBroadcaster()
        broadcaster.nav=SimpleNamespace(N=100,_file_ids=np.zeros(100,dtype=int))
        broadcaster._manual_points_3d_norm=np.array([[i/100,.5,.5] for i in range(100)])
        broadcaster._loop=asyncio.get_running_loop()
        broadcaster.bind_visual(engine.session,engine.revision)
        client=OscClient(engine.address,renew=.2,log=lambda _:None)
        cfg=Config(erae_in='Erae Fake In',erae_out='Erae Fake Out')
        cfg.visual.stale_seconds=.3
        backend=FakeMidiBackend(['Erae Fake In'],['Erae Fake Out'])
        perf=Performance(cfg,backend,client,log=lambda _:None)
        hardware=HardwareRuntime(cfg,backend,on_sample=perf.observe,on_reset=perf.reset,log=lambda _:None)
        try:
            async with serve(broadcaster._handle_client,'127.0.0.1',0) as server:
                cfg.visual.url=f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'
                await client.start(port=0)
                await hardware.start(); await perf.start(hardware)
                async with connect(cfg.visual.url) as browser:
                    await browser.send(json.dumps(dict(type='erae_subscribe',version=1,role='browser')))
                    await browser.send(json.dumps(dict(type='erae_claim',takeover=True)))
                    previews=[]
                    async def browser_pump():
                        async for message in browser:
                            packet=json.loads(message)
                            if packet.get('type')=='erae_preview':
                                previews.append(packet['status'])
                            if packet.get('type')=='erae_owner' and packet.get('mine'):
                                await browser.send(json.dumps(dict(type='erae_view',session=engine.session,
                                    corpus=engine.revision,owner=packet['owner'],revision=1,view=VIEW)))
                    reader=asyncio.create_task(browser_pump())
                    await until(lambda: perf.zone0.usable())
                    perf.observe(TouchEvent(0,1,'down',.25,.5,.5,(8,12,.5),time.monotonic()))
                    await until(lambda: bool(controller.selections))
                    selected=controller.selections[-1]
                    assert selected != controller.state['index']
                    perf.observe(TouchEvent(0,1,'up',.25,.5,0,(8,12,0),time.monotonic()))
                    controller.state['index']=80
                    await until(lambda: any('playback=80' in status for status in previews))
                    assert perf.zone0.requested==selected
                    assert hardware._frames.get(0)
                    reader.cancel(); await asyncio.gather(reader,return_exceptions=True)
                await until(lambda: perf.visual.state.view is None)
                await until(lambda: perf.zone0.requested is None)
                assert client.state.fresh() and controller.selections==[selected]
                engine.bind(Controller())
                broadcaster.bind_visual(engine.session,engine.revision)
                await until(lambda: perf.visual.state.identity==(engine.session,engine.revision))
                assert not perf.zone0.usable()
        finally:
            await perf.close(); await hardware.close(); await client.close(); engine.close()
            for ws in list(broadcaster.visual.peers): broadcaster.visual.remove(ws)
    asyncio.run(run())
