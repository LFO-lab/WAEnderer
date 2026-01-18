"""
WebSocket server for real-time visualization of navigation state.
Broadcasts state updates at ~30fps and receives control messages from web clients.
"""
import asyncio
import json
import threading
import time
from typing import Optional, Set, Tuple
import numpy as np

from ..config import LATENT_HZ

try:
    import websockets
    from websockets.server import serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


class WSBroadcaster:
    """
    WebSocket broadcaster for navigation state.

    Broadcasts:
        - Current index (normalized 0-1)
        - Recent trajectory (last 128 indices)
        - 2D projection of current position
        - Current regime (0=drift, 1=turn, 2=linger)
        - All control values
        - Grain player state
        - Scheduler state (multi-stream granular)
    """

    def __init__(self, nav, grain_player=None, scheduler=None, fps: float = 30.0):
        """
        Initialize broadcaster.

        Args:
            nav: NavigationEngine instance
            grain_player: GrainPlayer instance (optional)
            scheduler: GrainScheduler instance (optional)
            fps: Target broadcast rate in frames per second
        """
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required. Install with: pip install websockets")

        self.nav = nav
        self.grain_player = grain_player
        self.scheduler = scheduler
        self.fps = fps
        self.interval = 1.0 / fps

        self._clients: Set = set()
        self._running = False
        self._server = None
        self._loop = None

        # Precompute 2D projection if corpus is higher dimensional
        self._projection_matrix = None
        self._setup_projection()
    
    def _setup_projection(self):
        """Setup 2D projection for visualization."""
        if self.nav.ZZ.shape[1] > 2:
            # Use PCA for projection to 2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            self._ZZ_2d = pca.fit_transform(self.nav.ZZ)
            self._projection_matrix = pca.components_
        else:
            self._ZZ_2d = self.nav.ZZ[:, :2].copy()
            self._projection_matrix = None
        
        # Normalize to [0, 1]
        self._ZZ_2d_min = self._ZZ_2d.min(axis=0)
        self._ZZ_2d_range = self._ZZ_2d.max(axis=0) - self._ZZ_2d_min
        self._ZZ_2d_range = np.maximum(self._ZZ_2d_range, 1e-6)
        self._ZZ_2d_norm = (self._ZZ_2d - self._ZZ_2d_min) / self._ZZ_2d_range
    
    def _get_state_json(self) -> str:
        """Get current state as JSON string."""
        nav_state = self.nav.get_state()
        
        # Get 2D position for current index
        current_idx = int(round(nav_state["policy_index"]))
        current_idx = max(0, min(current_idx, len(self._ZZ_2d_norm) - 1))
        pos_2d = self._ZZ_2d_norm[current_idx].tolist()
        
        # Get 2D positions for recent trajectory
        recent_indices = nav_state["recent_indices"]
        trajectory_2d = [
            self._ZZ_2d_norm[max(0, min(int(i), len(self._ZZ_2d_norm) - 1))].tolist()
            for i in recent_indices[-64:]  # Last 64 points for visualization
        ]
        
        # Build state message
        state = {
            "type": "state",
            "timestamp": time.time(),
            "navigation": {
                "index": current_idx,
                "index_normalized": current_idx / max(1, self.nav.N - 1),
                "position_2d": pos_2d,
                "trajectory_2d": trajectory_2d,
                "velocity": nav_state["policy_velocity"],
                "regime": nav_state["policy_regime"],
                "file_id": nav_state["current_file_id"],
            },
            "controls": nav_state["controls"],
            "grain": {
                "rate": nav_state["grain_rate"],
                "jitter": nav_state["grain_jitter"],
            },
        }
        
        # Add grain player state if available
        if self.grain_player is not None:
            state["grain"].update(self.grain_player.get_state())

        # Add scheduler state if available
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.get_state()

        return json.dumps(state)
    
    def _get_corpus_json(self) -> str:
        """Get corpus data for initial visualization setup."""
        # Sample corpus points for visualization (max 2000 for performance)
        n_points = len(self._ZZ_2d_norm)
        if n_points > 2000:
            indices = np.linspace(0, n_points - 1, 2000, dtype=int)
            positions = self._ZZ_2d_norm[indices].tolist()
            file_ids = self.nav._file_ids[indices].tolist()
            regimes = self.nav.regime_table[indices].tolist() if self.nav.regime_table is not None else [0] * len(indices)
        else:
            positions = self._ZZ_2d_norm.tolist()
            file_ids = self.nav._file_ids.tolist()
            regimes = self.nav.regime_table.tolist() if self.nav.regime_table is not None else [0] * n_points
        
        return json.dumps({
            "type": "corpus",
            "total_points": n_points,
            "positions_2d": positions,
            "file_ids": file_ids,
            "regimes": regimes,
        })
    
    async def _handle_client(self, websocket):
        """Handle a single WebSocket client connection."""
        self._clients.add(websocket)
        print(f"[ws] Client connected ({len(self._clients)} total)")
        
        try:
            # Send corpus data on connect
            await websocket.send(self._get_corpus_json())
            
            # Handle incoming messages
            async for message in websocket:
                await self._handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            print(f"[ws] Client disconnected ({len(self._clients)} total)")
    
    async def _handle_message(self, websocket, message: str):
        """Handle incoming control message from client."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")
            
            if msg_type == "control":
                # Update navigation controls
                controls = data.get("controls", {})
                self.nav.set_policy_controls(**controls)
                
            elif msg_type == "cursor":
                # Update cursor position
                coords = data.get("coords", [])
                if coords:
                    self.nav.set_cursor_nd(coords)
                    
            elif msg_type == "reset":
                # Reset policy state
                idx = data.get("index")
                self.nav.reset_policy(idx=idx)
                
            elif msg_type == "grain":
                # Update grain parameters
                params = data.get("params", {})
                for key, value in params.items():
                    # Enforce minimum rate of LATENT_HZ for trigger_rate
                    if key == "trigger_rate":
                        value = max(LATENT_HZ, value)
                        self.nav.set_grain_rate(value)
                    elif key == "trigger_jitter":
                        self.nav.set_grain_jitter(value)

                    # Also update grain player if available
                    if self.grain_player is not None:
                        setter = getattr(self.grain_player, f"set_{key}", None)
                        if setter is not None:
                            try:
                                setter(value)
                            except Exception as e:
                                print(f"[ws] Error setting grain.{key}: {e}")

                    # Sync scheduler timing when grain_dur changes
                    if key == "grain_dur" and self.scheduler is not None:
                        try:
                            self.scheduler.set_grain_dur(value)
                        except Exception as e:
                            print(f"[ws] Error syncing scheduler grain_dur: {e}")
                            
            elif msg_type == "scheduler":
                # Update scheduler parameters
                if self.scheduler is not None:
                    params = data.get("params", {})
                    for key, value in params.items():
                        setter = getattr(self.scheduler, f"set_{key}", None)
                        if setter is not None:
                            try:
                                setter(value)
                            except Exception as e:
                                print(f"[ws] Error setting scheduler.{key}: {e}")

            elif msg_type == "request_corpus":
                # Client requesting corpus data
                await websocket.send(self._get_corpus_json())

        except json.JSONDecodeError:
            print(f"[ws] Invalid JSON message: {message[:100]}")
        except Exception as e:
            print(f"[ws] Error handling message: {e}")
    
    async def _broadcast_loop(self):
        """Broadcast state to all connected clients."""
        while self._running:
            if self._clients:
                state_json = self._get_state_json()
                # Broadcast to all clients
                disconnected = set()
                for client in self._clients.copy():
                    try:
                        await client.send(state_json)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected.add(client)
                    except Exception as e:
                        print(f"[ws] Broadcast error: {e}")
                        disconnected.add(client)
                
                # Remove disconnected clients
                self._clients -= disconnected
            
            await asyncio.sleep(self.interval)
    
    async def _run_server(self, host: str, port: int):
        """Run the WebSocket server."""
        self._running = True
        
        async with serve(self._handle_client, host, port):
            print(f"[ws] Server started on ws://{host}:{port}")
            await self._broadcast_loop()
    
    def start(self, host: str = "127.0.0.1", port: int = 8765):
        """Start the WebSocket server in a new thread."""
        def run_in_thread():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._run_server(host, port))
            except Exception as e:
                print(f"[ws] Server error: {e}")
            finally:
                self._loop.close()
        
        self._thread = threading.Thread(target=run_in_thread, daemon=True)
        self._thread.start()
        return self._thread
    
    def shutdown(self):
        """Shutdown the WebSocket server."""
        self._running = False
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)


def start_ws_server(
    nav,
    grain_player=None,
    scheduler=None,
    host: str = "127.0.0.1",
    port: int = 8765,
    fps: float = 30.0,
) -> Tuple[WSBroadcaster, threading.Thread]:
    """
    Start a WebSocket server for visualization.

    Args:
        nav: NavigationEngine instance
        grain_player: GrainPlayer instance (optional)
        scheduler: GrainScheduler instance (optional)
        host: Server host address
        port: Server port
        fps: Broadcast rate in frames per second

    Returns:
        Tuple of (WSBroadcaster, Thread)
    """
    broadcaster = WSBroadcaster(nav, grain_player, scheduler, fps=fps)
    thread = broadcaster.start(host, port)
    return broadcaster, thread
