"""
WebSocket server for real-time visualization of navigation state.
Broadcasts state updates at ~30fps and receives control messages from web clients.
"""
import asyncio
from contextlib import suppress
import json
import threading
import time
from typing import Callable, Optional, Set, Tuple
import numpy as np


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
        - All control values
        - Decoder state (gain, smoothing, underruns)
    """

    def __init__(
        self,
        nav,
        decoder=None,
        fps: float = 30.0,
        on_exit_request: Optional[Callable[[str], None]] = None,
        message_handler: Optional[Callable[[dict], bool]] = None,
        extra_state_provider: Optional[Callable[[], dict]] = None,
    ):
        """
        Initialize broadcaster.

        Args:
            nav: Navigation engine instance
            decoder: DecoderPlayer instance (optional)
            fps: Target broadcast rate in frames per second
            on_exit_request: Optional callback invoked when web UI sends {"type":"exit"}.
            message_handler: Optional callback for custom inbound messages.
            extra_state_provider: Optional callback that returns extra state fields.
        """
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required. Install with: pip install websockets")

        self.nav = nav
        self.decoder = decoder
        self.fps = fps
        self.interval = 1.0 / fps

        self._clients: Set = set()
        self._running = False
        self._server = None
        self._loop = None
        self._thread = None
        self._stop_event = None
        self._on_exit_request = on_exit_request
        self._message_handler = message_handler
        self._extra_state_provider = extra_state_provider

        # Precompute 2D projection if corpus is higher dimensional
        self._projection_matrix = None
        self._setup_projection()
    
    def _setup_projection(self):
        """Setup 2D projection for visualization."""
        # Check if this is a LatentNavigationEngine (has GG and geometry)
        self._is_latent_nav = hasattr(self.nav, 'GG') and hasattr(self.nav, 'geometry')

        if self._is_latent_nav:
            # For latent nav, use the stored PCA projection
            self._ZZ_2d = self.nav.geometry.project_to_2d(self.nav.GG)
            self._projection_matrix = self.nav.geometry.pca_components_2d
        elif self.nav.ZZ.shape[1] > 2:
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

        # Get fractional state for smooth interpolation
        frac_state = nav_state.get("fractional", {})

        # For latent navigation mode, use the actual continuous latent position
        # This provides smooth cursor movement instead of snapping to corpus points
        if self._is_latent_nav and "latent" in nav_state and "position_2d" in nav_state.get("latent", {}):
            raw_pos_2d = np.array(nav_state["latent"]["position_2d"])
            # Normalize to [0,1] using the same stats as corpus normalization
            pos_2d_normalized = (raw_pos_2d - self._ZZ_2d_min) / self._ZZ_2d_range
            pos_2d = [float(np.clip(p, 0.0, 1.0)) for p in pos_2d_normalized]
            current_idx = int(round(nav_state["policy_index"]))
            current_idx = max(0, min(current_idx, len(self._ZZ_2d_norm) - 1))
        # If fractional state is available, interpolate for smooth cursor movement
        elif frac_state and frac_state.get("frac", 0.0) > 0.0:
            idx_lower = frac_state["idx_lower"]
            idx_upper = frac_state["idx_upper"]
            frac = frac_state["frac"]
            # Clamp indices to valid range
            idx_lower = max(0, min(idx_lower, len(self._ZZ_2d_norm) - 1))
            idx_upper = max(0, min(idx_upper, len(self._ZZ_2d_norm) - 1))
            # Interpolate 2D position for smooth cursor movement
            pos_lower = self._ZZ_2d_norm[idx_lower]
            pos_upper = self._ZZ_2d_norm[idx_upper]
            pos_2d = ((1.0 - frac) * pos_lower + frac * pos_upper).tolist()
            current_idx = idx_lower  # For compatibility
        else:
            current_idx = int(round(nav_state["policy_index"]))
            current_idx = max(0, min(current_idx, len(self._ZZ_2d_norm) - 1))
            pos_2d = self._ZZ_2d_norm[current_idx].tolist()

        # Get 2D positions for recent trajectory
        # For latent mode, use the actual continuous 2D trajectory if available
        if self._is_latent_nav and "latent" in nav_state and "trajectory_2d" in nav_state.get("latent", {}):
            raw_trajectory = nav_state["latent"]["trajectory_2d"]
            # Normalize trajectory points using same stats as corpus
            trajectory_2d = []
            for pt in raw_trajectory[-64:]:  # Last 64 points
                pt_arr = np.array(pt)
                pt_norm = (pt_arr - self._ZZ_2d_min) / self._ZZ_2d_range
                trajectory_2d.append([float(np.clip(p, 0.0, 1.0)) for p in pt_norm])
        else:
            # Use corpus positions for trajectory
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
                "file_id": nav_state["current_file_id"],
                "fractional": frac_state if frac_state else None,
                "mode": "policy",
            },
            "controls": nav_state["controls"],
        }

        # Add latent-specific state if available
        if self._is_latent_nav and "latent" in nav_state:
            state["navigation"]["latent"] = nav_state["latent"]
        
        # Add decoder state if available
        if self.decoder is not None:
            state["decoder"] = self.decoder.get_state()

        if self._extra_state_provider is not None:
            try:
                extra = self._extra_state_provider()
            except Exception as e:
                print(f"[ws] Error getting extra state: {e}")
                extra = None
            if isinstance(extra, dict):
                transport = extra.get("transport")
                if isinstance(transport, dict):
                    state["transport"] = transport
                nav_mode = extra.get("navigation_mode")
                if isinstance(nav_mode, str):
                    state["navigation"]["mode"] = nav_mode
                manual_state = extra.get("manual")
                if isinstance(manual_state, dict):
                    state["manual"] = manual_state

        return json.dumps(state)
    
    def _get_corpus_json(self) -> str:
        """Get corpus data for initial visualization setup."""
        # Sample corpus points for visualization (max 2000 for performance)
        n_points = len(self._ZZ_2d_norm)
        if n_points > 2000:
            indices = np.linspace(0, n_points - 1, 2000, dtype=int)
            positions = self._ZZ_2d_norm[indices].tolist()
            file_ids = self.nav._file_ids[indices].tolist()
        else:
            positions = self._ZZ_2d_norm.tolist()
            file_ids = self.nav._file_ids.tolist()

        nav_mode = "policy"
        if self._extra_state_provider is not None:
            try:
                extra = self._extra_state_provider()
                if isinstance(extra, dict) and isinstance(extra.get("navigation_mode"), str):
                    nav_mode = extra["navigation_mode"]
            except Exception:
                pass

        return json.dumps({
            "type": "corpus",
            "total_points": n_points,
            "positions_2d": positions,
            "file_ids": file_ids,
            "navigation_mode": nav_mode,
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

            if msg_type in ("transport", "manual_controls"):
                if self._message_handler is not None:
                    try:
                        handled = bool(self._message_handler(data))
                        if handled:
                            return
                    except Exception as e:
                        print(f"[ws] Error in custom message handler: {e}")
                # Message types are recognized even if no handler is attached.
                print(f"[ws] Ignored message type without handler: {msg_type}")
                return

            if self._message_handler is not None:
                try:
                    handled = bool(self._message_handler(data))
                    if handled:
                        return
                except Exception as e:
                    print(f"[ws] Error in custom message handler: {e}")
            
            if msg_type == "control":
                # Update navigation controls
                controls = data.get("controls", {})
                self.nav.set_policy_controls(**controls)
                
            elif msg_type == "cursor":
                # Update cursor position
                coords = data.get("coords", [])
                if coords:
                    # For latent mode, de-normalize [0,1] coords back to raw PCA space
                    # so the inverse projection works correctly
                    if self._is_latent_nav and len(coords) >= 2:
                        coords_arr = np.array(coords[:2], dtype=np.float32)
                        coords_raw = coords_arr * self._ZZ_2d_range + self._ZZ_2d_min
                        coords = coords_raw.tolist()
                    self.nav.set_cursor_nd(coords)
                    
            elif msg_type == "reset":
                # Reset policy state
                idx = data.get("index")
                self.nav.reset_policy(idx=idx)
                
            elif msg_type == "decoder":
                params = data.get("params", {})
                if self.decoder is not None:
                    for key, value in params.items():
                        setter = getattr(self.decoder, f"set_{key}", None)
                        if setter is not None:
                            try:
                                setter(value)
                            except Exception as e:
                                print(f"[ws] Error setting decoder.{key}: {e}")

            elif msg_type == "request_corpus":
                # Client requesting corpus data
                await websocket.send(self._get_corpus_json())
            elif msg_type == "exit":
                reason = data.get("reason", "websocket")
                print(f"[ws] Exit requested by client (reason={reason})")
                if self._on_exit_request is not None:
                    try:
                        self._on_exit_request(str(reason))
                    except Exception as e:
                        print(f"[ws] Error handling exit request: {e}")

        except json.JSONDecodeError:
            print(f"[ws] Invalid JSON message: {message[:100]}")
        except Exception as e:
            print(f"[ws] Error handling message: {e}")
    
    async def _broadcast_loop(self):
        """Broadcast state to all connected clients."""
        while self._running:
            if self._clients:
                try:
                    state_json = self._get_state_json()
                except Exception as e:
                    print(f"[ws] Error building state JSON: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(self.interval)
                    continue

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
        self._stop_event = asyncio.Event()

        async with serve(self._handle_client, host, port) as server:
            self._server = server
            print(f"[ws] Server started on ws://{host}:{port}")
            broadcast_task = asyncio.create_task(self._broadcast_loop())
            try:
                await self._stop_event.wait()
            finally:
                self._running = False
                broadcast_task.cancel()
                with suppress(asyncio.CancelledError):
                    await broadcast_task
                self._server = None
    
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
        
        self._thread = threading.Thread(target=run_in_thread, daemon=True, name="ws-server")
        self._thread.start()
        return self._thread
    
    def shutdown(self, timeout: float = 2.0):
        """Shutdown the WebSocket server."""
        self._running = False
        if self._loop is not None and self._loop.is_running() and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)


def start_ws_server(
    nav,
    decoder=None,
    host: str = "127.0.0.1",
    port: int = 8765,
    fps: float = 30.0,
    on_exit_request: Optional[Callable[[str], None]] = None,
    message_handler: Optional[Callable[[dict], bool]] = None,
    extra_state_provider: Optional[Callable[[], dict]] = None,
) -> Tuple[WSBroadcaster, threading.Thread]:
    """
    Start a WebSocket server for visualization.

    Args:
        nav: Navigation engine instance
        decoder: DecoderPlayer instance (optional)
        host: Server host address
        port: Server port
        fps: Broadcast rate in frames per second
        on_exit_request: Optional callback invoked when web UI requests exit.
        message_handler: Optional callback for custom inbound WebSocket messages.
        extra_state_provider: Optional callback adding fields to broadcast state payloads.

    Returns:
        Tuple of (WSBroadcaster, Thread)
    """
    broadcaster = WSBroadcaster(
        nav,
        decoder,
        fps=fps,
        on_exit_request=on_exit_request,
        message_handler=message_handler,
        extra_state_provider=extra_state_provider,
    )
    thread = broadcaster.start(host, port)
    return broadcaster, thread
