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
        - 3D manual-space cursor/trajectory
        - All control values
        - Decoder state (gain, underruns)

    Supports late-binding: nav and decoder can be None at construction time
    and set later via bind_nav_decoder() when the perform phase starts.
    """

    def __init__(
        self,
        nav=None,
        decoder=None,
        fps: float = 30.0,
        on_exit_request: Optional[Callable[[str], None]] = None,
        message_handler: Optional[Callable[[dict], bool]] = None,
        extra_state_provider: Optional[Callable[[], dict]] = None,
        manual_points_3d: Optional[np.ndarray] = None,
        manual_file_ids: Optional[np.ndarray] = None,
        manual_fader_p01: Optional[np.ndarray] = None,
        manual_fader_p99: Optional[np.ndarray] = None,
        pipeline_message_handler: Optional[Callable[[dict], bool]] = None,
    ):
        """
        Initialize broadcaster.

        Args:
            nav: Navigation engine instance (can be None for late binding)
            decoder: DecoderPlayer instance (optional, can be None for late binding)
            fps: Target broadcast rate in frames per second
            on_exit_request: Optional callback invoked when web UI sends {"type":"exit"}.
            message_handler: Optional callback for custom inbound messages.
            extra_state_provider: Optional callback that returns extra state fields.
            pipeline_message_handler: Optional callback for pipeline_* messages.
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
        self._pipeline_message_handler = pipeline_message_handler
        self._manual_points_3d = None
        self._manual_points_3d_norm = None
        self._manual_points_3d_min = None
        self._manual_points_3d_range = None
        self._manual_color_values_norm = None
        self._manual_file_ids = None
        self._is_latent_nav = False
        self._ZZ_2d = None
        self._ZZ_2d_norm = None
        self._ZZ_2d_min = None
        self._ZZ_2d_range = None
        self._projection_matrix = None

        self._setup_manual_space(
            manual_points_3d,
            manual_file_ids,
            manual_fader_p01,
            manual_fader_p99,
        )

        if self.nav is not None:
            self._bind_nav_projections()

    def _bind_nav_projections(self):
        """Set up 2D projections from the bound nav engine."""
        self._is_latent_nav = hasattr(self.nav, 'GG') and hasattr(self.nav, 'geometry')

        if self._is_latent_nav:
            self._ZZ_2d = self.nav.geometry.project_to_2d(self.nav.GG)
            self._projection_matrix = self.nav.geometry.pca_components_2d
        else:
            return

        self._ZZ_2d_min = self._ZZ_2d.min(axis=0)
        self._ZZ_2d_range = self._ZZ_2d.max(axis=0) - self._ZZ_2d_min
        self._ZZ_2d_range = np.maximum(self._ZZ_2d_range, 1e-6)
        self._ZZ_2d_norm = (self._ZZ_2d - self._ZZ_2d_min) / self._ZZ_2d_range

    def bind_nav_decoder(
        self,
        nav,
        decoder,
        message_handler: Optional[Callable[[dict], bool]] = None,
        extra_state_provider: Optional[Callable[[], dict]] = None,
        manual_points_3d: Optional[np.ndarray] = None,
        manual_file_ids: Optional[np.ndarray] = None,
        manual_fader_p01: Optional[np.ndarray] = None,
        manual_fader_p99: Optional[np.ndarray] = None,
    ):
        """Late-bind nav engine and decoder after perform phase starts."""
        self.nav = nav
        self.decoder = decoder
        if message_handler is not None:
            self._message_handler = message_handler
        if extra_state_provider is not None:
            self._extra_state_provider = extra_state_provider
        self._setup_manual_space(
            manual_points_3d,
            manual_file_ids,
            manual_fader_p01,
            manual_fader_p99,
        )
        self._bind_nav_projections()

    def broadcast_pipeline_message(self, data: dict):
        """Push a pipeline message to all connected clients from any thread."""
        if not self._clients or self._loop is None:
            return
        msg = json.dumps(data)
        async def _send():
            disconnected = set()
            for client in self._clients.copy():
                try:
                    await client.send(msg)
                except Exception:
                    disconnected.add(client)
            self._clients -= disconnected
        try:
            self._loop.call_soon_threadsafe(asyncio.ensure_future, _send())
        except RuntimeError:
            pass

    def _setup_manual_space(
        self,
        manual_points_3d: Optional[np.ndarray],
        manual_file_ids: Optional[np.ndarray],
        manual_fader_p01: Optional[np.ndarray],
        manual_fader_p99: Optional[np.ndarray],
    ):
        if manual_points_3d is None:
            return
        points = np.asarray(manual_points_3d, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
            print(
                f"[ws] Ignoring invalid manual embedding shape: {getattr(points, 'shape', None)}"
            )
            return

        points_xyz = points[:, :3].astype(np.float32)
        self._manual_points_3d = points_xyz

        if points.shape[1] >= 4:
            color_axis = points[:, 3].astype(np.float32)
            c01 = float(np.percentile(color_axis, 1.0))
            c99 = float(np.percentile(color_axis, 99.0))
            c_range = max(c99 - c01, 1e-6)
            self._manual_color_values_norm = np.clip(
                (color_axis - c01) / c_range,
                0.0,
                1.0,
            )

        p01 = None
        p99 = None
        if manual_fader_p01 is not None and manual_fader_p99 is not None:
            p01 = np.asarray(manual_fader_p01, dtype=np.float32).reshape(-1)
            p99 = np.asarray(manual_fader_p99, dtype=np.float32).reshape(-1)
            if p01.shape[0] < 3 or p99.shape[0] < 3:
                p01 = None
                p99 = None
            else:
                p01 = p01[:3]
                p99 = p99[:3]

        if p01 is not None and p99 is not None:
            self._manual_points_3d_min = p01
            self._manual_points_3d_range = np.maximum(p99 - p01, 1e-6)
            self._manual_points_3d_norm = np.clip(
                (points_xyz - self._manual_points_3d_min) / self._manual_points_3d_range,
                0.0,
                1.0,
            )
        else:
            self._manual_points_3d_min = points_xyz.min(axis=0)
            self._manual_points_3d_range = np.maximum(
                points_xyz.max(axis=0) - self._manual_points_3d_min, 1e-6
            )
            self._manual_points_3d_norm = (
                points_xyz - self._manual_points_3d_min
            ) / self._manual_points_3d_range

        if manual_file_ids is not None:
            fids = np.asarray(manual_file_ids, dtype=np.int32).reshape(-1)
            if fids.shape[0] == points_xyz.shape[0]:
                self._manual_file_ids = fids
        if self._manual_file_ids is None and hasattr(self.nav, "_file_ids"):
            fids = np.asarray(self.nav._file_ids, dtype=np.int32).reshape(-1)
            if fids.shape[0] == points_xyz.shape[0]:
                self._manual_file_ids = fids

    def _manual_point_norm_for_index(self, idx: int) -> Optional[np.ndarray]:
        if self._manual_points_3d_norm is None or self._manual_points_3d_norm.shape[0] == 0:
            return None
        idx_i = int(np.clip(int(idx), 0, self._manual_points_3d_norm.shape[0] - 1))
        return self._manual_points_3d_norm[idx_i]

    def _manual_point_norm_for_fractional(
        self,
        idx_lower: int,
        idx_upper: int,
        frac: float,
    ) -> Optional[np.ndarray]:
        lower = self._manual_point_norm_for_index(idx_lower)
        upper = self._manual_point_norm_for_index(idx_upper)
        if lower is None or upper is None:
            return None
        frac_f = float(np.clip(float(frac), 0.0, 1.0))
        return ((1.0 - frac_f) * lower + frac_f * upper).astype(np.float32)

    def _manual_trajectory_norm_for_indices(self, indices) -> list:
        if self._manual_points_3d_norm is None or self._manual_points_3d_norm.shape[0] == 0:
            return []
        points = []
        n_points = self._manual_points_3d_norm.shape[0]
        for idx in indices[-64:]:
            idx_i = int(np.clip(int(idx), 0, n_points - 1))
            pt = self._manual_points_3d_norm[idx_i]
            points.append([float(pt[0]), float(pt[1]), float(pt[2])])
        return points

    def _get_state_json(self) -> str:
        """Get current state as JSON string."""
        if self.nav is None:
            # No nav engine bound yet (pre-perform phase)
            state = {"type": "state", "timestamp": time.time(), "navigation": {"mode": "idle"}}
            if self._extra_state_provider is not None:
                try:
                    extra = self._extra_state_provider()
                    if isinstance(extra, dict):
                        state.update(extra)
                except Exception:
                    pass
            return json.dumps(state)

        nav_state = self.nav.get_state()

        frac_state = nav_state.get("fractional", {})
        n_points = self._manual_points_3d_norm.shape[0] if self._manual_points_3d_norm is not None else int(self.nav.N)
        current_idx = int(round(nav_state["policy_index"]))
        current_idx = max(0, min(current_idx, max(n_points - 1, 0)))

        # Build state message
        state = {
            "type": "state",
            "timestamp": time.time(),
            "navigation": {
                "index": current_idx,
                "index_normalized": current_idx / max(1, self.nav.N - 1),
                "velocity": nav_state["policy_velocity"],
                "file_id": nav_state["current_file_id"],
                "fractional": frac_state if frac_state else None,
                "timbre_swap": nav_state.get("timbre_swap", {}),
                "recompose": nav_state.get("recompose", {}),
                "policy_v2": nav_state.get("policy_v2", {}),
                "reorganized": nav_state.get("reorganized", {}),
                "mode": "random",
            },
            "controls": nav_state["controls"],
        }

        if self._manual_points_3d_norm is not None:
            pos_3d = None
            if frac_state and frac_state.get("frac", 0.0) > 0.0:
                pos_3d = self._manual_point_norm_for_fractional(
                    frac_state.get("idx_lower", current_idx),
                    frac_state.get("idx_upper", current_idx),
                    frac_state.get("frac", 0.0),
                )
            if pos_3d is None:
                pos_3d = self._manual_point_norm_for_index(current_idx)
            if pos_3d is not None:
                state["navigation"]["position_3d"] = [
                    float(pos_3d[0]),
                    float(pos_3d[1]),
                    float(pos_3d[2]),
                ]
            state["navigation"]["trajectory_3d"] = self._manual_trajectory_norm_for_indices(
                nav_state.get("recent_indices", [])
            )

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
                    manual = dict(manual_state)
                    if self._manual_points_3d_norm is not None:
                        pos_raw = manual.get("position")
                        pos_norm = None
                        if (
                            isinstance(pos_raw, (list, tuple))
                            and len(pos_raw) >= 3
                        ):
                            pos_arr = np.asarray(pos_raw[:3], dtype=np.float32)
                            pos_norm = (
                                (pos_arr - self._manual_points_3d_min)
                                / self._manual_points_3d_range
                            )
                            pos_norm = np.clip(pos_norm, 0.0, 1.0)
                        else:
                            idx = int(manual.get("nearest_index", -1))
                            if 0 <= idx < self._manual_points_3d_norm.shape[0]:
                                pos_norm = self._manual_points_3d_norm[idx]
                        if pos_norm is not None:
                            manual["position_3d"] = [
                                float(pos_norm[0]),
                                float(pos_norm[1]),
                                float(pos_norm[2]),
                            ]
                    state["manual"] = manual

        return json.dumps(state)
    
    def _get_corpus_json(self) -> str:
        """Get corpus data for initial visualization setup."""
        if self.nav is None:
            return json.dumps({"type": "corpus", "total_points": 0, "point_indices": [], "file_ids": []})

        n_points = (
            int(self._manual_points_3d_norm.shape[0])
            if self._manual_points_3d_norm is not None
            else int(self.nav.N)
        )
        if n_points > 2000:
            indices = np.linspace(0, n_points - 1, 2000, dtype=int)
        else:
            indices = np.arange(n_points, dtype=int)
        file_ids = self.nav._file_ids[indices].tolist()

        manual_positions = []
        manual_color_values = []
        manual_file_ids = file_ids
        if self._manual_points_3d_norm is not None and self._manual_points_3d_norm.shape[0] == n_points:
            manual_positions = self._manual_points_3d_norm[indices].tolist()
            if self._manual_color_values_norm is not None:
                manual_color_values = (
                    self._manual_color_values_norm[indices].astype(np.float32).tolist()
                )
            if self._manual_file_ids is not None:
                manual_file_ids = self._manual_file_ids[indices].tolist()

        nav_mode = "random"
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
            "point_indices": indices.tolist(),
            "file_ids": file_ids,
            "manual_positions_3d": manual_positions,
            "manual_color_values": manual_color_values,
            "manual_file_ids": manual_file_ids,
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

            # Route pipeline_* messages to the pipeline handler
            if msg_type.startswith("pipeline_"):
                if self._pipeline_message_handler is not None:
                    try:
                        self._pipeline_message_handler(data)
                    except Exception as e:
                        print(f"[ws] Error in pipeline message handler: {e}")
                return

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
            
            if msg_type in ("random_control", "control"):
                # Random control update (with backward-compatible alias "control")
                controls = data.get("controls", {})
                if isinstance(controls, dict):
                    self.nav.set_random_controls(**controls)

            elif msg_type == "reorganized_control":
                controls = data.get("controls", {})
                if isinstance(controls, dict):
                    self.nav.set_reorganized_controls(**controls)

            elif msg_type == "cursor_index":
                idx = data.get("index")
                if idx is not None:
                    self.nav.set_cursor_index(idx)
                    
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
    nav=None,
    decoder=None,
    host: str = "127.0.0.1",
    port: int = 8765,
    fps: float = 30.0,
    on_exit_request: Optional[Callable[[str], None]] = None,
    message_handler: Optional[Callable[[dict], bool]] = None,
    extra_state_provider: Optional[Callable[[], dict]] = None,
    manual_points_3d: Optional[np.ndarray] = None,
    manual_file_ids: Optional[np.ndarray] = None,
    manual_fader_p01: Optional[np.ndarray] = None,
    manual_fader_p99: Optional[np.ndarray] = None,
    pipeline_message_handler: Optional[Callable[[dict], bool]] = None,
) -> Tuple[WSBroadcaster, threading.Thread]:
    """
    Start a WebSocket server for visualization.

    Args:
        nav: Navigation engine instance (can be None for late binding)
        decoder: DecoderPlayer instance (optional, can be None for late binding)
        host: Server host address
        port: Server port
        fps: Broadcast rate in frames per second
        on_exit_request: Optional callback invoked when web UI requests exit.
        message_handler: Optional callback for custom inbound WebSocket messages.
        extra_state_provider: Optional callback adding fields to broadcast state payloads.
        manual_points_3d: Optional manual embedding corpus points [N,D>=3] for shared 3D rendering.
            Dim 0..2 are XYZ; dim 3 (if present) is used as a color scalar.
        manual_file_ids: Optional file ids aligned with manual_points_3d.
        manual_fader_p01: Optional manual control-space lower bounds used for normalization.
        manual_fader_p99: Optional manual control-space upper bounds used for normalization.
        pipeline_message_handler: Optional callback for pipeline_* inbound messages.

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
        manual_points_3d=manual_points_3d,
        manual_file_ids=manual_file_ids,
        manual_fader_p01=manual_fader_p01,
        manual_fader_p99=manual_fader_p99,
        pipeline_message_handler=pipeline_message_handler,
    )
    thread = broadcaster.start(host, port)
    return broadcaster, thread
