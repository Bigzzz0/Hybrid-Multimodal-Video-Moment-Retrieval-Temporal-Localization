import json
import asyncio
from typing import Dict, Set, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.logger import logger

router = APIRouter()

class ConnectionManager:
    """Manages active WebSocket connections & in-memory progress cache for live telemetry."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.progress_cache: Dict[str, Dict[str, Any]] = {}
        self.main_loop = None

    def set_loop(self, loop):
        self.main_loop = loop

    async def connect(self, websocket: WebSocket, video_id: str):
        await websocket.accept()
        if video_id not in self.active_connections:
            self.active_connections[video_id] = set()
        self.active_connections[video_id].add(websocket)
        logger.info(f"WebSocket client connected for video_id: {video_id}")

        # Send latest cached status immediately upon connection
        if video_id in self.progress_cache:
            try:
                await websocket.send_text(json.dumps(self.progress_cache[video_id]))
            except Exception as e:
                logger.debug(f"Initial progress send error: {e}")

    def disconnect(self, websocket: WebSocket, video_id: str):
        if video_id in self.active_connections:
            self.active_connections[video_id].discard(websocket)
            if not self.active_connections[video_id]:
                del self.active_connections[video_id]
        logger.info(f"WebSocket client disconnected for video_id: {video_id}")

    def update_progress(
        self,
        video_id: str,
        progress_percent: int,
        message: str,
        stage: str = "decoding",
        phase: str = "phase1",
        stage_details: Dict[str, Any] = None
    ):
        """Thread-safe update of progress state and broadcasting."""
        data = {
            "video_id": video_id,
            "progress": progress_percent,
            "message": message,
            "stage": stage,
            "phase": phase,
            "details": stage_details or {}
        }
        self.progress_cache[video_id] = data

        payload = json.dumps(data)
        if video_id in self.active_connections:
            for connection in list(self.active_connections[video_id]):
                try:
                    if self.main_loop and self.main_loop.is_running():
                        asyncio.run_coroutine_threadsafe(connection.send_text(payload), self.main_loop)
                    else:
                        asyncio.create_task(connection.send_text(payload))
                except Exception as e:
                    logger.debug(f"Broadcast error: {e}")

    def get_progress(self, video_id: str) -> Dict[str, Any]:
        return self.progress_cache.get(video_id, {
            "video_id": video_id,
            "progress": 0,
            "message": "Initializing...",
            "stage": "decoding",
            "phase": "phase1",
            "details": {}
        })

ws_manager = ConnectionManager()

@router.websocket("/ws/progress/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    await ws_manager.connect(websocket, video_id)
    try:
        while True:
            # Keep-alive loop
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, video_id)

@router.get("/progress/{video_id}/status")
async def get_progress_status(video_id: str):
    """HTTP Polling fallback endpoint to fetch live ingestion status."""
    return ws_manager.get_progress(video_id)
