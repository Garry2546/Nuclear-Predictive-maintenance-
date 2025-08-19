from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List

websocketRouter = APIRouter()

class WebSocketManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connections."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove disconnected WebSockets."""
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send message to all connected WebSockets."""
        for connection in self.active_connections:
            await connection.send_json(message)

# Global WebSocket Manager
ws_manager = WebSocketManager()
ws_accident_data_manager = WebSocketManager()

@websocketRouter.websocket("/ws/sensor_data_predictions")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@websocketRouter.websocket("/ws/accident_data_predictions")
async def accident_data_websocket_endpoint(websocket: WebSocket):
    await ws_accident_data_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_accident_data_manager.disconnect(websocket)