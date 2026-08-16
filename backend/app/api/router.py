from fastapi import APIRouter
from app.api.v1 import upload, search, video, websocket, rag

api_router = APIRouter()

api_router.include_router(upload.router, prefix="/videos", tags=["Upload & Ingestion"])
api_router.include_router(search.router, prefix="/search", tags=["Moment Search"])
api_router.include_router(video.router, prefix="/videos", tags=["Video Management & Streaming"])
api_router.include_router(rag.router, prefix="/rag", tags=["Video-RAG QA"])
api_router.include_router(websocket.router, tags=["WebSocket Telemetry"])
