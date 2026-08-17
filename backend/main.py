import sys
import os
from pathlib import Path

# Auto-register NVIDIA CUDA DLL directories on Windows
if sys.platform == "win32":
    try:
        import site
        site_packages_dirs = site.getsitepackages()
        for sp in site_packages_dirs:
            nvidia_dir = Path(sp) / "nvidia"
            if nvidia_dir.exists():
                for bin_dir in nvidia_dir.glob("*/bin"):
                    if bin_dir.exists():
                        try:
                            os.add_dll_directory(str(bin_dir))
                        except Exception:
                            pass
    except Exception:
        pass

import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.logger import logger
from app.api.router import api_router
from app.api.v1.websocket import ws_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Loop in WS Manager and log info
    ws_manager.set_loop(asyncio.get_running_loop())
    logger.info(f"=== Starting {settings.PROJECT_NAME} ===")
    logger.info(f"Compute Device: {settings.DEVICE.upper()}")
    logger.info(f"Storage Directories: {settings.DATA_DIR}")
    yield
    logger.info("Shutting down Video Event Retrieval Server...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware (Allows Next.js frontend to communicate seamlessly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount static files for keyframe thumbnails and video streaming
app.mount("/data", StaticFiles(directory=str(settings.DATA_DIR)), name="data")

@app.get("/")
def root_status():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "device": settings.DEVICE,
        "docs": "/docs",
        "api_prefix": settings.API_V1_STR
    }

if __name__ == "__main__":
    logger.info("Starting Video Event Retrieval FastAPI Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
