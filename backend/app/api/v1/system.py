import os
import sys
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter
from app.core.config import settings
from app.db.connection import db_manager

router = APIRouter()

@router.get("/telemetry", response_model=Dict[str, Any])
def get_system_telemetry():
    """
    Returns real-time GPU, CPU, RAM, LanceDB storage, and AI Model telemetry for Developer Panel.
    """
    # 1. GPU Telemetry
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_name": "CPU Only",
        "device_count": 0,
        "allocated_vram_mb": 0.0,
        "reserved_vram_mb": 0.0,
        "total_vram_mb": 0.0,
        "compute_capability": "N/A"
    }

    if torch.cuda.is_available():
        gpu_info["device_count"] = torch.cuda.device_count()
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        
        # Query true physical VRAM from NVIDIA CUDA Driver / NVML
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        used_bytes = total_bytes - free_bytes
        
        gpu_info["used_vram_mb"] = round(used_bytes / (1024 ** 2), 1)
        gpu_info["allocated_vram_mb"] = round(used_bytes / (1024 ** 2), 1)
        gpu_info["free_vram_mb"] = round(free_bytes / (1024 ** 2), 1)
        gpu_info["total_vram_mb"] = round(total_bytes / (1024 ** 2), 1)
        gpu_info["reserved_vram_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1)
        
        cc = torch.cuda.get_device_capability(0)
        gpu_info["compute_capability"] = f"sm_{cc[0]}{cc[1]}"

    # 2. CPU & System RAM
    vm = psutil.virtual_memory()
    sys_info = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / (1024 ** 3), 1),
        "ram_used_gb": round(vm.used / (1024 ** 3), 1),
        "ram_percent": vm.percent,
        "platform": sys.platform,
        "python_version": sys.version.split()[0]
    }

    # 3. LanceDB Database Storage Stats
    tables_stats = {}
    total_records = 0
    table_names = ["videos", "video_frames", "scenes", "transcripts", "search_logs"]
    for t_name in table_names:
        try:
            tbl = db_manager.get_table(t_name)
            count = tbl.count_rows()
            tables_stats[t_name] = count
            total_records += count
        except Exception:
            tables_stats[t_name] = 0

    # 4. Model Registry Configuration
    models_info = {
        "visual_encoder": {
            "name": "SigLIP 2 (NaFlex)",
            "model_id": settings.SIGLIP2_MODEL_ID,
            "embedding_dim": 768,
            "acceleration": "CUDA FP16 / Tensor Cores" if torch.cuda.is_available() else "CPU"
        },
        "audio_asr": {
            "name": "Faster-Whisper",
            "model_size": settings.WHISPER_MODEL_SIZE,
            "engine": "CTranslate2 (CUDA 12 FP16)" if torch.cuda.is_available() else "CPU int8",
            "beam_size": 5,
            "vad_filter": True
        },
        "dense_captioner": {
            "name": "Qwen2.5-VL-7B-Instruct",
            "model_id": settings.QWEN_VL_MODEL_ID,
            "quantization": "4-bit NormalFloat (NF4) BitsAndBytes"
        },
        "temporal_localizer": {
            "name": "2D-TAN + 1D Wasserstein NMS",
            "iou_threshold": 0.5,
            "top_k": 5,
            "gaussian_sigma": getattr(settings, "TEMPORAL_GAUSSIAN_SIGMA", 1.5)
        }
    }

    # 5. Recent Log Snippet
    log_lines = []
    log_path = settings.DATA_DIR / "app.log"
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                log_lines = [l.strip() for l in lines[-40:] if l.strip()]
        except Exception:
            pass

    return {
        "status": "healthy",
        "gpu": gpu_info,
        "system": sys_info,
        "lancedb": {
            "tables": tables_stats,
            "total_records": total_records,
            "storage_path": str(settings.LANCEDB_DIR)
        },
        "models": models_info,
        "recent_logs": log_lines
    }
