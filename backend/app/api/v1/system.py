import os
import sys
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter
from app.core.config import settings
from app.db.connection import db_manager
from app.inference.client import inference_client

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
    table_names = [
        "videos",
        "video_frames_v2",
        "scenes_v2",
        "search_logs",
        "sam_tracks_v1",
        "sam_observations_v1",
        "scene_analysis_v1",
        "model_artifact_cache",
    ]
    for t_name in table_names:
        try:
            tbl = db_manager.get_table(t_name)
            count = tbl.count_rows()
            tables_stats[t_name] = count
            total_records += count
        except Exception:
            tables_stats[t_name] = 0
    try:
        metadata_table = db_manager.get_index_metadata_table()
        tables_stats[db_manager.index_metadata_table_name] = metadata_table.count_rows()
        total_records += tables_stats[db_manager.index_metadata_table_name]
    except Exception:
        tables_stats["index_metadata"] = 0

    # 4. Model Registry Configuration
    models_info = {
        "visual_encoder": {
            "name": "SigLIP 2 (NaFlex)",
            "model_id": settings.SIGLIP2_MODEL_ID,
            "embedding_dim": settings.SIGLIP2_EMBEDDING_DIM,
            "index_version": settings.VISUAL_INDEX_VERSION,
            "acceleration": "CUDA FP16 / Tensor Cores" if torch.cuda.is_available() else "CPU"
        },
        "visual_temporal_reranker": {
            "name": "Qwen3-VL-2B-Instruct (top-3)",
            "model_id": settings.QWEN_VL_MODEL_ID,
            "quantization": "4-bit NormalFloat (NF4) BitsAndBytes"
        },
        "sam_grounder": {
            "name": "SAM 3.1 text-grounded segmentation",
            "model_id": settings.SAM_MODEL_ID,
            "grounding_version": settings.GROUNDING_VERSION,
            "compile": settings.SAM_COMPILE,
        },
        "temporal_localizer": {
            "name": "Multi-scale visual proposals + Soft-NMS",
            "iou_threshold": 0.5,
            "top_k": 5,
            "calibrated": (settings.CALIBRATION_ARTIFACT_PATH.exists() if hasattr(settings, "CALIBRATION_ARTIFACT_PATH") else False)
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
        "visual_index": db_manager.validate_visual_index(),
        "models": models_info,
        "inference_worker": inference_client.health(),
        "recent_logs": log_lines
    }
