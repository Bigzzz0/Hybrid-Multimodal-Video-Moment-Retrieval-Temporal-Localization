from __future__ import annotations

import gc
import subprocess
import time
from typing import Any, Dict

import torch

from app.core.config import settings
from app.core.logger import logger
from pe_inference_worker.pe_core_service import PECoreService


class PEModelManager:
    def __init__(self) -> None:
        self.service: PECoreService | None = None
        self.model_id = ""
        self.state = "unloaded"
        self.current_task = ""
        self.loaded_at = 0.0
        self.release_count = 0
        self.peak_vram_mb = 0.0
        self.timings: Dict[str, float] = {}

    def _used_vram_mb(self) -> float:
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                used = float(total - free) / (1024 * 1024)
                self.peak_vram_mb = max(self.peak_vram_mb, used)
                return used
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=True,
            )
            used = float(result.stdout.strip().splitlines()[0].strip())
            self.peak_vram_mb = max(self.peak_vram_mb, used)
            return used
        except Exception:
            return 0.0

    def _free_vram_mb(self) -> float:
        try:
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                return float(free) / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def get(self, model_id: str) -> PECoreService:
        if self.service is not None and self.model_id == model_id:
            self.state = "ready"
            return self.service
        self.unload()
        self.state = "loading"
        started = time.perf_counter()
        try:
            service = PECoreService(model_id)
            service.load()
            self.service = service
            self.model_id = model_id
            self.loaded_at = time.time()
            used = self._used_vram_mb()
            if used > float(settings.PE_CORE_VRAM_BUDGET_MB):
                self.unload()
                raise RuntimeError(f"PE-Core VRAM budget exceeded ({settings.PE_CORE_VRAM_BUDGET_MB} MB)")
            self.timings["load_ms"] = round((time.perf_counter() - started) * 1000, 1)
            self.state = "ready"
            return service
        except Exception:
            self.state = "error"
            raise

    def unload(self) -> None:
        started = time.perf_counter()
        service = self.service
        self.service = None
        self.model_id = ""
        if service is not None:
            try:
                service.unload()
            except Exception as exc:
                logger.debug("PE-Core unload failed: {}", exc)
            self.release_count += 1
        self.state = "unloaded"
        self.loaded_at = 0.0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.timings["unload_ms"] = round((time.perf_counter() - started) * 1000, 1)

    def telemetry(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "model_id": self.model_id,
            "loaded_at": self.loaded_at,
            "free_vram_mb": round(self._free_vram_mb(), 1),
            "used_vram_mb": round(self._used_vram_mb(), 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "budget_mb": int(settings.PE_CORE_VRAM_BUDGET_MB),
            "current_task": self.current_task,
            "release_count": self.release_count,
            "timings_ms": dict(self.timings),
        }


model_manager = PEModelManager()
