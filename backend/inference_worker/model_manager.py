from __future__ import annotations

import gc
import subprocess
import time
from typing import Any, Dict

from app.core.config import settings
from app.core.logger import logger


class ModelManager:
    """Owns worker model lifecycle and exposes conservative VRAM telemetry."""

    def __init__(self) -> None:
        self.services: Dict[str, Any] = {}
        self.states: Dict[str, str] = {"siglip": "unloaded", "qwen": "unloaded", "sam": "unloaded"}
        self.loaded_at: Dict[str, float] = {}

    def _free_vram_mb(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                free_mb = float(free) / (1024 * 1024)
                if free_mb > 0:
                    return free_mb
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            return float(result.stdout.strip().splitlines()[0].strip())
        except Exception:
            pass
        return 0.0

    def _used_vram_mb(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                used_mb = max(0.0, float(total - free) / (1024 * 1024))
                if used_mb > 0:
                    return used_mb
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            return float(result.stdout.strip().splitlines()[0].strip())
        except Exception:
            pass
        return 0.0

    def telemetry(self) -> Dict[str, Any]:
        return {
            "states": dict(self.states),
            "loaded_at": dict(self.loaded_at),
            "free_vram_mb": round(self._free_vram_mb(), 1),
            "budget_mb": int(settings.MODEL_VRAM_BUDGET_MB),
        }

    def get(self, name: str, factory):
        if name in self.services:
            self.states[name] = "ready"
            return self.services[name]
        # Qwen and SAM are both large multimodal models. On a 12 GB card,
        # loading them together can leave no activation headroom even when
        # their weights individually fit. Keep those two heavy models
        # mutually exclusive, but keep SigLIP resident alongside Qwen: its
        # text encoder is small enough to fit and reloading it for every
        # action query makes Accurate needlessly slow.
        if name == "qwen":
            self.unload("sam")
        elif name == "sam":
            self.unload("qwen")
            # SAM 3.1 is the model that nearly fills this 12 GB card. Evict
            # SigLIP before its allocation so the checkpoint can load safely.
            self.unload("siglip")
        if self._free_vram_mb() and self._free_vram_mb() < float(settings.VLM_MIN_FREE_VRAM_MB):
            self.unload_inactive(except_name=name)
        self.states[name] = "loading"
        try:
            service = factory()
            service.load()
            self.services[name] = service
            self.loaded_at[name] = time.time()
            # Keep the newest requested model and evict older residents until
            # the physical VRAM budget is respected.  If a single model alone
            # exceeds the budget, fail this request instead of destabilizing
            # the worker process.
            if self._used_vram_mb() > float(settings.MODEL_VRAM_BUDGET_MB):
                for victim in sorted(self.loaded_at, key=self.loaded_at.get):
                    if victim != name and self._used_vram_mb() > float(settings.MODEL_VRAM_BUDGET_MB):
                        self.unload(victim)
            if self._used_vram_mb() > float(settings.MODEL_VRAM_BUDGET_MB):
                self.unload(name)
                raise RuntimeError(f"VRAM budget exceeded ({settings.MODEL_VRAM_BUDGET_MB} MB)")
            self.states[name] = "ready"
            return service
        except Exception:
            self.states[name] = "error"
            raise

    def unload(self, name: str) -> None:
        service = self.services.pop(name, None)
        if service is not None:
            try:
                service.unload()
            except Exception as exc:
                logger.debug("Worker model unload failed for {}: {}", name, exc)
        self.states[name] = "unloaded"
        self.loaded_at.pop(name, None)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload_inactive(self, except_name: str = "") -> None:
        for name in list(self.services):
            if name != except_name:
                self.unload(name)


model_manager = ModelManager()
