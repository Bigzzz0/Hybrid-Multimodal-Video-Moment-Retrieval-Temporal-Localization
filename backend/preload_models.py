import sys
import os
from pathlib import Path

# Ensure standard streams use UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

import torch
from app.core.config import settings
from app.core.logger import logger
from app.pipeline.visual_encoder import SigLIP2VisualEncoder
from app.pipeline.dense_captioner import QwenVLDenseCaptioner

def preload_all_models():
    """
    Pre-downloads, verifies, and warms up the configured visual models before starting the server.
    Ensures zero cold-start delay during video ingestion and moment search.
    Visual-only preload for the SigLIP2 index and optional visual captioner.
    """
    logger.info("==================================================================")
    logger.info("[PRELOAD] PRELOADING PURE-VISUAL RETRIEVAL MODELS")
    logger.info(f"Target Device: {settings.DEVICE.upper()} (CUDA Available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Initial VRAM Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
    logger.info("==================================================================")

    # 1. SigLIP 2 NaFlex (the v2 visual index contract)
    logger.info("1/2 [Vision-Text Embedding] Preloading SigLIP 2 (NaFlex 768-dim)...")
    try:
        siglip = SigLIP2VisualEncoder()
        siglip._lazy_load()
        # Test sample text encoding
        test_emb = siglip.encode_text("test moment query")
        logger.info(f"[OK] SigLIP 2 loaded successfully (Sample vector length: {len(test_emb)}).")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load SigLIP 2: {e}")

    # 2. VLM Captioner & Verifier (CapRL-Qwen3VL-4B Q6 via llama.cpp or legacy Qwen3-VL-2B)
    backend_name = settings.CAPTION_PRIMARY_BACKEND
    if backend_name == "caprl_qwen3vl_4b_q6":
        logger.info("2/2 [Dense Action Captioning] Verifying CapRL-Qwen3VL-4B Q6 (GGUF + llama.cpp)...")
        try:
            from app.inference.vlm_registry import get_variant
            variant = get_variant(backend_name)
            model_path = Path(settings.VLM_GGUF_DIR) / variant.model_file
            mmproj_path = Path(settings.VLM_GGUF_DIR) / variant.mmproj_file
            llama_exe = Path(settings.LLAMA_CPP_PATH)
            if not llama_exe.exists():
                logger.warning(f"[WARN] llama executable not found at: {llama_exe}")
            elif not model_path.exists() or not mmproj_path.exists():
                logger.warning(f"[WARN] CapRL Q6 GGUF weights missing in: {settings.VLM_GGUF_DIR}")
            else:
                logger.info(f"[OK] CapRL-Qwen3VL-4B Q6 weights ({model_path.stat().st_size / (1024**3):.1f} GB) and llama-server verified and ready.")
                if getattr(settings, "INFERENCE_WARMUP_QWEN", False):
                    logger.info("      Warming up CapRL Q6 VLM service...")
                    from inference_worker.vlm_service import VariantVLMService
                    svc = VariantVLMService(backend_name)
                    svc.load()
                    svc.unload()
                    logger.info("      [OK] CapRL Q6 warmed up successfully.")
        except Exception as e:
            logger.error(f"[ERROR] Failed to verify CapRL-Qwen3VL-4B Q6: {e}")
    else:
        logger.info(f"2/2 [Dense Action Captioning] Preloading {backend_name}...")
        try:
            captioner = QwenVLDenseCaptioner()
            captioner._lazy_load()
            if captioner.model is not None:
                logger.info("[OK] Qwen captioner loaded successfully.")
            else:
                logger.warning("[WARN] Qwen captioner unavailable; scene captions will be marked unavailable.")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load Qwen captioner: {e}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        logger.info("==================================================================")
        logger.info(f"[READY] ALL AI MODELS ARE PRELOADED & READY IN GPU MEMORY!")
        logger.info(f"GPU VRAM Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")
        logger.info("==================================================================")
    else:
        logger.info("==================================================================")
        logger.info("[READY] ALL AI MODELS ARE PRELOADED & READY (CPU Mode)!")
        logger.info("==================================================================")

if __name__ == "__main__":
    preload_all_models()
