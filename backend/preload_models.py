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
    logger.info("🚀 PRELOADING PURE-VISUAL RETRIEVAL MODELS")
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
        logger.info(f"✅ SigLIP 2 loaded successfully (Sample vector length: {len(test_emb)}).")
    except Exception as e:
        logger.error(f"❌ Failed to load SigLIP 2: {e}")

    # 2. Qwen2.5-VL-7B-Instruct (4-bit Quantized)
    logger.info("2/2 [Dense Action Captioning] Preloading Qwen2.5-VL-7B (4-bit quantized)...")
    try:
        captioner = QwenVLDenseCaptioner()
        captioner._lazy_load()
        if captioner.model is not None:
            logger.info("✅ Qwen2.5-VL-7B 4-bit loaded successfully.")
        else:
            logger.warning("⚠️ Qwen2.5-VL-7B unavailable; scene captions will be marked unavailable.")
    except Exception as e:
        logger.error(f"❌ Failed to load Qwen2.5-VL-7B: {e}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        logger.info("==================================================================")
        logger.info(f"🎉 ALL AI MODELS ARE PRELOADED & READY IN GPU MEMORY!")
        logger.info(f"GPU VRAM Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")
        logger.info("==================================================================")

if __name__ == "__main__":
    preload_all_models()
