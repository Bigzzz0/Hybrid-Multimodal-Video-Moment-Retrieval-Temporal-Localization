import os
from pathlib import Path
from typing import Optional

BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
env_file = BASE_DIR / ".env"

# Safe .env loader
try:
    from dotenv import load_dotenv
    load_dotenv(env_file)
except ImportError:
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel as BaseSettings

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Hybrid Multimodal Video Moment Retrieval"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    
    # Base Directories
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_VIDEOS_DIR: Path = DATA_DIR / "raw_videos"
    KEYFRAMES_DIR: Path = DATA_DIR / "keyframes"
    LANCEDB_DIR: Path = DATA_DIR / "lancedb"
    
    # Authentication & API Keys
    HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN", None)
    
    # Model Configurations
    SIGLIP2_MODEL_ID: str = os.environ.get("SIGLIP2_MODEL_ID", "google/siglip2-base-patch16-256")
    QWEN_VL_MODEL_ID: str = os.environ.get("QWEN_VL_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    MINICPMV_MODEL_ID: str = os.environ.get("QWEN_VL_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")  # Backward-compat
    WHISPER_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL_SIZE", "disabled")
    ENABLE_AUDIO_ASR: bool = False
    
    # Ingestion & Sampling Parameters
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu"
    KEYFRAME_SAMPLE_INTERVAL_SEC: float = 1.0
    SSIM_THRESHOLD: float = 0.65
    MAX_FRAMES_PER_SCENE: int = 60
    
    # Retrieval & Temporal Localization Parameters (Visual-Centric SOTA)
    DEFAULT_RRF_K: int = 60
    DEFAULT_WEIGHT_VISUAL: float = 0.60
    DEFAULT_WEIGHT_CAPTION: float = 0.40
    DEFAULT_WEIGHT_AUDIO: float = 0.00
    TEMPORAL_GAUSSIAN_SIGMA: float = 1.5
    DYNAMIC_THRESHOLD_FACTOR: float = 0.8
    ENABLE_MOTION_DELTA: bool = True
    MOTION_DELTA_WEIGHT: float = 0.25
    ENABLE_BG_CONTRASTIVE: bool = True
    BG_CONTRASTIVE_LAMBDA: float = 0.20
    ENABLE_TTA_ENSEMBLE: bool = True
    ENABLE_VLM_RERANK: bool = False

    # ==============================================================================
    # SOTA 2025-2026 PRECISION RETRIEVAL PARAMETERS (7 ENHANCED MODULES)
    # ==============================================================================

    # Strategy 1: Pixel Motion Energy Gating
    ENABLE_PIXEL_MOTION_GATE: bool = True
    PIXEL_MOTION_WEIGHT: float = 0.12          # gamma_motion (calibrated to preserve subtle actions)
    PIXEL_MOTION_TANH_BETA: float = 2.5        # beta slope

    # Strategy 2: Dual-Scale Temporal Context Hierarchy (UniTime NeurIPS 2025)
    ENABLE_DUAL_SCALE_CONTEXT: bool = True
    DUAL_SCALE_LOCAL_WINDOW: int = 3           # W_local (frames)
    DUAL_SCALE_GLOBAL_WINDOW: int = 7          # W_global (frames)
    DUAL_SCALE_LOCAL_WEIGHT: float = 0.65      # alpha_local
    DUAL_SCALE_GLOBAL_WEIGHT: float = 0.35     # alpha_global

    # Strategy 3: Self-Similarity Matrix (SSM) Directional Gradient Cut
    ENABLE_SSM_BOUNDARY_SNAP: bool = True
    SSM_SNAP_WINDOW_SEC: float = 1.5           # delta search radius (seconds)

    # Strategy 4: Hard Static Negative Anchor Subtraction
    ENABLE_STATIC_NEGATIVE: bool = True
    STATIC_NEGATIVE_LAMBDA: float = 0.20       # lambda_null penalty factor
    STATIC_NULL_PROMPT: str = "an empty static background scene with zero human activity, no movement, no action, motionless"

    # Strategy 5: Two-Stage VLM Temporal Verification & Endpoint Snapping
    ENABLE_VLM_STAGE2_VERIFY: bool = False     # Default False for instant ultra-fast benchmark; set True for VLM deep mode
    VLM_VERIFY_TOP_K: int = 3                  # Verify top 3 candidates
    VLM_VERIFY_FRAMES_PER_MOMENT: int = 4      # Sample 4 equidistant frames

    # Strategy 6: Concept Disentanglement Scoring (Visual CoT)
    ENABLE_CONCEPT_DISENTANGLEMENT: bool = True
    DISENTANGLE_WEIGHT_SUB: float = 0.25       # Subject weight
    DISENTANGLE_WEIGHT_VERB: float = 0.50      # Action Verb weight
    DISENTANGLE_WEIGHT_CTX: float = 0.25       # Context weight

    # Strategy 7: 1D Continuous Gaussian Soft-NMS
    ENABLE_GAUSSIAN_SOFT_NMS: bool = True
    GAUSSIAN_SOFT_NMS_SIGMA: float = 0.40      # sigma_soft
    GAUSSIAN_SOFT_NMS_FLOOR: float = 0.20      # Minimum score retention threshold
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()

# Export HF token to environment for HuggingFace Hub and Transformers
if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
settings.KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)
settings.LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
