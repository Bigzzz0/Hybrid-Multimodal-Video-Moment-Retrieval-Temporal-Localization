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
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseModel as BaseSettings
    SettingsConfigDict = None

try:
    import torch
    _DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _DEFAULT_DEVICE = "cpu"

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Pure-Visual Video Moment Retrieval"
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
    # Index v2 is deliberately tied to the NaFlex checkpoint.  Keeping the
    # contract here lets the database layer reject stale vectors early.
    SIGLIP2_MODEL_ID: str = os.environ.get("SIGLIP2_MODEL_ID", "google/siglip2-base-patch16-naflex")
    VISUAL_INDEX_VERSION: str = os.environ.get("VISUAL_INDEX_VERSION", "v2")
    SIGLIP2_EMBEDDING_DIM: int = 768
    SIGLIP2_EMBEDDING_VERSION: str = "siglip2-naflex-v2"
    QWEN_VL_MODEL_ID: str = os.environ.get("QWEN_VL_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Ingestion & Sampling Parameters
    DEVICE: str = os.environ.get("DEVICE", _DEFAULT_DEVICE)
    KEYFRAME_SAMPLE_INTERVAL_SEC: float = 1.0
    SSIM_THRESHOLD: float = 0.65
    MAX_FRAMES_PER_SCENE: int = 60
    
    # Retrieval & Temporal Localization Parameters
    DEFAULT_RRF_K: int = 60
    DEFAULT_WEIGHT_VISUAL: float = 0.80
    DEFAULT_WEIGHT_CAPTION: float = 0.20
    TEMPORAL_GAUSSIAN_SIGMA: float = 1.5  # display smoothing only
    ENABLE_TTA_ENSEMBLE: bool = True
    DEFAULT_SEARCH_PROFILE: str = os.environ.get("SEARCH_PROFILE", "fast")
    VLM_RERANK_MAX_SECONDS: float = 15.0
    VLM_MIN_FREE_VRAM_MB: int = 1024
    CALIBRATION_TEMPERATURE: float = 1.0
    CALIBRATION_ARTIFACT_PATH: Path = DATA_DIR / "calibration_v2.json"
    FUSION_ARTIFACT_PATH: Path = DATA_DIR / "fusion_v2.json"
    NO_MATCH_THRESHOLD: float = 0.50

    # ==============================================================================
    # Pure-visual retrieval parameters
    # ==============================================================================

    # Strategy 5: Two-Stage VLM Temporal Verification & Endpoint Snapping
    ENABLE_VLM_STAGE2_VERIFY: bool = True      # Accurate profile reranks the top three visual proposals
    VLM_VERIFY_TOP_K: int = 3                  # Verify top 3 candidates
    # Four uniformly spaced frames keep three Qwen verification calls inside
    # the 15-second budget on the target 6.5-8GB GPUs. The retrieval contract
    # allows up to eight; using all eight made Accurate mode regularly time out.
    VLM_VERIFY_FRAMES_PER_MOMENT: int = 4

    # Gaussian Soft-NMS is applied after proposal generation.
    ENABLE_GAUSSIAN_SOFT_NMS: bool = True
    GAUSSIAN_SOFT_NMS_SIGMA: float = 0.40      # sigma_soft
    GAUSSIAN_SOFT_NMS_FLOOR: float = 0.05
    
    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(case_sensitive=True, env_file=".env", extra="ignore")
    else:
        class Config:
            case_sensitive = True
            env_file = ".env"
            extra = "ignore"

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
