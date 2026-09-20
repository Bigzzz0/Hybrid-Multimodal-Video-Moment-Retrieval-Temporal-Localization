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
    QWEN_VL_MODEL_ID: str = os.environ.get("QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
    SAM_MODEL_ID: str = os.environ.get("SAM_MODEL_ID", "facebook/sam3.1")
    ENABLE_SAM_GROUNDING: bool = os.environ.get("ENABLE_SAM_GROUNDING", "false").lower() in {"1", "true", "yes"}
    CAPTION_VERSION: str = os.environ.get("CAPTION_VERSION", "qwen3vl2b-v1")
    GROUNDING_VERSION: str = os.environ.get("GROUNDING_VERSION", "sam31-v1")

    # Optional local inference worker.  Keeping this opt-in preserves the
    # existing single-process development path while the worker environment
    # is installed separately for CUDA/SAM compatibility.
    INFERENCE_WORKER_ENABLED: bool = os.environ.get("INFERENCE_WORKER_ENABLED", "false").lower() in {"1", "true", "yes"}
    INFERENCE_WORKER_URL: str = os.environ.get("INFERENCE_WORKER_URL", "http://127.0.0.1:8011")
    INFERENCE_WORKER_TOKEN: Optional[str] = os.environ.get("INFERENCE_WORKER_TOKEN", None)
    INFERENCE_REQUEST_TIMEOUT_SEC: float = float(os.environ.get("INFERENCE_REQUEST_TIMEOUT_SEC", "35"))
    INFERENCE_WARMUP_QWEN: bool = os.environ.get("INFERENCE_WARMUP_QWEN", "true").lower() in {"1", "true", "yes"}
    MODEL_VRAM_BUDGET_MB: int = int(os.environ.get("MODEL_VRAM_BUDGET_MB", "11000"))
    SAM_COMPILE: bool = os.environ.get("SAM_COMPILE", "false").lower() in {"1", "true", "yes"}
    SAM_PRECOMPUTE_FPS: float = float(os.environ.get("SAM_PRECOMPUTE_FPS", "1"))
    SAM_SEARCH_FPS: float = float(os.environ.get("SAM_SEARCH_FPS", "4"))
    SAM_MAX_WINDOW_SEC: float = float(os.environ.get("SAM_MAX_WINDOW_SEC", "12"))
    SAM_MAX_FRAMES_PER_WINDOW: int = int(os.environ.get("SAM_MAX_FRAMES_PER_WINDOW", "48"))
    # SAM 3.1 has a materially longer cold start than the VLM on a 12 GB GPU.
    # Keep the normal Accurate budget for Qwen, but give object grounding its
    # own bounded budget and avoid spending it on several nearly identical
    # proposals during the first request.
    SAM_ACCURATE_MAX_SECONDS: float = float(os.environ.get("SAM_ACCURATE_MAX_SECONDS", "60"))
    SAM_SEARCH_TOP_K: int = int(os.environ.get("SAM_SEARCH_TOP_K", "1"))
    QWEN_MAX_FRAMES_PER_CANDIDATE: int = int(os.environ.get("QWEN_MAX_FRAMES_PER_CANDIDATE", "4"))
    QWEN_VERIFY_MAX_NEW_TOKENS: int = int(os.environ.get("QWEN_VERIFY_MAX_NEW_TOKENS", "96"))
    ACCURATE_MAX_SECONDS: float = float(os.environ.get("ACCURATE_MAX_SECONDS", "30"))
    GROUNDING_ARTIFACTS_DIR: Path = DATA_DIR / "grounding"
    
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
    VLM_RERANK_MAX_SECONDS: float = float(os.environ.get("VLM_RERANK_MAX_SECONDS", os.environ.get("ACCURATE_MAX_SECONDS", "30")))
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
    VLM_VERIFY_TOP_K: int = int(os.environ.get("VLM_VERIFY_TOP_K", "2"))
    # Accurate mode is capped by ACCURATE_MAX_SECONDS; up to four frames are
    # sent per candidate when the remaining budget permits it.
    VLM_VERIFY_FRAMES_PER_MOMENT: int = int(os.environ.get("VLM_VERIFY_FRAMES_PER_MOMENT", "4"))

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
settings.GROUNDING_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
