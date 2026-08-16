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
    MINICPMV_MODEL_ID: str = os.environ.get("MINICPMV_MODEL_ID", "openbmb/MiniCPM-V-2_6")
    WHISPER_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL_SIZE", "large-v3-turbo")
    
    # Ingestion & Sampling Parameters
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu"
    KEYFRAME_SAMPLE_INTERVAL_SEC: float = 1.0
    SSIM_THRESHOLD: float = 0.65
    MAX_FRAMES_PER_SCENE: int = 10
    
    # Retrieval & Temporal Localization Parameters
    DEFAULT_RRF_K: int = 60
    DEFAULT_WEIGHT_VISUAL: float = 0.45
    DEFAULT_WEIGHT_CAPTION: float = 0.35
    DEFAULT_WEIGHT_AUDIO: float = 0.20
    TEMPORAL_GAUSSIAN_SIGMA: float = 1.5
    DYNAMIC_THRESHOLD_FACTOR: float = 0.8
    
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
