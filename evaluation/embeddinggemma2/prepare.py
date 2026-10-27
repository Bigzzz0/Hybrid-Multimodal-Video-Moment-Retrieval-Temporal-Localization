"""Download/check prerequisites only. Never construct models or run inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_models(config: dict) -> None:
    from huggingface_hub import snapshot_download

    for role in ("baseline", "candidate"):
        spec = config[role]
        print(f"Downloading {role}: {spec['model_id']} @ {spec['revision']}", flush=True)
        snapshot_download(
            repo_id=spec["model_id"], revision=spec["revision"],
            local_dir=ROOT / spec["local_dir"],
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "README.md"],
            max_workers=4,
        )
        # A complete snapshot download must finish before marking the revision.
        (ROOT / spec["local_dir"] / "experiment_revision.json").write_text(
            json.dumps({"model_id": spec["model_id"], "revision": spec["revision"]}, indent=2),
            encoding="utf-8",
        )


def check_prerequisites(config: dict) -> dict:
    """Read local files, inspect imports/GPU/media metadata; no model forward."""
    import cv2
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_MAPPING
    from safetensors import safe_open
    from sentence_transformers import SentenceTransformer  # noqa: F401

    errors: list[str] = []
    warnings: list[str] = []
    versions = {}
    for package in ("torch", "transformers", "sentence-transformers", "tokenizers", "numpy", "scipy", "pillow"):
        versions[package] = importlib.metadata.version(package)
    if "embedding_gemma2" not in CONFIG_MAPPING:
        errors.append("Installed transformers does not support embedding_gemma2")
    gpu = {"cuda_available": torch.cuda.is_available()}
    if gpu["cuda_available"]:
        free, total = torch.cuda.mem_get_info()
        gpu.update(name=torch.cuda.get_device_name(0), bf16_supported=torch.cuda.is_bf16_supported(),
                   free_vram_mb=round(free / 1024**2), total_vram_mb=round(total / 1024**2))
    elif config["device"].startswith("cuda"):
        errors.append("Configured CUDA device is unavailable")
    model_files = {}
    for role in ("baseline", "candidate"):
        spec = config[role]
        folder = ROOT / spec["local_dir"]
        required = ["config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json", "experiment_revision.json"]
        if role == "candidate":
            required += ["modules.json", "config_sentence_transformers.json", "1_Pooling/config.json", "2_Normalize/config.json"]
        missing = [name for name in required if not (folder / name).is_file() or (folder / name).stat().st_size == 0]
        if missing:
            errors.append(f"{role} missing files: {', '.join(missing)}")
        marker = folder / "experiment_revision.json"
        if marker.exists():
            downloaded = json.loads(marker.read_text(encoding="utf-8"))
            if any(downloaded.get(key) != spec[key] for key in ("model_id", "revision")):
                errors.append(f"{role} local model revision does not match config")
        if not missing:
            try:
                # Resolve the implementation and tokenizer/processor only.
                # Never instantiate MODEL_MAPPING[...], SentenceTransformer or AutoModel.
                local_config = AutoConfig.from_pretrained(str(folder), local_files_only=True)
                implementation = MODEL_MAPPING[type(local_config)]
                processor = AutoProcessor.from_pretrained(str(folder), local_files_only=True)
                with safe_open(str(folder / "model.safetensors"), framework="pt", device="cpu") as weights:
                    tensor_count = len(weights.keys())
                model_files[role] = {"implementation": implementation.__name__,
                                     "processor": type(processor).__name__, "weight_tensor_count": tensor_count}
                del processor, local_config
            except Exception as exc:
                errors.append(f"{role} local loading prerequisites: {type(exc).__name__}: {exc}")
        model_files[role] = {"path": str(folder), "missing": missing,
                             "weights_bytes": (folder / "model.safetensors").stat().st_size if (folder / "model.safetensors").exists() else 0,
                             **model_files.get(role, {})}
    dataset_path = ROOT / config["dataset"]
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    videos = []
    query_count = 0
    for video in dataset["videos"]:
        path = ROOT / config["media_dir"] / video["filename"]
        query_count += sum(len(event["queries"]) for event in video["events"])
        if not path.is_file():
            errors.append(f"Missing benchmark media: {path}")
            continue
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps <= 0 or frame_count <= 0:
            errors.append(f"Cannot read media metadata: {path.name}")
            continue
        duration = frame_count / fps
        for event in video["events"]:
            intervals = event.get("ground_truths", [[event.get("t_start", 0), event.get("t_end", 0)]])
            for start, end in intervals:
                if not (0 <= start < end <= duration + 0.1):
                    errors.append(f"Invalid ground truth {video['video_id']}/{event['event_id']}: {start}, {end}")
        videos.append({"video_id": video["video_id"], "path": str(path), "sha256": file_hash(path),
                       "fps": fps, "duration_sec": duration, "frame_count": frame_count})
    if len(videos) < 10 or query_count < 100:
        warnings.append("Small smoke/regression set; results cannot establish general superiority or pass the production acceptance gate")
    return {"ready": not errors, "inference_run": False, "benchmark_run": False,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
            "versions": versions, "gpu": gpu, "models": model_files,
            "dataset_sha256": file_hash(dataset_path), "video_count": len(videos), "query_count": query_count,
            "videos": videos, "errors": errors, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--download", action="store_true", help="Download pinned weights; no inference")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.download:
        download_models(config)
    report = check_prerequisites(config)
    output = ROOT / "scratch/embeddinggemma2/readiness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
