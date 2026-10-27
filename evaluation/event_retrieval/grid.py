"""Fair fixed-frame vision-budget comparison, then MRL after validation selection."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from .benchmark import run, IndexConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("vision", "dimensions"), required=True)
    parser.add_argument("--validation-lock", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.phase == "vision":
        # 12 x 1120 exceeds context. All FOUR arms use the SAME six frames,
        # including 140/280/560; no hidden unequal frame counts.
        configs = [(str(tokens), IndexConfig(video_fps=1, max_frames=6, video_tokens=tokens)) for tokens in (140, 280, 560, 1120)]
    else:
        if not args.validation_lock:
            raise ValueError("Select an accuracy-qualified validation configuration before comparing dimensions")
        lock = json.loads(args.validation_lock.read_text())
        if not lock.get("validation_passed") or lock.get("selection_split") != "validation":
            raise ValueError("Invalid validation lock")
        base = IndexConfig(**{**lock["config"], "modalities": tuple(lock["config"]["modalities"])})
        configs = [(str(dim), replace(base, dimension=dim)) for dim in (768, 512, 256)]
    for label, config in configs:
        run(manifest, args.output / label, names=["gemma2_video", "gemma2_video_audio"], base=config)

if __name__ == "__main__":
    main()
