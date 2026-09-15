"""Grid-search visual/caption RRF weights on a development JSONL file."""

import argparse
import json
from pathlib import Path


def tune(rows: list[dict], index_version: str = "v2", model_id: str = "") -> dict:
    if not rows:
        raise ValueError("development set is empty")
    best = None
    for visual_weight_i in range(0, 11):
        visual_weight = visual_weight_i / 10.0
        caption_weight = 1.0 - visual_weight
        total = 0.0
        for row in rows:
            visual = float(row.get("visual", 0.0))
            caption = float(row.get("caption", 0.0))
            target = float(row.get("target", row.get("iou", 0.0)))
            total += (visual_weight * visual + caption_weight * caption) * target
        if best is None or total > best["objective"]:
            best = {"visual": visual_weight, "caption": caption_weight, "objective": total}
    return {
        "index_version": index_version,
        "model_id": model_id,
        "visual_weight": best["visual"],
        "caption_weight": best["caption"],
        "objective": best["objective"],
        "row_count": len(rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-version", default="v2")
    parser.add_argument("--model-id", default="")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    artifact = tune(rows, args.index_version, args.model_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
