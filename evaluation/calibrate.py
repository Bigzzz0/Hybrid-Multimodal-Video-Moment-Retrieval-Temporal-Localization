"""Fit Platt calibration and a no-match threshold from development results."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, List, Tuple


def _fit_platt(xs: List[float], ys: List[int]) -> Tuple[float, float]:
    slope, intercept = 1.0, 0.0
    for _ in range(100):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            z = max(-40.0, min(40.0, slope * x + intercept))
            p = 1.0 / (1.0 + math.exp(-z))
            w = max(1e-6, p * (1.0 - p))
            err = p - y
            g0 += err * x
            g1 += err
            h00 += w * x * x
            h01 += w * x
            h11 += w
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-10:
            break
        ds = (h11 * g0 - h01 * g1) / det
        di = (-h01 * g0 + h00 * g1) / det
        slope -= ds
        intercept -= di
        if abs(ds) + abs(di) < 1e-7:
            break
    return slope, intercept


def fit_calibration(rows: List[dict], index_version: str, model_id: str) -> dict:
    if len(rows) < 100:
        raise ValueError("calibration requires at least 100 development queries")
    query_ids = {
        str(row.get("query_id", row.get("query", row.get("id", index))))
        for index, row in enumerate(rows)
    }
    if len(query_ids) < 100:
        raise ValueError("calibration requires at least 100 unique development queries")
    no_match_rows = [
        row for row in rows
        if bool(row.get("is_no_match", False))
        or row.get("ground_truths") == []
        or row.get("gt_interval") == []
    ]
    if not no_match_rows:
        raise ValueError("calibration requires explicit negative/no-match queries")
    xs = [float(row.get("rank_score", row.get("score", 0.0))) for row in rows]
    ys = []
    for row in rows:
        explicit = row.get("is_positive")
        if explicit is None:
            ys.append(int(float(row.get("iou", 0.0)) >= 0.5))
        elif isinstance(explicit, str):
            ys.append(int(explicit.strip().lower() in {"1", "true", "yes", "positive"}))
        else:
            ys.append(int(bool(explicit)))
    if not any(ys) or all(ys):
        raise ValueError("calibration requires both positive and negative/no-match queries")
    slope, intercept = _fit_platt(xs, ys)
    candidates = sorted({1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, slope * x + intercept)))) for x in xs})
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        tp = sum(y == 1 and p >= threshold for x, y in zip(xs, ys) for p in [1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, slope * x + intercept))))])
        fp = sum(y == 0 and p >= threshold for x, y in zip(xs, ys) for p in [1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, slope * x + intercept))))])
        fn = sum(y == 1 and p < threshold for x, y in zip(xs, ys) for p in [1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, slope * x + intercept))))])
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return {"index_version": index_version, "model_id": model_id, "slope": slope,
            "intercept": intercept, "no_match_threshold": best_threshold,
            "query_count": len(rows), "positive_count": sum(ys)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSONL dev predictions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-version", default="v2")
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    artifact = fit_calibration(rows, args.index_version, args.model_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
