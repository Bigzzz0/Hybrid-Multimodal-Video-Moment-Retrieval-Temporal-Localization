"""Compare two benchmark reports with paired bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

from evaluation.compute_metrics import temporal_f1


def _query_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("video_id", "")), str(row.get("event_id", "")), str(row.get("query", "")))


def paired_bootstrap_ci(
    baseline: list[float], candidate: list[float], samples: int = 10_000, seed: int = 7
) -> dict[str, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap requires non-empty equal-length query vectors")
    deltas = [float(new) - float(old) for old, new in zip(baseline, candidate)]
    rng = random.Random(seed)
    means = []
    for _ in range(max(100, samples)):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(draw) / len(draw))
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return {
        "delta": sum(deltas) / len(deltas),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "query_count": len(deltas),
    }


def compare_reports(baseline_path: Path, candidate_path: Path, samples: int = 10_000) -> dict[str, Any]:
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_report = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_rows = {_query_key(row): row for row in baseline_report.get("queries", [])}
    candidate_rows = {_query_key(row): row for row in candidate_report.get("queries", [])}
    keys = sorted(set(baseline_rows) & set(candidate_rows))
    if not keys:
        raise ValueError("reports have no paired query keys")

    def vector(row_map: dict[tuple[str, str, str], dict[str, Any]], fn: Callable[[dict[str, Any]], float]) -> list[float]:
        return [fn(row_map[key]) for key in keys]

    baseline_iou = vector(baseline_rows, lambda row: float(row.get("top1_iou", 0.0)))
    candidate_iou = vector(candidate_rows, lambda row: float(row.get("top1_iou", 0.0)))
    baseline_hit = [float(value >= 0.5) for value in baseline_iou]
    candidate_hit = [float(value >= 0.5) for value in candidate_iou]
    def row_f1(row: dict[str, Any]) -> float:
        predictions = [tuple(interval) for interval in row.get("all_predictions", [])]
        ground_truth = row.get("gt_interval", [])
        return float(temporal_f1(predictions, ground_truth, iou_threshold=0.5))

    baseline_f1 = vector(baseline_rows, row_f1)
    candidate_f1 = vector(candidate_rows, row_f1)
    report = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "profile": candidate_report.get("profile", "fast"),
        "paired_queries": len(keys),
        "primary_R@1_IoU_0.5": paired_bootstrap_ci(baseline_hit, candidate_hit, samples),
        "mIoU": paired_bootstrap_ci(baseline_iou, candidate_iou, samples),
        "tF1_IoU_0.5": paired_bootstrap_ci(baseline_f1, candidate_f1, samples),
        "metric_snapshot": {
            "baseline": baseline_report.get("metrics", {}),
            "candidate": candidate_report.get("metrics", {}),
        },
    }
    return report


def acceptance_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the plan's quantitative release gate to a paired comparison."""
    baseline = report["metric_snapshot"]["baseline"]
    candidate = report["metric_snapshot"]["candidate"]
    failures: list[str] = []

    primary = report["primary_R@1_IoU_0.5"]
    primary_delta_pp = primary["delta"] * 100.0
    if primary_delta_pp < 5.0:
        failures.append(f"R@1@IoU=.5 delta {primary_delta_pp:.3f} < 5 percentage points")
    if primary["ci95_lower"] <= 0.0:
        failures.append("primary 95% bootstrap CI still includes zero")

    miou_delta = float(candidate.get("mIoU", 0.0)) - float(baseline.get("mIoU", 0.0))
    if miou_delta < 0.03:
        failures.append(f"mIoU delta {miou_delta:.4f} < 0.03")
    tf1_delta = report["tF1_IoU_0.5"]["delta"] * 100.0
    if tf1_delta < 5.0:
        failures.append(f"tF1@IoU=.5 delta {tf1_delta:.3f} < 5 percentage points")

    # All tracked metrics other than the three primary improvements may not
    # regress by more than one percentage point.
    percentage_metrics = [
        "R@1@IoU=0.3", "R@1@IoU=0.7", "R@5@IoU=0.3", "R@5@IoU=0.5", "R@5@IoU=0.7",
        "tF1@IoU=0.3", "tF1@IoU=0.7", "count_accuracy",
    ]
    for key in percentage_metrics:
        if key in baseline and key in candidate and float(candidate[key]) - float(baseline[key]) < -1.0:
            failures.append(f"{key} regressed by more than one percentage point")

    profile = str(candidate.get("profile", report.get("profile", "fast"))).lower()
    p95 = float(candidate.get("p95_query_latency_ms", float("inf")))
    latency_limit = 1000.0 if profile == "fast" else 30000.0
    if p95 > latency_limit:
        failures.append(f"{profile} p95 latency {p95:.1f}ms > {latency_limit:.0f}ms")
    if "peak_vram_mb" not in candidate:
        failures.append("peak VRAM was not recorded")
    else:
        peak_vram = float(candidate["peak_vram_mb"])
        if peak_vram > 11000.0:
            failures.append(f"peak VRAM {peak_vram:.1f}MB > 11000MB")

    return {"passed": not failures, "failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison for benchmark reports")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--acceptance", action="store_true", help="Fail unless quantitative acceptance gate passes")
    parser.add_argument(
        "--ablation", action="append", default=[], metavar="NAME=PATH",
        help="Optional ablation report(s) compared to --baseline; repeat for NaFlex/RRF/proposal/calibration/Qwen runs",
    )
    args = parser.parse_args()
    result = compare_reports(args.baseline, args.candidate, args.samples)
    if args.ablation:
        result["ablations"] = {}
        for spec in args.ablation:
            if "=" not in spec:
                raise ValueError("--ablation must use NAME=PATH")
            name, report_path = spec.split("=", 1)
            result["ablations"][name] = compare_reports(args.baseline, Path(report_path), args.samples)
    if args.acceptance:
        result["acceptance_gate"] = acceptance_gate(result)
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    if args.acceptance and not result["acceptance_gate"]["passed"]:
        raise SystemExit(2)
