"""Choose a configuration using validation only and lock before test."""
import argparse
import hashlib
import json
from pathlib import Path
from .metrics import gates

def select(reports):
    passing = []
    for report in reports:
        if report.get("split") != "validation":
            raise ValueError("Only validation reports can select thresholds/configuration")
        result = gates(report["screening"], report["final"], report["performance"])
        if result["passed"]:
            passing.append(report)
    if not passing:
        raise ValueError("No configuration passes the prototype targets; keep experimental")
    return min(passing, key=lambda row: (row["performance"]["processing_seconds"], row["performance"]["allocated_gib"]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()
    if args.lock.exists():
        raise ValueError("Validation is already locked. Use a new experiment directory")
    data = args.reports.read_bytes()
    winner = select(json.loads(data))
    lock = {"config": winner["config"], "threshold": winner["threshold"], "margin": winner["margin"],
            "validation_passed": True, "validation_sha256": hashlib.sha256(data).hexdigest(), "selection_split": "validation"}
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(json.dumps(lock, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
