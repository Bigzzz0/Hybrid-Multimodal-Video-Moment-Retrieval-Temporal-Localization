"""Recompute saved view metrics after order-independent interval union repair.
Never edits raw replies/protocols or calls a model.
"""
import hashlib
import json
from pathlib import Path

from .validate_dense_verification import metrics


def main():
    root = Path("scratch/event-retrieval")
    target = root/"spatial-temporal-analysis.json"
    if target.exists():
        raise ValueError("Preserve analysis")
    source = json.loads((root/"continuous-ut/results.json").read_text(encoding="utf-8"))
    specs = [
        ("full_development", "temporal-dev", False, (1, 2), "temporal-v1"),
        ("hog_development", "person-region-dev", False, (1, 2), None),
        ("ssdlite_development", "person-region-ssdlite-dev", False, (1, 2), None),
        ("motion_development", "motion-zoom-dev", True, (1, 2), None),
        ("motion_validation", "motion-zoom-validation", True, (13, 14, 15, 16), None),
    ]
    result = dict(scope="Offline recomputation; raw artifacts retained. Time extent fields describe input views, not verified normal recording hours.", runs={})
    for name, directory, dense, sequences, mode in specs:
        path = root/directory/"results.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if not artifact["complete"]:
            raise ValueError("Incomplete run: " + name)
        run = artifact["runs"][mode] if mode else artifact
        result["runs"][name] = dict(metrics=metrics(source, run["attempts"], sequences, dense),
            artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if name == "motion_validation":
            result["runs"]["full_validation"] = dict(metrics=metrics(source, run["baseline"]["attempts"], sequences, False))
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, row in result["runs"].items():
        m = row["metrics"]
        print(name, m["covered_events"], "/", m["events"], "non-target", m["non_target_windows"], "union", m["selected_union_seconds"])


if __name__ == "__main__":
    main()
