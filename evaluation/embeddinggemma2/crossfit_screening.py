"""Lock conservative development cross-fit caps before reading test metrics.
Run --lock while the main continuous experiment builds validation; --evaluate
only after it finishes. No test labels participate in the selection rule.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from app.gemma2.config import IndexConfig
from app.gemma2.ranking import EVENTS
from app.gemma2.service import WorkerClient
from .continuous_screening import METHODS, annotations, calibrate, measure, reference_bank, role, score


def load_features(root, allowed):
    features = {}
    for path in (root/"features").glob("*.json"):
        provenance = json.loads(path.read_text(encoding="utf-8"))
        if provenance["seq"] in allowed:
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
                features[provenance["seq"]] = {k: data[k].copy() for k in data.files}
    if set(features) != set(allowed):
        raise ValueError("Feature set incomplete")
    return features


def crossfit(features, labels, prompts):
    if len(features) < 2 or any(role(seq) != "development" for seq in features):
        raise ValueError("Crossfit needs multiple development recordings only")
    records = []
    for seq, feature in features.items():
        bank, _ = reference_bank({key: value for key, value in features.items() if key != seq}, labels)
        records.append((seq, feature, score(feature, bank, prompts)))
    return records


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--lock", action="store_true"); parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output", default="scratch/event-retrieval/continuous-ut")
    args = parser.parse_args()
    if args.lock == args.evaluate:
        parser.error("Choose exactly one of --lock or --evaluate")
    root = Path(args.output)
    sys.path.insert(0, str(Path("scratch/evaluation-tools").resolve()))
    labels = annotations("scratch/event-retrieval/datasets/ut-labels.xls")
    config = IndexConfig(modalities=("image", "video"))
    spec = EVENTS["pushing_punching"]
    query = WorkerClient().embed("query", config, texts=spec["positive"]+spec["negative"])
    prompts = query[:len(spec["positive"])], query[len(spec["positive"]):]
    development = load_features(root, range(1, 13))
    bank, _ = reference_bank(development, labels)
    lock_path = root/"crossfit-lock.json"
    if args.lock:
        if lock_path.exists():
            raise ValueError("Crossfit lock already exists")
        # First freeze caps from development-only leave-one-source-out records.
        records = crossfit(development, labels, prompts)
        caps = {method: calibrate(records, method, labels, split="development")[0] for method in METHODS}
        rule = dict(caps=caps, methods=list(METHODS), reference_ids=list(range(1, 13)),
            rule="min(development crossfit cap, validation-only threshold); require >=95% coverage in both; max minimum recall then fewest combined candidates; simpler method ties",
            created_ns=time.time_ns())
        (root/"crossfit-caps.json").write_text(json.dumps(rule, indent=2), encoding="utf-8")
        print("CAPS LOCKED", caps, flush=True)
        deadline = time.monotonic()+600
        while not (root/"lock.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Main validation has not finished; frozen caps remain saved")
            time.sleep(1)
        original = json.loads((root/"lock.json").read_text(encoding="utf-8"))
        validation = load_features(root, range(13, 17))
        val_records = [(seq, feature, score(feature, bank, prompts)) for seq, feature in validation.items()]
        arms = {}
        for method in METHODS:
            threshold = min(caps[method], original["methods"][method]["threshold"])
            arms[method] = dict(threshold=threshold, crossfit=measure(records, method, threshold, labels),
                                validation=measure(val_records, method, threshold, labels))
        selected = max(METHODS, key=lambda m: (min(arms[m][s]["coverage_recall"] for s in ("crossfit", "validation")) >= .95,
            min(arms[m][s]["coverage_recall"] for s in ("crossfit", "validation")),
            -sum(arms[m][s]["selected_windows"] for s in ("crossfit", "validation")), -METHODS.index(m)))
        rule.update(selected=selected, arms=arms, locked_ns=time.time_ns(), annotation_sha256=original["annotation_sha256"], config=original["config"])
        lock_path.write_text(json.dumps(rule, indent=2), encoding="utf-8")
        print("CROSSFIT SELECTED", selected, flush=True)
    else:
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
        # Selection is loaded, never recomputed using these held-source labels.
        test = load_features(root, range(17, 21))
        records = [(seq, feature, score(feature, bank, prompts)) for seq, feature in test.items()]
        results = {method: measure(records, method, locked["arms"][method]["threshold"], labels) for method in METHODS}
        output = dict(lock=locked, test=results)
        (root/"crossfit-results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        print("CROSSFIT TEST", locked["selected"], json.dumps(results[locked["selected"]]), flush=True)


if __name__ == "__main__":
    main()
