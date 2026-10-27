"""Frozen-encoder reference experiments. Clip classification, not localization.

Development supplies references, validation selects thresholds/method; UT test
is opened only after writing the lock. UR preview test remains sealed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import requests

from app.gemma2.config import IndexConfig
from app.gemma2.media import extract, metadata, windows
from app.gemma2.ranking import EVENTS
from app.gemma2.service import WorkerClient
from app.gemma2.store import file_hash

VERSION = "references-v1"
METHODS = ("prompt_margin", "reference_video", "reference_sequence", "reference_fusion", "normal_novelty")


def unit(value):
    value = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(value)
    if not np.isfinite(value).all() or norm < 1e-8:
        raise ValueError("Invalid feature")
    return value / norm


def sequence_feature(frames):
    # Three ordered bins retain coarse chronology; no learned parameters.
    frames = np.asarray(frames)
    if len(frames) == 0:
        raise ValueError("No frames")
    bins = np.array_split(np.arange(len(frames)), 3) if len(frames) >= 3 else [np.array([round(x)]) for x in np.linspace(0, len(frames)-1, 3)]
    return unit(np.concatenate([unit(frames[slots].mean(axis=0)) for slots in bins]))


def audit(rows):
    groups, hashes, ids = {}, {}, set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate id")
        ids.add(row["id"])
        for table, key in ((groups, row["recording_id"]), (hashes, row["sha256"])):
            if key in table and table[key] != row["split"]:
                raise ValueError("Split leakage")
            table[key] = row["split"]
        if row["split"] not in ("development", "validation", "test"):
            raise ValueError("Unknown split")


def metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=bool)
    pred = np.asarray(scores) >= threshold
    tp = int(np.sum(labels & pred)); fp = int(np.sum(~labels & pred))
    fn = int(np.sum(labels & ~pred)); tn = int(np.sum(~labels & ~pred))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=tp/(tp+fn) if tp+fn else 0,
                precision=tp/(tp+fp) if tp+fp else 0, selected=int(pred.sum()), total=len(labels))


def calibrate(rows, scores):
    if not rows or any(r["split"] != "validation" for r in rows):
        raise ValueError("Calibration requires validation only")
    labels = [r["label"] for r in rows]
    if not any(labels) or all(labels):
        raise ValueError("Both classes required")
    thresholds = sorted(set(map(float, scores))) + [float(np.nextafter(max(scores), np.inf))]
    options = [(threshold, metrics(labels, scores, threshold)) for threshold in thresholds]
    # First meet screening recall, then minimize false positives. Stable tie.
    return max(options, key=lambda item: (item[1]["recall"] >= .95, item[1]["recall"], -item[1]["fp"], item[0]))


def neighbor_margin(feature, positive, normal):
    def top_mean(bank):
        values = bank @ feature
        return float(np.sort(values)[-min(3, len(values)):].mean())
    return top_mean(positive)-top_mean(normal), 1-top_mean(normal)


def encode(row, root, client, config, telemetry):
    if file_hash(Path(row["path"])) != row["sha256"]:
        raise ValueError("Source fingerprint changed")
    duration = metadata(row["path"])["duration"]
    key = hashlib.sha256(json.dumps([VERSION, row["sha256"], duration, config.json()], sort_keys=True).encode()).hexdigest()
    cache = root / "features" / (key + ".npz")
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}
    before = time.perf_counter(); vids, imgs, stamps = [], [], []
    for wi, (start, end) in enumerate(windows(duration)):
        media = extract(row["path"], start, end, root / "frames" / key / str(wi), config, False)
        vids.append(client.embed("video", config, frame_paths=media["frame_paths"])[0])
        telemetry.append(client.last_telemetry)
        paths = [media["frame_paths"][s] for s in media["image_slots"]]
        images = client.embed("image", config, frame_paths=paths)
        telemetry.append(client.last_telemetry)
        imgs.extend(images); stamps.extend(media["timestamps"][s] for s in media["image_slots"])
    # Overlapping windows must not overweight duplicated instants.
    unique = {round(stamp, 5): vec for stamp, vec in zip(stamps, imgs)}
    feature = dict(video=unit(np.mean(vids, axis=0)), sequence=sequence_feature([unique[s] for s in sorted(unique)]),
                   images=np.array(imgs), videos=np.array(vids), encode_seconds=np.array(time.perf_counter()-before),
                   actual_duration=np.array(duration), manifest_duration=np.array(row["duration"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.npz")
    np.savez(temporary, **feature); temporary.replace(cache)
    return feature


def score(row, features, bank, prompts):
    if any(r["split"] != "development" for r in bank):
        raise ValueError("Reference bank requires development only")
    feat = features[row["id"]]
    pos, neg = prompts
    all_media = np.concatenate([feat["images"], feat["videos"]])
    p = (all_media @ pos.T).max(axis=1)
    n = (all_media @ neg.T).max(axis=1)
    margin = np.where(p >= .35, p-n, -2).max()
    video, novelty = neighbor_margin(feat["video"],
        np.stack([features[r["id"]]["video"] for r in bank if r["label"]]),
        np.stack([features[r["id"]]["video"] for r in bank if not r["label"]]))
    seq, _ = neighbor_margin(feat["sequence"],
        np.stack([features[r["id"]]["sequence"] for r in bank if r["label"]]),
        np.stack([features[r["id"]]["sequence"] for r in bank if not r["label"]]))
    return dict(prompt_margin=float(margin), reference_video=video, reference_sequence=seq,
                reference_fusion=(video+seq)/2, normal_novelty=novelty)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scratch/event-retrieval/training-free")
    args = parser.parse_args(); root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    if (root / "lock.json").exists():
        raise RuntimeError("Use a new output directory for another experiment; existing test lock preserved")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise RuntimeError("Screening jobs active")
    manifest = json.loads(Path("scratch/event-retrieval/datasets/manifest.combined.json").read_text(encoding="utf-8"))
    rows = []
    for raw in manifest["videos"]:
        if raw["id"].startswith("ut-") and raw.get("source_class_id") in (0, 1, 3, 4, 5):
            rows.append(dict(raw, kind="pushing_punching", dataset="UT", label=raw["source_class_id"] in (4, 5)))
        elif raw["id"].startswith("ur-") and raw["split"] != "test":
            rows.append(dict(raw, kind="falling", dataset="UR-preview", label=bool(raw["events"])))
    audit(rows)
    client = WorkerClient(); config = IndexConfig(modalities=("image", "video"))
    texts = [t for kind in ("pushing_punching", "falling") for role in ("positive", "negative") for t in EVENTS[kind][role]]
    embeddings = client.embed("query", config, texts=texts); lookup = dict(zip(texts, embeddings))
    prompts = {kind: tuple(np.stack([lookup[t] for t in EVENTS[kind][role]]) for role in ("positive", "negative")) for kind in ("pushing_punching", "falling")}
    features, telemetry, results, lock = {}, [], {}, {}
    initial = [r for r in rows if r["split"] != "test"]
    for i, row in enumerate(initial, 1):
        features[row["id"]] = encode(row, root, client, config, telemetry)
        print(f"ENCODE {i}/{len(initial)} {row['id']}", flush=True)
    for dataset in ("UT", "UR-preview"):
        bank = [r for r in rows if r["dataset"] == dataset and r["split"] == "development"]
        val = [r for r in rows if r["dataset"] == dataset and r["split"] == "validation"]
        scores = [score(r, features, bank, prompts[r["kind"]]) for r in val]
        arms = {}
        for method in METHODS:
            threshold, result = calibrate(val, [s[method] for s in scores])
            arms[method] = dict(threshold=threshold, validation=result)
        # Prefer simpler method on identical validation confusion counts.
        selected = max(METHODS, key=lambda m: (arms[m]["validation"]["recall"] >= .95, arms[m]["validation"]["recall"], -arms[m]["validation"]["fp"], -METHODS.index(m)))
        baseline = metrics([r["label"] for r in val], [s["prompt_margin"] for s in scores], .02)
        results[dataset] = dict(baseline_validation=baseline, methods=arms, selected=selected,
                               validation_records=[dict(id=r["id"], label=r["label"], scores=s) for r, s in zip(val, scores)])
        lock[dataset] = dict(selected=selected, thresholds={m: arms[m]["threshold"] for m in METHODS}, reference_ids=[r["id"] for r in bank], validation_ids=[r["id"] for r in val])
        print("VALIDATION", dataset, json.dumps(results[dataset], ensure_ascii=False), flush=True)
    lock.update(config=config.json(), version=VERSION, test_policy="UT once after lock; UR test sealed; no temporal claims")
    (root / "lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    # No test vectors/labels accessed for method or threshold selection above.
    test = [r for r in rows if r["dataset"] == "UT" and r["split"] == "test"]
    bank = [r for r in rows if r["dataset"] == "UT" and r["split"] == "development"]
    records = []
    for i, row in enumerate(test, 1):
        features[row["id"]] = encode(row, root, client, config, telemetry)
        records.append(dict(id=row["id"], label=row["label"], scores=score(row, features, bank, prompts[row["kind"]])))
        print(f"TEST ENCODE {i}/{len(test)} {row['id']}", flush=True)
    results["UT"]["test_records"] = records
    results["UT"]["baseline_test"] = metrics([r["label"] for r in records], [r["scores"]["prompt_margin"] for r in records], .02)
    for method in METHODS:
        results["UT"]["methods"][method]["test"] = metrics([r["label"] for r in records], [r["scores"][method] for r in records], lock["UT"]["thresholds"][method])
    output = dict(results=results, lock=lock, telemetry=telemetry,
                  limitations=["Clip classification only; not temporal localization or alarms/hour", "UT small official trimmed clips, no hospital certification", "UR preview labels exploratory; timestamps pending review and actors not audited", "No shouting/audio evaluation", "Reference data and validation calibration required; no model weights trained"],
                  encode_seconds=sum(float(f["encode_seconds"]) for f in features.values()),
                  duration_differences={r["id"]: dict(actual=float(features[r["id"]]["actual_duration"]), manifest=r["duration"]) for r in rows if abs(float(features[r["id"]]["actual_duration"])-r["duration"])>.02})
    (root / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", root.resolve(), flush=True)


if __name__ == "__main__":
    main()
