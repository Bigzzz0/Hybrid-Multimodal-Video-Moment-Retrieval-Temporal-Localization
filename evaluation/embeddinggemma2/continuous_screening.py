"""Training-free, temporal UT screening. Labels are used for evaluation/reference
preparation only; no ground-truth boxes or times are inputs at inference.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

from app.gemma2.config import IndexConfig
from app.gemma2.media import extract, metadata, windows
from app.gemma2.ranking import EVENTS
from app.gemma2.service import WorkerClient
from app.gemma2.store import file_hash, vectors
from .training_free import unit

METHODS = ("prompt_margin", "window_reference", "centered_reference", "fusion")


def prepare():
    from evaluation.event_retrieval.acquire_public import download, extract as unpack
    root = Path("scratch/event-retrieval/datasets/ut-full")
    base = "https://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/"
    checksums = {1: "172965ce7208704165448c6d8890d9853e9fee96ffc7e06036bb0ea1b6c29b2f",
                 2: "8c79287321b2aaa0d0a348e9994fd3a9442fafce1f1ac1c8b28a7deef545e07e"}
    for number, expected in checksums.items():
        archive = download(base+f"ut-interaction_set{number}.zip", root/f"set{number}.zip")
        if file_hash(archive) != expected:
            raise ValueError("Original archive changed or download incomplete")
        unpack(archive, root)
    download(base+"ut-interaction_labels_110912.xls", root.parent/"ut-labels.xls")


def overlap(a, b):
    return max(0., min(a[1], b[1])-max(a[0], b[0]))


def interval_iou(a, b):
    intersection = overlap(a, b)
    return intersection / (max(a[1], b[1])-min(a[0], b[0])) if intersection else 0.


def annotations(path):
    import xlrd
    book = xlrd.open_workbook(str(path))
    result = {i: [] for i in range(1, 21)}
    for sheet in book.sheets():
        offset = 1 if sheet.name == "Set1" else 0
        for ri in range(2, sheet.nrows):
            row = sheet.row_values(ri)
            if not isinstance(row[offset], str) or not row[offset].startswith("seq"):
                continue
            seq = int(row[offset][3:])
            if not isinstance(row[offset+1], (int, float)):
                continue
            start, end = int(row[offset+2]), int(row[offset+3])
            if start < 0 or end < start:
                raise ValueError("Invalid source-frame annotation")
            # Original files are verified 30 FPS. Inclusive frame ends become
            # half-open intervals. Frame-number origin may carry 1-frame error.
            result[seq].append(dict(id=f"seq{seq}-row{ri}", class_id=int(row[offset+1]),
                bounds=[start/30., (end+1)/30.], source_frames=[start, end],
                official_selected=isinstance(row[-1], (int, float))))
    return result


def role(seq):
    return "development" if seq <= 12 else "validation" if seq <= 16 else "test"


def selected_targets(events):
    return [e for e in events if e["class_id"] in (4, 5) and e["official_selected"]]


def center(matrix):
    delta = matrix-matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    # Constant/near-constant clips have no usable residual; keep a zero vector
    # and let calibrated scores handle uncertainty rather than creating NaNs.
    return np.divide(delta, norms, out=np.zeros_like(delta), where=norms > 1e-7)


def encode(seq, root, config, client):
    path = Path("scratch/event-retrieval/datasets/ut-full") / f"seq{seq}.avi"
    info = metadata(path)
    if abs(info["fps"]-30) > .01:
        raise ValueError("Label/video frame-rate mismatch")
    fingerprint = file_hash(path)
    key = hashlib.sha256(json.dumps([fingerprint, info, config.json()], sort_keys=True).encode()).hexdigest()
    cache = root / "features" / f"{key}.npz"
    sidecar = cache.with_suffix(".json")
    if cache.exists() and sidecar.exists():
        with np.load(cache, allow_pickle=False) as saved:
            feature = {k: saved[k].copy() for k in saved.files}
        vectors(feature["video"], config.dimension)
        vectors(feature["image"], config.dimension)
        return feature, json.loads(sidecar.read_text(encoding="utf-8"))
    started = time.perf_counter(); video, image, image_windows, telemetry = [], [], [], []
    timeline = windows(info["duration"], config.window_seconds, config.stride_seconds)
    traces = []
    for wi, (start, end) in enumerate(timeline):
        media = extract(path, start, end, root / "frames" / key / str(wi), config, False)
        video.append(client.embed("video", config, frame_paths=media["frame_paths"])[0])
        telemetry.append(client.last_telemetry)
        paths = [media["frame_paths"][s] for s in media["image_slots"]]
        image.extend(client.embed("image", config, frame_paths=paths))
        telemetry.append(client.last_telemetry); image_windows.extend([wi]*len(paths))
        traces.append(dict(bounds=[start, end], timestamps=media["timestamps"], frame_count=media["frame_count"], held_last_frame=media["held_last_frame"]))
        if wi % 10 == 0:
            print(f"ENCODE seq{seq} {wi+1}/{len(timeline)}", flush=True)
    feature = dict(video=np.array(video), image=np.array(image), image_windows=np.array(image_windows), bounds=np.array(timeline))
    provenance = dict(seq=seq, split=role(seq), sha256=fingerprint, path=str(path.resolve()), metadata=info,
        config=config.json(), encode_seconds=time.perf_counter()-started, telemetry=telemetry, traces=traces)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.npz")
    np.savez(temporary, **feature); temporary.replace(cache)
    sidecar.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return feature, provenance


def reference_bank(features, labels):
    positive, normal, c_positive, c_normal, counts = [], [], [], [], {}
    for seq, feature in features.items():
        if role(seq) != "development":
            raise ValueError("Reference leakage: development recordings only")
        targets = selected_targets(labels[seq]); residual = center(feature["video"])
        p_count = n_count = 0
        for wi, bounds in enumerate(feature["bounds"]):
            if any(overlap(bounds, e["bounds"])/(e["bounds"][1]-e["bounds"][0]) >= .5 for e in targets):
                positive.append(feature["video"][wi]); c_positive.append(residual[wi]); p_count += 1
            elif not any(overlap(bounds, e["bounds"]) > 0 for e in labels[seq] if e["class_id"] in (2, 4, 5)):
                # Exclude every annotated push/punch, including official 'others',
                # and kicks from the normal reference bank.
                normal.append(feature["video"][wi]); c_normal.append(residual[wi]); n_count += 1
        counts[seq] = dict(positive=p_count, normal=n_count)
    if not positive or not normal:
        raise ValueError("Both reference classes required")
    return tuple(np.stack(x) for x in (positive, normal, c_positive, c_normal)), counts


def reference_scores(matrix, positive, normal):
    def neighbors(bank):
        similarity = matrix @ bank.T
        return np.sort(similarity, axis=1)[:, -min(3, len(bank)):].mean(axis=1)
    return neighbors(positive)-neighbors(normal)


def score(feature, bank, prompts):
    pos, neg = prompts
    vp = (feature["video"] @ pos.T).max(axis=1)
    vn = (feature["video"] @ neg.T).max(axis=1)
    prompt = np.where(vp >= .35, vp-vn, -2.)
    ip = (feature["image"] @ pos.T).max(axis=1)
    inn = (feature["image"] @ neg.T).max(axis=1)
    np.maximum.at(prompt, feature["image_windows"], np.where(ip >= .35, ip-inn, -2.))
    ref = reference_scores(feature["video"], bank[0], bank[1])
    residual = reference_scores(center(feature["video"]), bank[2], bank[3])
    return dict(prompt_margin=prompt, window_reference=ref, centered_reference=residual, fusion=(prompt+residual)/2)


def measure(records, method, threshold, labels):
    hits, total, iou_hits, selected, windows_total, irrelevant = 0, 0, 0, 0, 0, 0
    duration, full_duration, missed, per_video = 0., 0., [], []
    for seq, feature, scores in records:
        bounds = feature["bounds"]; chosen = bounds[np.asarray(scores[method], dtype=np.float64) >= threshold]
        targets = selected_targets(labels[seq]); local_hits = 0
        for event in targets:
            target = event["bounds"]; length = target[1]-target[0]
            found = any(overlap(b, target)/length >= .5 for b in chosen)
            hits += found; local_hits += found; total += 1
            iou_hits += any(interval_iou(b, target) >= .3 for b in chosen)
            if not found:
                missed.append(dict(seq=seq, **event))
        ignored = [e for e in labels[seq] if e["class_id"] == 2 or (e["class_id"] in (4, 5) and not e["official_selected"])]
        irrelevant += sum(not any(overlap(b, e["bounds"]) > 0 for e in targets+ignored) for b in chosen)
        selected += len(chosen); windows_total += len(bounds)
        # Union duration avoids double-counting overlapping verification windows.
        merged = []
        for start, end in sorted(chosen.tolist()):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        duration += sum(end-start for start, end in merged)
        full_duration += float(np.max(bounds[:, 1])) if len(bounds) else 0.
        per_video.append(dict(seq=seq, covered=local_hits, events=len(targets), selected=len(chosen), windows=len(bounds)))
    return dict(coverage_recall=hits/total if total else 0, covered_events=hits, events=total,
                temporal_iou_recall=iou_hits/total if total else 0, iou_matched_events=iou_hits,
                selected_windows=selected, total_windows=windows_total,
                candidate_reduction=1-selected/windows_total if windows_total else 0,
                non_target_windows=irrelevant, selected_union_seconds=duration, total_seconds=full_duration,
                missed_events=missed, per_video=per_video)


def calibrate(records, method, labels, split="validation"):
    if split not in ("validation", "development") or not records or any(role(seq) != split for seq, _, _ in records):
        raise ValueError(f"Thresholds require {split} only")
    if not any(selected_targets(labels[seq]) for seq, _, _ in records):
        raise ValueError("Calibration requires target events")
    values = np.concatenate([scores[method] for _, _, scores in records])
    thresholds = np.unique(values).tolist() + [float(np.nextafter(float(values.max()), np.inf))]
    options = [(t, measure(records, method, t, labels)) for t in thresholds]
    return max(options, key=lambda item: (item[1]["coverage_recall"] >= .95,
        item[1]["coverage_recall"], -item[1]["selected_windows"], item[0]))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="scratch/event-retrieval/continuous-ut")
    parser.add_argument("--prepare", action="store_true", help="Download/verify original source archives without inference")
    args = parser.parse_args(); root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare()
        return
    sys.path.insert(0, str(Path("scratch/evaluation-tools").resolve()))
    if (root / "lock.json").exists():
        raise RuntimeError("Experiment already locked; keep its results and use a new output directory")
    labels = annotations("scratch/event-retrieval/datasets/ut-labels.xls")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise RuntimeError("Screening jobs active")
    config = IndexConfig(modalities=("image", "video")); client = WorkerClient()
    spec = EVENTS["pushing_punching"]
    texts = spec["positive"]+spec["negative"]
    query = client.embed("query", config, texts=texts)
    prompts = query[:len(spec["positive"])], query[len(spec["positive"]):]
    features, provenance = {}, {}
    for seq in range(1, 17):
        features[seq], provenance[seq] = encode(seq, root, config, client)
        print("READY", seq, flush=True)
    bank, bank_counts = reference_bank({seq: f for seq, f in features.items() if role(seq) == "development"}, labels)
    validation = [(seq, features[seq], score(features[seq], bank, prompts)) for seq in range(13, 17)]
    arms = {}
    for method in METHODS:
        threshold, result = calibrate(validation, method, labels)
        arms[method] = dict(threshold=threshold, validation=result)
        print("VALIDATION", method, threshold, result["covered_events"], result["events"], result["selected_windows"], result["total_windows"], flush=True)
    selected = max(METHODS, key=lambda m: (arms[m]["validation"]["coverage_recall"] >= .95,
        arms[m]["validation"]["coverage_recall"], -arms[m]["validation"]["selected_windows"], -METHODS.index(m)))
    lock = dict(selected=selected, methods=arms, reference_counts=bank_counts, config=config.json(),
                annotation_sha256=file_hash(Path("scratch/event-retrieval/datasets/ut-labels.xls")),
                metric="Each event >=50% covered by at least one candidate; secondary tIoU>=.3", validation_ids=list(range(13,17)), test_ids=list(range(17,21)),
                caveat="Source-disjoint regression; trimmed clips from these test recordings were evaluated in an earlier experiment. Not a never-seen external test.")
    (root / "lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    test = []
    for seq in range(17, 21):
        features[seq], provenance[seq] = encode(seq, root, config, client)
        test.append((seq, features[seq], score(features[seq], bank, prompts)))
        print("TEST READY", seq, flush=True)
    for method in METHODS:
        arms[method]["test"] = measure(test, method, arms[method]["threshold"], labels)
    output = dict(lock=lock, methods=arms, baseline_validation=measure(validation, "prompt_margin", .02, labels),
                  baseline_test=measure(test, "prompt_margin", .02, labels),
                  oracle_test=measure(test, "prompt_margin", -3, labels),
                  oracle_validation=measure(validation, "prompt_margin", -3, labels),
                  scores={str(seq): {"bounds": feature["bounds"].tolist(), **{m: values.tolist() for m, values in scores.items()}} for seq, feature, scores in validation+test},
                  annotations=labels, provenance=provenance,
                  limitations=["Coverage is not precise temporal/spatial detection", "No Qwen confirmation requested; candidate counts are not measured Qwen calls", "Kicks and official others are ignored for non-target-window counts", "Small same-dataset regression; prior trimmed-test exposure", "No fall or audio claims"])
    (root / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    for method, arm in arms.items():
        print("TEST", method, arm["test"]["covered_events"], arm["test"]["events"], arm["test"]["selected_windows"], arm["test"]["total_windows"], flush=True)
    print("COMPLETE", root.resolve(), flush=True)


if __name__ == "__main__":
    main()
