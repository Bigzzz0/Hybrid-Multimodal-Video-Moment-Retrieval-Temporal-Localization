"""Versioned, scene-specific normal references; no learned model weights."""
import hashlib
import json
import math
import time
import uuid

import numpy as np

from .store import ReindexRequired, vectors

POLICY = "open-anomaly-v1"
PROMPTS = {
    "normal": ["People walking, sitting, waiting or talking calmly in their usual places.",
               "Routine cleaning, deliveries and moving a wheelchair or trolley safely.",
               "Vehicles traveling normally in permitted areas.",
               "คนเดิน นั่งรอ พูดคุย และทำกิจกรรมตามปกติในพื้นที่"],
    "suspicious": ["An unusual dangerous interaction, a person in distress or an unexpected collapse.",
                   "People suddenly fleeing, chasing someone or struggling with each other.",
                   "A vehicle entering a pedestrian area or colliding with a person or object.",
                   "Smoke, fire, a falling object or damage disrupting normal activity.",
                   "A person crossing a restricted boundary or leaving an obstructing object.",
                   "An unusual scream, crash, explosion or alarm disrupting normal ambient sound.",
                   "เกิดเหตุผิดปกติ คนมีอาการเดือดร้อน การเคลื่อนไหวหรือสิ่งแวดล้อมเปลี่ยนอย่างน่าสงสัย"],
}
DEFAULT_THRESHOLDS = {m: {"novelty": .25, "margin": .03, "similarity": .35}
                      for m in ("image", "video", "audio", "caption")}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def create_profile(store, name, context):
    if not name.strip() or not context.strip():
        raise ValueError("Profile name and context are required")
    profile = {"id": str(uuid.uuid4()), "name": name.strip(), "context": context.strip(),
               "created_at": time.time(), "current_version": None}
    with store.db() as db:
        db.execute("INSERT INTO normality_profiles VALUES (?,?)", (profile["id"], json.dumps(profile)))
    return profile


def profiles(store):
    with store.db() as db:
        return [json.loads(row[0]) for row in db.execute("SELECT data FROM normality_profiles ORDER BY id")]


def profile(store, identifier):
    with store.db() as db:
        row = db.execute("SELECT data FROM normality_profiles WHERE id=?", (identifier,)).fetchone()
    if row is None:
        raise KeyError(identifier)
    return json.loads(row[0])


def bank(store, identifier, version=None):
    record = profile(store, identifier)
    version = record["current_version"] if version is None else version
    if version is None:
        return None
    with store.db() as db:
        row = db.execute("SELECT data FROM normality_banks WHERE profile_id=? AND version=?", (identifier, version)).fetchone()
    if row is None:
        raise KeyError(version)
    return json.loads(row[0])


def add_references(service, identifier, ranges, expected_version=None):
    """Copy complete indexed development windows; atomic publication, no GPU work."""
    current = profile(service.store, identifier)
    previous = bank(service.store, identifier)
    config = json.loads(json.dumps(service.config.json()))
    if previous and previous["config"] != config:
        raise ReindexRequired("Reference configuration changed; create a new profile and rebuild")
    references = list(previous["references"]) if previous else []
    if not ranges:
        raise ValueError("At least one verified normal range is required")
    for interval in ranges:
        if interval.get("split") != "development" or interval.get("verified_normal") is not True:
            raise ValueError("Only explicitly verified normal development ranges may enter the bank")
        video = service.store.get("videos", interval["video_id"])
        if video.get("dataset_split") in ("validation", "test"):
            raise ValueError("Known validation/test recording cannot enter a normal reference bank")
        service.validate_source(video)
        a, b = interval["t_start"], interval["t_end"]
        if not all(math.isfinite(x) for x in (a, b)) or not 0 <= a < b <= video["duration"]:
            raise ValueError("Invalid normal reference interval")
        namespace = service.namespace(video)
        timeline = service.store.require(video["id"], namespace)
        channels = service.store.embeddings(namespace, service.config.dimension)
        selected = [w for w in timeline if w["t_start"] >= a and w["t_end"] <= b]
        if not selected:
            raise ValueError("Range contains no complete indexed window")
        for window in selected:
            evidence = {"video_id": video["id"], "fingerprint": video["fingerprint"], "namespace": namespace,
                        "window_id": window["id"], "t_start": window["t_start"], "t_end": window["t_end"],
                        "split": "development", "verified_normal": True, "review_event_id": interval.get("review_event_id")}
            evidence["key"] = digest([namespace, window["id"]])
            evidence["channels"] = {m: values[ids == window["id"]].tolist() for m, (ids, values) in channels.items()
                                    if m in DEFAULT_THRESHOLDS and np.any(ids == window["id"])}
            if not any(r["key"] == evidence["key"] for r in references):
                references.append(evidence)
    updated = {"profile_id": identifier, "version": digest([config, references, current["context"]]),
               "config": config, "context": current["context"], "references": references, "created_at": time.time()}
    with service.store.db() as db:
        latest = json.loads(db.execute("SELECT data FROM normality_profiles WHERE id=?", (identifier,)).fetchone()[0])
        if latest["current_version"] != current["current_version"] or (expected_version is not None and expected_version != latest["current_version"]):
            raise ValueError("Reference bank changed concurrently; refresh and retry")
        latest["current_version"] = updated["version"]
        db.execute("INSERT OR IGNORE INTO normality_banks VALUES (?,?,?)", (identifier, updated["version"], json.dumps(updated)))
        db.execute("UPDATE normality_profiles SET data=? WHERE id=?", (json.dumps(latest), identifier))
    return updated


def score_channels(channels, reference_bank, prompts, thresholds, dimension):
    """OR of independent channel gates; never average unlike raw modality scores."""
    evidence = {}
    for modality, (ids, raw) in channels.items():
        if modality not in thresholds:
            continue
        matrix = vectors(raw, dimension)
        positive, negative = (vectors(p, dimension) for p in prompts)
        normal_vectors, owners = [], []
        for ref in (reference_bank or {}).get("references", []):
            # One normalized centroid per reference window, not duplicate image slots.
            if ref["channels"].get(modality):
                mean = vectors(ref["channels"][modality], dimension).mean(axis=0)
                normal_vectors.append(vectors([mean], dimension)[0]); owners.append(ref)
        normal = np.stack(normal_vectors) if normal_vectors else None
        limits = thresholds[modality]
        for window_id, value in zip(ids, matrix):
            pos, neg = float((positive @ value).max()), float((negative @ value).max())
            nearest = np.argsort(-(normal @ value), kind="stable")[:3] if normal is not None else []
            novelty = float(np.mean(1 - np.clip(normal[nearest] @ value, -1, 1))) if len(nearest) else None
            semantic = pos >= limits["similarity"] and pos - neg >= limits["margin"]
            novel = novelty is not None and novelty >= limits["novelty"]
            item = {"similarity": pos, "normal_similarity": neg, "margin": pos-neg, "novelty": novelty,
                    "reference_status": "ready" if len(nearest) else "reference_missing",
                    "selected": bool(novel or semantic), "triggers": (["normality_deviation"] if novel else []) + (["semantic_concern"] if semantic else []),
                    "references": [{k: owners[i][k] for k in ("key", "video_id", "window_id", "t_start", "t_end", "fingerprint")} for i in nearest]}
            entries = evidence.setdefault(int(window_id), {}).setdefault(modality, [])
            entries.append(item)
    return evidence


def segments(timeline, evidence):
    hits = [w for w in timeline if any(s["selected"] for entries in evidence.get(w["id"], {}).values() for s in entries)]
    groups = []
    for row in sorted(hits, key=lambda w: w["t_start"]):
        if groups and row["t_start"] <= groups[-1]["t_end"] + 3:
            groups[-1]["t_end"] = max(groups[-1]["t_end"], row["t_end"])
            groups[-1]["source_windows"].append(row)
        else:
            groups.append({"t_start": row["t_start"], "t_end": row["t_end"], "source_windows": [row]})
    result = []
    for group in groups:
        for start in np.arange(group["t_start"], group["t_end"], 12.):
            end = min(float(start)+12, group["t_end"])
            ids = [w["id"] for w in group["source_windows"] if w["t_start"] < end and w["t_end"] > start]
            result.append({"kind": "open_anomaly", "t_start": float(start), "t_end": end,
                           "window_ids": ids, "evidence": {str(i): evidence[i] for i in ids}})
    return result


def sample_frames(path, start, end, duration, folder, context=True):
    """Decode requested timestamps directly, deduplicate by actual source-frame index."""
    import cv2
    from pathlib import Path
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    stamps = list(np.linspace(start, end, 8, endpoint=False))
    if context and start > 0:
        stamps += list(np.linspace(max(0, start-5), start, 2, endpoint=False))
    if context and end < duration:
        stamps += list(np.linspace(end, min(duration, end+5), 2, endpoint=False))
    capture = cv2.VideoCapture(str(path)); result = []
    try:
        fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or count <= 0:
            raise ValueError("Video cannot be decoded")
        indices = sorted({min(count-1, max(0, int(round(t*fps)))) for t in stamps})
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Missing verification frame {index}")
            target = folder / f"{index}.jpg"
            if not cv2.imwrite(str(target), image):
                raise OSError("Cannot save verification frame")
            result.append({"timestamp": index/fps, "path": str(target.resolve())})
    finally:
        capture.release()
    return result
