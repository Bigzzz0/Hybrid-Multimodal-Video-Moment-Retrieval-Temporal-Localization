"""Read-only cross-domain window probe; decisions are not ground-truth accuracy."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import requests

from app.gemma2.config import IndexConfig
from app.gemma2.ranking import EVENTS
from app.gemma2.service import WorkerClient
from app.gemma2.store import vectors
from .training_free import METHODS, encode, score, sequence_feature, unit


def main():
    root = Path("scratch/event-retrieval/training-free")
    lock = json.loads((root / "lock.json").read_text(encoding="utf-8"))
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise RuntimeError("Screening jobs active")
    manifest = json.loads(Path("scratch/event-retrieval/datasets/manifest.combined.json").read_text(encoding="utf-8"))["videos"]
    banks, features, client = {}, {}, WorkerClient()
    config = IndexConfig(modalities=("image", "video"))
    for dataset in ("UT", "UR-preview"):
        ids = set(lock[dataset]["reference_ids"])
        bank = []
        for raw in manifest:
            if raw["id"] in ids:
                row = dict(raw, label=bool(raw["events"]))
                if row["split"] != "development":
                    raise ValueError("Reference split changed")
                features[row["id"]] = encode(row, root, client, config, [])
                bank.append(row)
        if len(bank) != len(ids):
            raise ValueError("Missing reference")
        banks[dataset] = bank
    texts = [t for kind in ("pushing_punching", "falling") for role in ("positive", "negative") for t in EVENTS[kind][role]]
    lookup = dict(zip(texts, client.embed("query", config, texts=texts)))
    prompts = {kind: tuple(np.stack([lookup[t] for t in EVENTS[kind][role]]) for role in ("positive", "negative")) for kind in ("pushing_punching", "falling")}
    output = dict(purpose="Exploratory cross-domain window decisions; no temporal ground truth, no tuning or activation", videos={})
    conn = sqlite3.connect("file:backend/data/gemma2/events.sqlite3?mode=ro", uri=True)
    for video_id, dataset, kind in (("ucf-long-fighting011", "UT", "pushing_punching"),
                                    ("ucf-long-normal634", "UT", "pushing_punching"),
                                    ("ur-fall-demo", "UR-preview", "falling")):
        ns = conn.execute("SELECT namespace FROM active WHERE video_id=?", (video_id,)).fetchone()[0]
        active, status = conn.execute("SELECT config,status FROM indexes WHERE namespace=?", (ns,)).fetchone()
        active = json.loads(active)
        for key in ("model_id", "revision", "dimension", "video_tokens", "image_tokens", "video_fps", "image_fps", "window_seconds", "stride_seconds"):
            if active[key] != config.json()[key]:
                raise ValueError("Index configuration mismatch")
        if status != "ready":
            raise ValueError("Incomplete index")
        records = []
        for window_id, data in conn.execute("SELECT id,data FROM windows WHERE namespace=? ORDER BY id", (ns,)):
            media = {}
            for modality in ("image", "video"):
                values = [np.frombuffer(blob, dtype=np.float32) for (blob,) in conn.execute("SELECT vector FROM embeddings WHERE namespace=? AND window_id=? AND modality=? ORDER BY slot", (ns, window_id, modality))]
                media[modality] = vectors(values, 768)
            key = f"probe-{video_id}-{window_id}"
            features[key] = dict(images=media["image"], videos=media["video"], video=unit(media["video"].mean(axis=0)), sequence=sequence_feature(media["image"]))
            scores = score(dict(id=key), features, banks[dataset], prompts[kind])
            timeline = json.loads(data)
            records.append(dict(window_id=window_id, t_start=timeline["t_start"], t_end=timeline["t_end"], scores=scores))
        counts = {method: sum(r["scores"][method] >= lock[dataset]["thresholds"][method] for r in records) for method in METHODS}
        output["videos"][video_id] = dict(windows=len(records), fixed_baseline=sum(r["scores"]["prompt_margin"] >= .02 for r in records), selected=lock[dataset]["selected"], counts=counts, records=records)
        print(video_id, output["videos"][video_id]["selected"], counts, flush=True)
    conn.close()
    (root / "transfer.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
