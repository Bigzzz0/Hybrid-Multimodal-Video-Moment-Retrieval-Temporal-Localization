"""Read-only transfer check for the locked continuous-UT experiment."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import requests

from app.gemma2.config import IndexConfig
from app.gemma2.ranking import EVENTS
from app.gemma2.service import WorkerClient
from app.gemma2.store import vectors
from .continuous_screening import METHODS, annotations, reference_bank, score


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="scratch/event-retrieval/continuous-ut")
    root = Path(parser.parse_args().output)
    result = json.loads((root/"results.json").read_text(encoding="utf-8"))
    crossfit = json.loads((root/"crossfit-lock.json").read_text(encoding="utf-8")) if (root/"crossfit-lock.json").exists() else None
    config = IndexConfig(modalities=("image", "video"))
    if result["lock"]["config"] != json.loads(json.dumps(config.json())):
        raise ValueError("Experiment configuration changed")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise RuntimeError("Screening jobs active")
    features = {}
    for sidecar in (root/"features").glob("*.json"):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record["seq"] <= 12:
            with np.load(sidecar.with_suffix(".npz"), allow_pickle=False) as data:
                features[record["seq"]] = {k: data[k].copy() for k in data.files}
    if set(features) != set(range(1, 13)):
        raise ValueError("Incomplete development bank")
    sys.path.insert(0, str(Path("scratch/evaluation-tools").resolve()))
    bank, counts = reference_bank(features, annotations("scratch/event-retrieval/datasets/ut-labels.xls"))
    if json.loads(json.dumps(counts)) != result["lock"]["reference_counts"]:
        raise ValueError("Reference bank changed")
    spec = EVENTS["pushing_punching"]
    query = WorkerClient().embed("query", config, texts=spec["positive"]+spec["negative"])
    prompts = query[:len(spec["positive"])], query[len(spec["positive"]):]
    conn = sqlite3.connect("file:backend/data/gemma2/events.sqlite3?mode=ro", uri=True)
    output = dict(purpose="Exploratory transfer decisions, not temporal accuracy; no tuning or policy activation", videos={})
    for video_id in ("ucf-long-fighting011", "ucf-long-normal634"):
        namespace = conn.execute("SELECT namespace FROM active WHERE video_id=?", (video_id,)).fetchone()[0]
        raw, status = conn.execute("SELECT config,status FROM indexes WHERE namespace=?", (namespace,)).fetchone()
        current = json.loads(raw)
        if status != "ready":
            raise ValueError("Index incomplete")
        for key in ("model_id", "revision", "dimension", "video_tokens", "image_tokens", "video_fps", "image_fps", "window_seconds", "stride_seconds"):
            if current[key] != config.json()[key]:
                raise ValueError("Index configuration mismatch")
        video, image, image_windows, bounds = [], [], [], []
        for wi, (window_id, data) in enumerate(conn.execute("SELECT id,data FROM windows WHERE namespace=? ORDER BY id", (namespace,))):
            row = json.loads(data); bounds.append([row["t_start"], row["t_end"]])
            for modality, destination in (("video", video), ("image", image)):
                values = [np.frombuffer(b, dtype=np.float32) for (b,) in conn.execute("SELECT vector FROM embeddings WHERE namespace=? AND window_id=? AND modality=? ORDER BY slot", (namespace, window_id, modality))]
                checked = vectors(values, 768)
                if modality == "video" and len(checked) != 1:
                    raise ValueError("Expected one video vector per window")
                destination.extend(checked)
                if modality == "image":
                    image_windows.extend([wi]*len(checked))
        feature = dict(video=np.array(video), image=np.array(image), image_windows=np.array(image_windows), bounds=np.array(bounds))
        scores = score(feature, bank, prompts)
        selected = {method: [int(i) for i in np.flatnonzero(scores[method] >= result["methods"][method]["threshold"])] for method in METHODS}
        output["videos"][video_id] = dict(total_windows=len(bounds), baseline=int(np.sum(scores["prompt_margin"] >= .02)),
            selected_method=result["lock"]["selected"], counts={m: len(ids) for m, ids in selected.items()},
            selected_ids=selected, bounds=bounds, scores={m: s.tolist() for m, s in scores.items()})
        if crossfit:
            output["videos"][video_id].update(crossfit_selected_method=crossfit["selected"],
                crossfit_counts={m: int(np.sum(scores[m] >= crossfit["arms"][m]["threshold"])) for m in METHODS})
        print(video_id, output["videos"][video_id]["counts"], flush=True)
        if crossfit:
            print("CROSSFIT", output["videos"][video_id]["crossfit_counts"], flush=True)
    conn.close()
    (root/"transfer.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
