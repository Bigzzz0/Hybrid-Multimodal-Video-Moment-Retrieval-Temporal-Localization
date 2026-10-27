"""Copy completed experimental evidence into the local demo, preserving reviews.

Does not install thresholds, train, enqueue inference, or promote a default.
Reference banks contain development examples only; validation clips are reviewable.
"""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import requests

from app.gemma2.config import DATA
from app.gemma2.normality import digest
from app.gemma2.store import Store, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--target", default=str(DATA))
    args = parser.parse_args()
    experiment = Path(args.experiment)
    result_path = experiment / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("production_activation") or not result.get("code_unchanged"):
        raise ValueError("Only completed unchanged experimental runs may be published")
    ledger = {r["id"]: r for r in result["lock"].get("source_ledger", [])}
    if not ledger:
        raise ValueError("Explicit source split ledger is required")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise ValueError("Demo jobs active")
    target = Store(args.target)
    source = sqlite3.connect(f"file:{(experiment/'store/events.sqlite3').resolve().as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        events = [json.loads(r[0]) for r in source.execute("SELECT data FROM events")]
        profiles = {e["open_policy"]["profile_id"] for e in events}
        provenance = file_hash(result_path)
        with target.db() as db:
            for row in source.execute("SELECT * FROM videos"):
                video = json.loads(row["data"])
                if video["id"] not in ledger:
                    raise ValueError("Video not in audited source ledger")
                video["dataset_split"] = ledger[video["id"]]["split"]
                video["dataset"] = ledger[video["id"]]["dataset"]
                if file_hash(Path(video["path"])) != video["fingerprint"]:
                    raise ValueError("Published video source changed")
                old = db.execute("SELECT data FROM videos WHERE id=?", (video["id"],)).fetchone()
                if old and json.loads(old[0])["fingerprint"] != video["fingerprint"]:
                    raise ValueError("Demo video ID collision")
                db.execute("INSERT OR REPLACE INTO videos VALUES (?,?)", (row["id"], json.dumps(video)))
            for table in ("indexes", "windows", "embeddings", "active"):
                for row in source.execute(f"SELECT * FROM {table}"):
                    placeholders = ",".join("?" for _ in row)
                    db.execute(f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})", tuple(row))
            for profile_id in profiles:
                row = source.execute("SELECT data FROM normality_profiles WHERE id=?", (profile_id,)).fetchone()
                profile = json.loads(row[0]); profile["name"] = "Demo · " + profile["name"] + " · synthetic validation subset"
                db.execute("INSERT OR IGNORE INTO normality_profiles VALUES (?,?)", (profile_id, json.dumps(profile)))
                for row in source.execute("SELECT * FROM normality_banks WHERE profile_id=?", (profile_id,)):
                    bank = json.loads(row["data"])
                    if any(r.get("split") != "development" or not r.get("verified_normal") for r in bank["references"]):
                        raise ValueError("Unsafe reference provenance")
                    db.execute("INSERT OR IGNORE INTO normality_banks VALUES (?,?,?)", tuple(row))
        job_ids = {}
        for raw_job in result["verification_jobs"]:
            job = dict(raw_job)
            job_ids[job["id"]] = "demo-" + provenance[:10] + "-" + job["id"]
            job.update(id=job_ids[job["id"]], kind="screen", mode="open_anomaly", status="complete", phase="complete",
                       created_at=time.time(), error="", profile_id=job["open_policy"]["profile_id"],
                       imported_benchmark_hash=provenance, experimental_subset=True)
            target.put("jobs", job["id"], job)
        for event in events:
            event["id"] = "demo-" + provenance[:10] + "-" + event["id"]
            event["screening_job_id"] = job_ids[event["screening_job_id"]]
            event["imported_benchmark_hash"] = provenance
            event["experimental_subset"] = True
            target.save_event(event)
        print(json.dumps({"profiles": list(profiles), "events": len(events), "result_hash": provenance}))
    finally:
        source.close()


if __name__ == "__main__":
    main()
