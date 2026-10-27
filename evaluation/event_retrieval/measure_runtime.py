"""API wall-clock latency: first load, fresh query vectors, and cache repeats."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import requests

QUERIES = ["แพนเค้กบนโต๊ะริมหน้าต่าง", "รถไฟวิ่งริมทะเล", "คนเดินริมถนน", "ชายหาดและคลื่นทะเล", "ร้านอาหารริมทะเล",
           "pancakes by a seaside window", "train beside the ocean", "people walking on a street", "beach and waves",
           "helicopter landing on the roof", "สุนัขไล่ลูกบอล", "ร้านขายขนม"]

def measure(video_id, output, api="http://127.0.0.1:8002", queries=QUERIES):
    rows = []
    for query in queries:
        payload = {"video_id": video_id, "query": query, "backend": "gemma2", "profile": "fast", "top_k": 5}
        started = time.perf_counter()
        response = requests.post(api + "/api/v1/search/moment", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        first = time.perf_counter()-started
        repeats = []
        for _ in range(5):
            started = time.perf_counter()
            again = requests.post(api + "/api/v1/search/moment", json=payload, timeout=120)
            again.raise_for_status()
            repeats.append(time.perf_counter()-started)
        rows.append({"query": query, "first_seconds": first, "query_cache_hit": result["cache_hits"]["query"],
                     "cache_repeat_seconds": repeats, "moments": result["moments"], "model_revision": result["model_revision"]})
    fresh = [row["first_seconds"] for row in rows[1:] if not row["query_cache_hit"]]
    report = {"rows": rows, "cold_first_seconds": rows[0]["first_seconds"] if rows else None,
              "fresh_query_p50_seconds": float(np.percentile(fresh, 50)) if fresh else None,
              "fresh_query_p95_seconds": float(np.percentile(fresh, 95)) if fresh else None,
              "cache_repeat_p95_seconds": float(np.percentile([x for row in rows for x in row["cache_repeat_seconds"]], 95)) if rows else None,
              "health": requests.get(api + "/api/v1/events/health", timeout=15).json(),
              "scope": "Runtime only, no accuracy inferred. First response includes model cold load ONLY if broker unloaded and API query cache reset before this run."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = measure(args.video_id, args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows", "health")}, ensure_ascii=False))
