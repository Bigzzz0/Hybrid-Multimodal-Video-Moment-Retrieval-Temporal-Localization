"""Read the source checkout only; register real videos and existing active captions.

Never transfers SigLIP2 vectors, generated outputs, secrets, or environments into Git.
"""
import argparse
import json
from pathlib import Path
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--api", default="http://127.0.0.1:8002")
    args = parser.parse_args()
    import lancedb
    source = lancedb.connect(str(args.source / "backend/data/lancedb"))
    videos = source.open_table("videos").to_arrow().to_pylist()
    artifacts = source.open_table("vlm_caption_artifacts_v1").to_arrow().to_pylist()
    metadata = source.open_table("vlm_artifact_metadata_v1").to_arrow().to_pylist()
    wanted = "ee5359e0-b1c0-4226-b6c0-ec9c2edae6c1"
    for video in videos:
        if video["id"] != wanted:
            continue
        versions = {row.get("active_artifact_version") for row in metadata if row["video_id"] == wanted and row.get("active_artifact_version")}
        if not versions:
            # Older additive schemas have no active-version column. A ready metadata
            # row plus one unambiguous stored version is the legacy active contract.
            ready = any(row["video_id"] == wanted and row.get("status") == "ready" for row in metadata)
            available = {row["artifact_version"] for row in artifacts if row["video_id"] == wanted and row.get("status") == "ready"}
            if ready and len(available) == 1:
                versions = available
            else:
                raise ValueError("Cannot identify the active caption version; refusing a guessed import")
        captions = [{"t_start": row["t_start"], "t_end": row["t_end"],
                     "text": row.get("structured_json") or row.get("caption_text") or row.get("summary", ""),
                     "source": "existing_caprl_q6", "artifact_id": row["id"]}
                    for row in artifacts if row["video_id"] == wanted and row.get("artifact_version") in versions and row.get("status") == "ready"]
        response = requests.post(args.api + "/api/v1/events/videos/register", json={"path": video["filepath"], "video_id": wanted, "captions": captions, "name": video.get("filename") or "Kamakura — Japan Travel Vlog"}, timeout=120)
        response.raise_for_status()
        print(json.dumps({"id": wanted, "caption_count": len(captions)}, ensure_ascii=False))
    for name in ("benchmark_person_bicycle_car", "benchmark_classroom"):
        path = args.source / "backend/data/raw_videos" / (name + ".mp4")
        response = requests.post(args.api + "/api/v1/events/videos/register", json={"path": str(path), "video_id": name}, timeout=120)
        response.raise_for_status()
        print(json.dumps({"id": name, "registered": True}))

if __name__ == "__main__":
    main()
