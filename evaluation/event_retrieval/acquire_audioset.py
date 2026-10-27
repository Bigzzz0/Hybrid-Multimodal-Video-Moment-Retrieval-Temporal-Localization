"""Fetch original AV segments listed by AudioSet; do not bypass inaccessible videos.

AudioSet labels are weak labels. Every downloaded segment remains pending human
review, including onset/offset and whether the voice belongs to a visible person.
The acquisition report records unavailable sources without inventing replacements.
"""
import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
import requests
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func

CSV = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
SOURCE = "https://research.google.com/audioset/download.html"
SHOUT = "/m/07p6fty"

def acquire(root, limit=30, consecutive_failures=3):
    root.mkdir(parents=True, exist_ok=True)
    response = requests.get(CSV, timeout=30)
    response.raise_for_status()
    (root / "balanced_train_segments.csv").write_bytes(response.content)
    rows = csv.reader(io.StringIO("\n".join(line for line in response.text.splitlines() if not line.startswith("#"))), skipinitialspace=True)
    selected = [row for row in rows if len(row) == 4 and SHOUT in row[3].split(",")]
    report, videos, failed = [], [], 0
    for row in selected:
        identifier, start, stop = row[0], float(row[1]), float(row[2])
        target = root / (identifier + ".mp4")
        record = {"id": identifier, "source_url": "https://www.youtube.com/watch?v="+identifier, "source_segment": [start, stop], "annotation_source": SOURCE}
        try:
            if not target.is_file():
                with YoutubeDL({"format": "bestvideo[height<=480]+bestaudio/best[height<=480]", "outtmpl": str(root / (identifier+".%(ext)s")),
                        "merge_output_format": "mp4", "download_ranges": download_range_func(None, [(start, stop)]),
                        "force_keyframes_at_cuts": True, "socket_timeout": 15, "retries": 0, "fragment_retries": 0,
                        "quiet": True, "noplaylist": True}) as downloader:
                    downloader.download([record["source_url"]])
            if not target.is_file():
                raise ValueError("No browser-compatible AV segment produced")
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(target)], text=True))
            duration = float(probe["format"]["duration"])
            if {"audio", "video"} - {stream["codec_type"] for stream in probe["streams"]} or abs(duration-(stop-start)) > 1:
                raise ValueError("Missing synchronized AV or incorrect segment duration")
            videos.append({**record, "id": "audioset-"+identifier, "recording_id": "youtube-"+identifier,
                          "split": "development", "path": str(target.resolve()), "duration": duration,
                          "events": [], "proposed_labels": ["shouting"], "annotation_review": "pending", "has_audio": True,
                          "audio_visual_pairing": "original", "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                          "license": "Original uploader rights apply; AudioSet metadata is not a media redistribution license"})
            record["status"], failed = "downloaded_pending_review", 0
        except Exception as exc:
            failed += 1
            record.update(status="unavailable", error=str(exc))
        report.append(record)
        (root / "acquisition.json").write_text(json.dumps({"videos": videos, "attempts": report, "listed_shout_segments": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(identifier, record["status"], flush=True)
        if len(videos) >= limit or failed >= consecutive_failures:
            break
    return videos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    acquire(args.output, args.limit)
