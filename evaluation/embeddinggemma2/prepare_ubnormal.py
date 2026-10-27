"""Fetch a bounded official development/validation subset using ZIP HTTP ranges.

No test split or test video is requested. Original videos are copied byte-for-byte;
labels follow the authors' write_frame_level_ground_truth.py track convention.
"""
import argparse
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import requests

from app.gemma2.media import metadata
from app.gemma2.store import file_hash
from .open_anomaly_eval import audit, union

SOURCE = "https://github.com/lilygeorgescu/UBnormal"
DRIVE = "https://drive.usercontent.google.com/download"
FILE_ID = "1KbfdyasribAMbbKoBU1iywAhtoAt9QI0"


def select_recordings(split_lists, scenes, per_validation_class, development_count=2):
    """Choose names before opening annotations; never read official test lists."""
    if per_validation_class < 1 or development_count < 1 or not scenes:
        raise ValueError("Positive sample counts and at least one scene required")
    chosen = []
    for scene in sorted(set(scenes)):
        groups = {key: sorted(v for v in names if f"_scene_{scene}_" in v)
                  for key, names in split_lists.items()}
        if len(groups[("normal", "training")]) < development_count or any(
                not groups[(cat, "validation")] for cat in ("normal", "abnormal")):
            raise ValueError(f"Scene {scene} lacks independent normal development or both validation classes")
        chosen.extend((v, "development", "training", scene)
                      for v in groups[("normal", "training")][:development_count])
        chosen.extend((v, "validation", "validation", scene) for cat in ("normal", "abnormal")
                      for v in groups[(cat, "validation")][:per_validation_class])
    return chosen


class RangeFile(io.RawIOBase):
    def __init__(self, session, url):
        self.session, self.url, self.pos = session, url, 0
        response = session.get(url, headers={"Range": "bytes=-22"}, timeout=45)
        response.raise_for_status()
        if response.status_code != 206:
            raise ValueError("Server must support byte ranges; refusing an unbounded download")
        self.size = int(response.headers["Content-Range"].split("/")[-1])
    def seekable(self):
        return True
    def tell(self):
        return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos
    def read(self, length=-1):
        if length < 0:
            length = self.size-self.pos
        if not length:
            return b""
        start = self.pos
        response = self.session.get(self.url, headers={"Range": f"bytes={start}-{start+length-1}"}, timeout=90)
        response.raise_for_status()
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
            raise ValueError("Server ignored requested range")
        value = response.content
        if len(value) != length:
            raise ValueError("Truncated range")
        self.pos += len(value)
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scratch/event-retrieval/datasets/ubnormal-open")
    parser.add_argument("--scene", type=int, default=4)
    parser.add_argument("--scenes", type=int, nargs="+", help="Explicit scene IDs selected before reading labels")
    parser.add_argument("--per-validation-class", type=int, default=1)
    parser.add_argument("--cache-dir", help="Reuse original video bytes from a prior download without modifying it")
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("Use a fresh output; do not overwrite a prior experiment manifest")
    session = requests.Session()
    response = session.get("https://api.github.com/repos/lilygeorgescu/UBnormal/commits/main", timeout=30)
    response.raise_for_status()
    revision = response.json()["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid source revision")
    split_lists = {}
    for category, split in (("normal", "training"), ("normal", "validation"), ("abnormal", "validation")):
        name = f"{category}_{split}_video_names.txt"
        response = session.get(f"https://raw.githubusercontent.com/lilygeorgescu/UBnormal/{revision}/scripts/{name}", timeout=30)
        response.raise_for_status()
        (root / name).write_bytes(response.content)
        split_lists[(category, split)] = response.text.splitlines()
    chosen = select_recordings(split_lists, args.scenes or [args.scene], args.per_validation_class)
    selection = {"source_revision": revision, "scenes": sorted(set(args.scenes or [args.scene])),
                 "per_validation_class": args.per_validation_class,
                 "recordings": [v[0] for v in chosen], "test_opened": False}
    (root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    response = session.get(DRIVE, params={"id": FILE_ID, "export": "download"}, timeout=30)
    response.raise_for_status()
    parameters = dict(re.findall(r'name="([^"]+)" value="([^"]+)"', response.text))
    url = requests.Request("GET", DRIVE, params=parameters or {"id": FILE_ID, "export": "download", "confirm": "t"}).prepare().url
    archive = zipfile.ZipFile(RangeFile(session, url))
    (root / "license.txt").write_bytes(archive.read("license.txt"))
    rows = []
    for name, split, official, scene in chosen:
        member = f"Scene{scene}/{name}.mp4"
        target = root / (name + ".mp4")
        if not target.exists() and args.cache_dir:
            cached = Path(args.cache_dir) / target.name
            if cached.is_file():
                shutil.copyfile(cached, target)
        # Check cached/existing bytes against the authoritative archive CRC/size.
        if target.exists():
            import zlib
            checksum = 0
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum = zlib.crc32(block, checksum)
            entry = archive.getinfo(member)
            if target.stat().st_size != entry.file_size or checksum != entry.CRC:
                raise ValueError("Cached source differs from official archive: " + name)
        if not target.exists():
            temporary = target.with_suffix(".part")
            temporary.write_bytes(archive.read(member))
            temporary.replace(target)
        info = metadata(target)
        intervals = []
        if name.startswith("abnormal"):
            tracks_member = f"Scene{scene}/{name}_annotations/{name}_tracks.txt"
            tracks_path = root / (name + "_tracks.txt")
            if not tracks_path.exists():
                tracks_path.write_bytes(archive.read(tracks_member))
            tracks = np.atleast_2d(np.loadtxt(tracks_path, delimiter=","))
            intervals = union([(float(t[1])/info["fps"], min(info["duration"], (float(t[2])+1)/info["fps"])) for t in tracks])
        row = {"id": name, "recording_id": name.removesuffix("_fog"), "scene_id": f"UBnormal-{scene}",
               "dataset": "UBnormal", "split": split, "official_split": official,
               "path": str(target.resolve()), "sha256": file_hash(target), **info,
               "events": [{"t_start": a, "t_end": b} for a, b in intervals],
               "source_url": SOURCE, "license": "CC-BY-NC-ND-4.0", "synthetic": True}
        rows.append(row)
        print(json.dumps({"downloaded": name, "duration": info["duration"], "events": intervals}), flush=True)
    audit(rows)
    manifest = {"version": "open-anomaly-subset-v2", "subset": True, "selection": selection,
                "source": SOURCE, "videos": rows, "test_opened": False}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
