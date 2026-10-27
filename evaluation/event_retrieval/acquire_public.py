"""Acquire original research media with provenance, without inventing AV pairs.

Outputs are ignored. UR RGB timestamps come from the official synchronization
CSV. UT clips remain grouped by the original recording, including both classes.
FSD previews are audio-only diagnostics, not AV shouting ground truth.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import requests

UR = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"
UT = "https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html"
FSD = "https://fsannotator.upf.edu/fsd/release/FSD50K/"

def download(url, target):
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    if "fenix.ur.edu.pl" in url and url.endswith(".zip"):
        # The host's large streaming responses can stall. Its advertised HTTP
        # ranges let acquisition resume without downloading the whole file again.
        probe = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=(15, 30))
        probe.raise_for_status()
        total = int(probe.headers["Content-Range"].split("/")[-1])
        offset = temp.stat().st_size if temp.exists() else 0
        if offset > total:
            raise ValueError("Partial download exceeds source size")
        with temp.open("ab") as stream:
            while offset < total:
                end = min(total-1, offset+1024*1024-1)
                chunk = requests.get(url, headers={"Range": f"bytes={offset}-{end}"}, timeout=(15, 30))
                chunk.raise_for_status()
                if chunk.status_code != 206 or len(chunk.content) != end-offset+1:
                    raise ValueError("Unexpected range response")
                stream.write(chunk.content)
                stream.flush()
                offset = end+1
        temp.replace(target)
        return target
    with requests.get(url, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        with temp.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    temp.replace(target)
    return target

def extract(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        root = target.resolve()
        for member in zipped.infolist():
            path = (target / member.filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Archive path outside dataset directory")
            if not member.is_dir():
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.is_file():
                    path.write_bytes(zipped.read(member))

def info(path):
    data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], text=True))
    return float(data["format"]["duration"]), any(row["codec_type"] == "audio" for row in data["streams"])

def split(sequence, total):
    return "development" if sequence <= total * .6 else "validation" if sequence <= total * .8 else "test"

def ur_sequence(root, page, name, total, preview=False):
    links = re.findall(r'href\s*=\s*"([^"]+)"', page)
    rgb = next(urljoin(UR, link) for link in links if link.endswith(name + "-cam0-rgb.zip"))
    sync = next(urljoin(UR, link) for link in links if link.endswith(name + "-data.csv"))
    if preview:
        media_url = next(urljoin(UR, link) for link in links if link.endswith(name + "-cam0.mp4"))
        media = download(media_url, root / (name + "-cam0-preview.mp4"))
        video = root / "preview-rgb" / (name + ".mp4")
        video.parent.mkdir(exist_ok=True)
        if not video.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(media), "-vf", "crop=iw/2:ih:iw/2:0", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True)
        duration, _ = info(video)
        events = []
        if name.startswith("fall"):
            labels = download(urljoin(UR, "data/urfall-cam0-falls.csv"), root / "urfall-cam0-falls.csv")
            falling = [int(row[1]) for row in csv.reader(io.StringIO(labels.read_text())) if row[0] == name and row[2] == "0"]
            if falling:
                events = [{"kind": "falling", "t_start": (min(falling)-1)/30, "t_end": min(duration, max(falling)/30)}]
        return {"id": "ur-"+name, "path": str(video.resolve()), "duration": duration, "has_audio": False,
                "recording_id": "ur-"+name, "split": split(int(name.rsplit("-",1)[1]), total), "source_url": UR,
                "media_source_url": media_url, "license": "CC BY-NC-SA 4.0; academic non-commercial",
                "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "audio_visual_pairing": "no_audio",
                "annotation_method": "Official falling label 0 / 30 FPS preview; frame alignment requires human review before held-out evaluation",
                "annotation_review": "pending", "quality": "Official preview RGB panel, 320x240; excluded from resolution comparison",
                "events": events, "hard_negatives": ["falling"] if name.startswith("adl") else []}
    archive = download(rgb, root / (name + "-cam0-rgb.zip"))
    data = download(sync, root / (name + "-data.csv"))
    folder = root / "rgb" / name
    extract(archive, folder)
    frames = sorted(folder.rglob("*.png"))
    stamps = {int(row[0]): float(row[1]) / 1000 for row in csv.reader(io.StringIO(data.read_text())) if row and row[0].isdigit()}
    timed = [(int(re.search(r'(\d+)\.png$', frame.name).group(1)), frame) for frame in frames]
    if len(timed) != len(stamps) or any(number not in stamps for number, _ in timed):
        raise ValueError("UR RGB frame/time alignment requires inspection: " + name)
    origin = min(stamps.values())
    ordered = sorted((stamps[number] - origin, frame, number) for number, frame in timed)
    deltas = [ordered[i+1][0]-ordered[i][0] for i in range(len(ordered)-1)]
    if any(delta <= 0 for delta in deltas):
        raise ValueError("Non-monotonic source timestamps")
    # Preserve actual capture times instead of assuming that all frames are 30 FPS.
    concat = folder / "frames.ffconcat"
    lines = ["ffconcat version 1.0"]
    for i, (_, frame, _) in enumerate(ordered):
        lines += ["file '" + frame.resolve().as_posix().replace("'", "'\\''") + "'", "duration " + str(deltas[i] if i < len(deltas) else deltas[-1])]
    concat.write_text("\n".join(lines), encoding="utf-8")
    video = root / "videos" / (name + ".mp4")
    video.parent.mkdir(exist_ok=True)
    if not video.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-safe", "0", "-i", str(concat), "-fps_mode", "vfr", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True)
    duration, audio = info(video)
    events = []
    if name.startswith("fall"):
        labels = download(urljoin(UR, "data/urfall-cam0-falls.csv"), root / "urfall-cam0-falls.csv")
        falling = [int(row[1]) for row in csv.reader(io.StringIO(labels.read_text())) if row[0] == name and row[2] == "0"]
        if falling:
            start = max(0, stamps[min(falling)]-origin)
            stop = min(duration, stamps[max(falling)]-origin + deltas[-1])
            events = [{"kind": "falling", "t_start": start, "t_end": stop}]
    sequence = int(name.rsplit("-", 1)[1])
    return {"id": "ur-" + name, "path": str(video.resolve()), "duration": duration, "has_audio": audio,
            "recording_id": "ur-" + name, "split": split(sequence, total), "source_url": UR,
            "media_source_url": rgb, "license": "CC BY-NC-SA 4.0; academic non-commercial",
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "audio_visual_pairing": "no_audio",
            "annotation_method": "Official cam0 temporary-falling label 0, aligned with synchronization timestamps; depth/RGB nearest-time alignment",
            "events": events, "hard_negatives": ["falling"] if name.startswith("adl") else []}

def acquire(root, ur_count=30, fsd_count=30, ur_preview=False):
    root.mkdir(parents=True, exist_ok=True)
    videos, diagnostics = [], []
    for set_number in (1, 2):
        url = urljoin(UT, f"videos/competition_1/ut-interaction_segmented_set{set_number}.zip")
        archive = download(url, root / f"ut-set{set_number}.zip")
        folder = root / "ut"
        extract(archive, folder)
        for path in sorted((folder / f"segmented_set{set_number}").glob("*.avi")):
            _, recording, label = map(int, path.stem.split("_"))
            duration, audio = info(path)
            original = path
            playable = root / "ut-mp4" / (path.stem + ".mp4")
            playable.parent.mkdir(exist_ok=True)
            if not playable.exists():
                subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(playable)], check=True)
            path = playable
            positive = label in (4, 5)
            videos.append({"id": "ut-" + path.stem, "path": str(path.resolve()), "recording_id": f"ut-seq{recording}",
                "split": split(recording, 20), "duration": duration, "has_audio": audio, "source_url": UT, "media_source_url": url,
                "source_class_id": label, "original_path": str(original.resolve()), "license": "Research release; cite Ryoo and Aggarwal, UT-Interaction 2010; no explicit CC license",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "audio_visual_pairing": "original" if audio else "no_audio",
                "annotation_method": "Official action-trimmed segment, full segment interval; class 4 punch/5 push; not a continuous-recording localization evaluation",
                "events": [{"kind": "pushing_punching", "t_start": 0., "t_end": duration}] if positive else [],
                "hard_negatives": ["pushing_punching"] if label in (0, 1, 3) else []})
    page = requests.get(UR, timeout=30).text
    names = [f"{kind}-{number:02d}" for kind in ("fall", "adl") for number in range(1, ur_count+1)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(lambda name: ur_sequence(root, page, name, ur_count, ur_preview), names):
            videos.append(row)
            print(row["id"], "ready", flush=True)
            (root / "manifest.partial.json").write_text(json.dumps({"videos": videos}, ensure_ascii=False, indent=2), encoding="utf-8")
    # Faithful preview audio stays audio-only and is never paired with unrelated frames.
    for label, mid in (("Shout", "07p6fty"), ("Speech", "09x0r")):
        urls = []
        for page_number in range(1, (fsd_count+9)//10 + 1):
            page_url = FSD + "%252Fm%252F" + mid + f"/?page={page_number}"
            html = requests.get(page_url, timeout=30).text
            urls.extend(re.findall(r'sound_url:\s*"([^"]+)"', html))
        for url in list(dict.fromkeys(urls))[:fsd_count]:
            actual = "https:" + url if url.startswith("//") else url
            try:
                path = download(actual, root / "audio" / label / actual.rsplit("/", 1)[-1])
                diagnostics.append({"label": label, "path": str(path.resolve()), "source_url": actual, "annotation_source": FSD,
                    "pairing": "audio_only", "license": "Per Freesound clip license; retain uploader attribution before redistribution",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            except requests.RequestException as exc:
                diagnostics.append({"source_url": actual, "status": "unavailable", "error": str(exc)})
    manifest = {"videos": videos, "audio_diagnostics": diagnostics,
                "limitations": ["No real AV shouting videos yet", "No annotated 90-minute continuous normal sample yet", "UT actor/background overlap across recording splits; report generalization limitation", "FSD previews are diagnostics, not AV ground truth"]}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ur-count", type=int, default=30)
    parser.add_argument("--fsd-count", type=int, default=30)
    parser.add_argument("--ur-preview", action="store_true", help="Explicit 320x240 RGB preview fallback; annotations await review")
    args = parser.parse_args()
    acquire(args.output, args.ur_count, args.fsd_count, args.ur_preview)
