"""Native multimodal Kamakura experiment, isolated from production indexes."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .prepare import ROOT, load_config, file_hash

VIDEO = Path("C:/Users/User/Downloads/Ghibli-Like Seaside Town _ Kamakura One Day Train Trip near Tokyo _ JAPAN TRAVEL VLOG.mp4")
WORK = ROOT / "scratch/embeddinggemma2/kamakura"
sys.path.insert(0, str(ROOT / "backend"))


class ResourceMonitor:
    """Process RSS/CPU and device-wide GPU telemetry; sampled at one second."""
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        self.process.cpu_percent()
        while not self.stop.is_set():
            sample = {"rss_mib": self.process.memory_info().rss / 1024**2,
                      "cpu_percent_one_core": self.process.cpu_percent()}
            try:
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"], text=True, timeout=5)
                memory, utilization, power = map(float, raw.strip().splitlines()[0].split(","))
                sample.update(device_vram_mib=memory, gpu_percent=utilization, device_power_w=power)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.samples.append(sample)
            self.stop.wait(1)

    def __enter__(self):
        self.started = time.perf_counter()
        self.cpu_started = self.process.cpu_times()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        elapsed = time.perf_counter() - self.started
        cpu = self.process.cpu_times()
        self.summary = {"wall_seconds": elapsed,
            "cpu_core_seconds": cpu.user + cpu.system - self.cpu_started.user - self.cpu_started.system,
            "sample_count": len(self.samples),
            "peaks": {key: max(row[key] for row in self.samples if key in row) for key in set().union(*(row.keys() for row in self.samples))},
            "means": {key: float(np.mean([row[key] for row in self.samples if key in row])) for key in set().union(*(row.keys() for row in self.samples))}}


def prepare():
    import cv2
    from PIL import Image, ImageDraw
    WORK.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    step = round(fps)
    folder = WORK / "frames"
    folder.mkdir(exist_ok=True)
    if len(list(folder.glob("*.jpg"))) != (count + step - 1) // step:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(VIDEO),
            "-vf", f"select=not(mod(n\\,{step}))", "-fps_mode", "vfr", "-q:v", "3", str(folder / "frame-%06d.jpg")], check=True)
    frames = [{"id": f"kamakura:{i}", "timestamp": i * step / fps, "path": str(path)} for i, path in enumerate(sorted(folder.glob("*.jpg")))]
    audio = WORK / "audio-16k.wav"
    if not audio.is_file():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(VIDEO), "-vn", "-ac", "1", "-ar", "16000", str(audio)], check=True)
    media = {"video_id": "kamakura", "path": str(VIDEO), "duration_sec": count / fps,
             "frames": frames, "audio_path": str(audio), "media_sha256": file_hash(VIDEO), "sample_hz": fps / step}
    (WORK / "media.json").write_text(json.dumps(media, indent=2), encoding="utf-8")
    # Review every 24 seconds, independent of model rankings.
    selected = frames[::24]
    for page, offset in enumerate(range(0, len(selected), 25)):
        sheet = Image.new("RGB", (1250, 825), "#101827")
        draw = ImageDraw.Draw(sheet)
        for slot, frame in enumerate(selected[offset:offset+25]):
            with Image.open(frame["path"]) as image:
                image.thumbnail((248, 140))
                x, y = (slot % 5) * 250, (slot // 5) * 165
                sheet.paste(image, (x, y))
                draw.text((x+5, y+142), f'{frame["timestamp"]:.1f}s', fill="white")
        sheet.save(WORK / f"review-{page}.jpg")
    print(json.dumps({"duration": media["duration_sec"], "frames": len(frames), "audio": str(audio)}), flush=True)


def run(profile, limit=None):
    import torch
    import soundfile as sf
    from PIL import Image
    from .encoders import SigLIPEncoder, EmbeddingGemmaEncoder, normalized_vectors
    from .benchmark import sync_gpu
    config = load_config()
    media = json.loads((WORK / "media.json").read_text(encoding="utf-8"))
    frames = media["frames"]
    windows = [(i, min(i+6, len(frames))) for i in range(0, len(frames), 6)]
    if limit:
        windows = windows[:limit]
    role = "baseline" if profile == "siglip" else "candidate"
    full = profile == "gemma_av"
    waveform, sr = sf.read(media["audio_path"], dtype="float32")
    assert sr == 16000 and waveform.ndim == 1
    with ResourceMonitor() as total_monitor:
        with ResourceMonitor() as load_monitor:
            encoder = (SigLIPEncoder if role == "baseline" else EmbeddingGemmaEncoder)(
                ROOT / config[role]["local_dir"], "cuda", 768, **({"full_multimodal": full} if full else {}))
            sync_gpu("cuda")
        parameter_count = sum(p.numel() for p in (encoder.model.parameters() if role == "baseline" else encoder.model.parameters()))
        def embed_window(start_i, end_i):
            images = []
            try:
                for frame in frames[start_i:end_i]:
                    with Image.open(frame["path"]) as image:
                        images.append(image.convert("RGB"))
                if full:
                    start = frames[start_i]["timestamp"]
                    end = min(media["duration_sec"], frames[end_i-1]["timestamp"] + 1 / media["sample_hz"])
                    audio = waveform[round(start*sr):round(end*sr)]
                    return encoder.encode_clip(images, audio, vision_tokens=1120, sample_hz=media["sample_hz"])[0]
                # Equal six-second retrieval units for all profiles; visual baselines
                # aggregate individual frame vectors, native AV jointly encodes them.
                return normalized_vectors(encoder.encode_images(images).mean(axis=0, keepdims=True), 1, 768)[0]
            finally:
                for image in images:
                    image.close()
        # Warm up both media and query paths BEFORE timing indexing and queries.
        embed_window(*windows[0])
        encoder.encode_queries(["a train travelling beside the sea"])
        sync_gpu("cuda")
        torch.cuda.reset_peak_memory_stats()
        vectors = []
        times = []
        with ResourceMonitor() as index_monitor:
            for number, (start_i, end_i) in enumerate(windows):
                vectors.append(embed_window(start_i, end_i))
                sync_gpu("cuda")
                times.append([frames[start_i]["timestamp"], media["duration_sec"] if end_i == len(frames) else frames[end_i]["timestamp"]])
                if number % 10 == 0 or number+1 == len(windows):
                    print(f"{profile}: {number+1}/{len(windows)} windows", flush=True)
        matrix = np.stack(vectors)
        queries = json.loads((WORK / "queries.json").read_text(encoding="utf-8")) if (WORK / "queries.json").is_file() else []
        rows = []
        with ResourceMonitor() as query_monitor:
            for query in queries:
                elapsed = []
                for _ in range(5):
                    sync_gpu("cuda")
                    started = time.perf_counter()
                    vector = encoder.encode_queries([query["query"]])[0]
                    sync_gpu("cuda")
                    scores = matrix @ vector
                    order = np.argsort(-scores, kind="stable")[:5]
                    elapsed.append((time.perf_counter() - started)*1000)
                hits = [{"window_id": int(i), "start": times[i][0], "end": times[i][1], "score": float(scores[i]), "frame_index": windows[i][0]+(windows[i][1]-windows[i][0])//2} for i in order]
                rows.append({**query, "latency_ms": float(np.median(elapsed)), "top_windows": hits})
        report = {"profile": profile, "full_clip": limit is None, "media_sha256": media["media_sha256"],
            "model": config[role], "parameter_count": parameter_count, "window_seconds": 6,
            "vision_soft_tokens_per_frame": 1120 if full else "model default",
            "modalities": ["video", "audio", "text query"] if full else ["image", "text query"],
            "warmup": "one media window and one text query excluded", "query_repeats": 5,
            "window_count": len(windows), "load_resources": load_monitor.summary,
            "index_resources": index_monitor.summary, "query_resources": query_monitor.summary,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
            "index_windows_per_second": len(windows)/index_monitor.summary["wall_seconds"],
            "queries": rows, "telemetry_scope": "RSS and CPU of benchmark process; nvidia-smi GPU/power/memory are device-wide, not incremental or energy measurements"}
        if rows:
            report["query_p50_ms"] = float(np.median([row["latency_ms"] for row in rows]))
            report["query_p95_ms"] = float(np.percentile([row["latency_ms"] for row in rows],95))
            report["accuracy"] = {f"CenterFrameHit@{k}": float(np.mean([any(hit["window_id"] in row["positive_windows"] for hit in row["top_windows"][:k]) for row in rows]))*100 for k in (1,5)}
            report["accuracy_scope"] = "Manually reviewed center-frame visual relevance; does not measure exact temporal IoU, audio recognition, or no-match queries. Ten events, each phrased in two languages, are not 20 independent events."
        encoder.close()
    report["total_resources"] = total_monitor.summary
    suffix = "-probe" if limit else ""
    np.savez_compressed(WORK / f"{profile}{suffix}-vectors.npz", vectors=matrix, intervals=np.asarray(times))
    (WORK / f"{profile}{suffix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{profile}: done, {index_monitor.summary['wall_seconds']:.1f}s indexing", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--profile", choices=["siglip", "gemma_frames", "gemma_av"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.profile and args.run:
        run(args.profile, args.limit)
    else:
        parser.error("Use --prepare or explicitly --run --profile NAME")
