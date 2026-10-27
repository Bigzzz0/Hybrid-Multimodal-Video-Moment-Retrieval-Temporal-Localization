"""Local experiment API. All uploads and indexes stay under scratch/."""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .prepare import ROOT, file_hash, load_config
from .encoders import normalized_vectors

sys.path.insert(0, str(ROOT / "backend"))


class SearchRequest(BaseModel):
    video_id: str
    query: str = Field(min_length=1, max_length=300)
    mode: Literal["both", "baseline", "candidate", "candidate_av"] = "both"


class ExperimentStore:
    def __init__(self, root: Path = ROOT, config: dict | None = None):
        self.root = root
        self.config = config or load_config()
        self.work = root / "scratch/embeddinggemma2/ui"
        self.work.mkdir(parents=True, exist_ok=True)
        self.catalog: dict[str, dict] = {}
        self.indices: dict[str, dict] = {}
        self.full_indices: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}
        self.state_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedding-experiment")
        self.encoder = None
        self.encoder_role = None
        self._load_catalog()
        self._load_full_cache()

    def _load_full_cache(self):
        folder = self.root / "scratch/embeddinggemma2/kamakura"
        if not (folder / "gemma_av.json").is_file():
            return
        report = json.loads((folder / "gemma_av.json").read_text(encoding="utf-8"))
        media = json.loads((folder / "media.json").read_text(encoding="utf-8"))
        if not report["full_clip"] or report["model"]["revision"] != self.config["candidate"]["revision"]:
            return
        with np.load(folder / "gemma_av-vectors.npz", allow_pickle=False) as archive:
            vectors = normalized_vectors(archive["vectors"], report["window_count"], 768)
            intervals = archive["intervals"].copy()
        for key, video in self.catalog.items():
            if file_hash(Path(video["path"])) == report["media_sha256"]:
                self.indices[key] = {"video_id": key, "duration_sec": video["duration_sec"], "frames": media["frames"], "vectors": {}, "actual_sample_hz": media["sample_hz"]}
                self.full_indices[key] = {"vectors": vectors, "intervals": intervals,
                    "frame_indices": [min(i*6+3, len(media["frames"])-1) for i in range(len(intervals))], "has_audio": True}

    def _load_catalog(self):
        dataset = json.loads((self.root / self.config["dataset"]).read_text(encoding="utf-8"))
        for entry in dataset["videos"]:
            path = self.root / self.config["media_dir"] / entry["filename"]
            if path.is_file():
                self.catalog[entry["video_id"]] = {
                    "id": entry["video_id"], "name": entry["filename"], "path": str(path),
                    "duration_sec": entry["duration_sec"], "source": "benchmark",
                    "examples": [q for event in entry["events"] for q in event["queries"]],
                }
        for manifest in (self.work / "uploads").glob("*/video.json"):
            record = json.loads(manifest.read_text(encoding="utf-8"))
            if Path(record["path"]).is_file():
                self.catalog[record["id"]] = record
        # Restore only compatible vectors from the completed benchmark.
        result_root = self.root / self.config["output_dir"]
        for run in sorted(result_root.glob("*"), reverse=True):
            if not (run / "candidate.json").is_file() or not (run / "baseline.json").is_file():
                continue
            for video_id, video in self.catalog.items():
                if video_id in self.indices or video["source"] != "benchmark":
                    continue
                try:
                    saved_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
                    if saved_config["sample_hz"] != self.config["sample_hz"]:
                        continue
                    arrays = {}
                    timestamps = None
                    media_hash = file_hash(Path(video["path"]))
                    for role in ("baseline", "candidate"):
                        with np.load(run / f"{role}-{video_id}-vectors.npz", allow_pickle=False) as archive:
                            if str(archive["revision"]) != self.config[role]["revision"] or str(archive["model_id"]) != self.config[role]["model_id"] or str(archive["media_sha256"]) != media_hash:
                                raise ValueError("Stale vector artifact")
                            role_times = archive["timestamps"].copy()
                            if timestamps is not None and not np.array_equal(timestamps, role_times):
                                raise ValueError("Model indexes use different frames")
                            timestamps = role_times
                            arrays[role] = normalized_vectors(archive["vectors"], len(timestamps), self.config["embedding_dim"])
                    paths = sorted((run / "frames" / video_id).glob("frame-*.png"))
                    if len(paths) != len(timestamps):
                        continue
                    self.indices[video_id] = {"video_id": video_id, "duration_sec": video["duration_sec"],
                        "frames": [{"id": f"{video_id}:{i}", "timestamp": float(ts), "path": str(path)} for i, (ts, path) in enumerate(zip(timestamps, paths))],
                        "vectors": arrays}
                except (OSError, ValueError, KeyError):
                    continue

    def video(self, video_id: str) -> dict:
        with self.state_lock:
            if video_id not in self.catalog:
                raise HTTPException(404, "ไม่พบวิดีโอที่เลือก")
            return dict(self.catalog[video_id])

    def public_videos(self) -> list[dict]:
        with self.state_lock:
            return [{k: v for k, v in record.items() if k != "path"} | {
                "indexed_models": sorted(set(self.indices.get(key, {}).get("vectors", {})) | ({"candidate_av"} if key in self.full_indices else set())),
                "frame_count": len(self.indices.get(key, {}).get("frames", [])),
            } for key, record in self.catalog.items()]

    def update_job(self, job_id: str, **values):
        with self.state_lock:
            self.jobs[job_id].update(values)

    def _load_encoder(self, role: str) -> float:
        if self.encoder_role == role and self.encoder is not None:
            return 0.0
        if self.encoder is not None:
            self.encoder.close()
            self.encoder = None
            self.encoder_role = None
        from .encoders import SigLIPEncoder, EmbeddingGemmaEncoder
        started = time.perf_counter()
        kind = SigLIPEncoder if role == "baseline" else EmbeddingGemmaEncoder
        config_role = "candidate" if role == "candidate_av" else role
        kwargs = {"full_multimodal": True} if role == "candidate_av" else {}
        self.encoder = kind(self.root / self.config[config_role]["local_dir"], self.config["device"], self.config["embedding_dim"], **kwargs)
        self.encoder_role = role
        # Attribute first-use GPU/kernel setup to preparation, keeping the displayed
        # cached-query time comparable with the warmed benchmark measurements.
        self.encoder.encode_queries(["a train travelling beside the sea"])
        from .benchmark import sync_gpu
        sync_gpu(self.config["device"])
        return (time.perf_counter() - started) * 1000

    def submit(self, request: SearchRequest) -> str:
        self.video(request.video_id)
        if not request.query.strip():
            raise HTTPException(422, "กรุณาพิมพ์เหตุการณ์ที่ต้องการค้นหา")
        job_id = str(uuid.uuid4())
        with self.state_lock:
            active = sum(j["status"] in {"queued", "running"} for j in self.jobs.values())
            if active >= 8:
                raise HTTPException(429, "มีคำค้นรออยู่หลายรายการ กรุณารอสักครู่")
            if len(self.jobs) >= 100:
                for key in list(self.jobs):
                    if self.jobs[key]["status"] in {"done", "error"}:
                        del self.jobs[key]
                        break
            self.jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0,
                                 "message": "รอคิวค้นหา", "result": None, "error": None}
        self.executor.submit(self._search, job_id, request)
        return job_id

    def _search(self, job_id: str, request: SearchRequest):
        from PIL import Image
        from .benchmark import sample_frames, retrieve_moments, sync_gpu
        started = time.perf_counter()
        roles = ["baseline", "candidate"] if request.mode == "both" else [request.mode]
        try:
            if request.mode == "candidate_av":
                self._search_full(job_id, request, started)
                return
            self.update_job(job_id, status="running", progress=5, message="กำลังเตรียมวิดีโอ")
            video = self.video(request.video_id)
            if request.video_id not in self.indices:
                sampled = sample_frames(Path(video["path"]), request.video_id, self.config["sample_hz"], self.work,
                    on_progress=lambda current, total: self.update_job(job_id, progress=5 + round(current / total * 4), message=f"เตรียมภาพจากวิดีโอ {current}/{total}"))
                sampled["vectors"] = {}
                with self.state_lock:
                    self.indices[request.video_id] = sampled
            index = self.indices[request.video_id]
            results = {}
            for role_i, role in enumerate(roles):
                name = "SigLIP2" if role == "baseline" else "EmbeddingGemma 2"
                base_progress = 10 + role_i * (80 // len(roles))
                self.update_job(job_id, progress=base_progress, message=f"กำลังโหลด {name}")
                load_ms = self._load_encoder(role)
                index_ms = 0.0
                if role not in index["vectors"]:
                    index_started = time.perf_counter()
                    batches = []
                    frames = index["frames"]
                    batch_size = self.config["image_batch_size"]
                    for offset in range(0, len(frames), batch_size):
                        images = []
                        try:
                            for frame in frames[offset:offset + batch_size]:
                                with Image.open(frame["path"]) as image:
                                    images.append(image.convert("RGB"))
                            batches.append(self.encoder.encode_images(images))
                        finally:
                            for image in images:
                                image.close()
                        processed = min(len(frames), offset + batch_size)
                        self.update_job(job_id, progress=base_progress + round((processed / len(frames)) * (60 / len(roles))),
                                        message=f"{name} เตรียมภาพ {processed}/{len(frames)}")
                    with self.state_lock:
                        index["vectors"][role] = np.concatenate(batches)
                    index_ms = (time.perf_counter() - index_started) * 1000
                self.update_job(job_id, message=f"{name} กำลังค้นหาเหตุการณ์")
                sync_gpu(self.config["device"])
                query_started = time.perf_counter()
                qvec = self.encoder.encode_queries([request.query.strip()])[0]
                sync_gpu(self.config["device"])
                moments, best_frames = retrieve_moments(index, index["vectors"][role], qvec, self.config)
                query_ms = (time.perf_counter() - query_started) * 1000
                nearest = lambda timestamp: min(range(len(index["frames"])), key=lambda i: abs(index["frames"][i]["timestamp"] - timestamp))
                results[role] = {"name": name, "query_ms": round(query_ms, 2), "load_ms": round(load_ms, 2),
                    "index_ms": round(index_ms, 2),
                    "moments": [{"start": a, "end": b, "frame_index": nearest((a+b)/2)} for a, b in moments],
                    "frames": [dict(frame, frame_index=nearest(frame["timestamp"])) for frame in best_frames]}
            self.update_job(job_id, status="done", progress=100, message="ค้นหาเสร็จแล้ว",
                result={"query": request.query.strip(), "video_id": request.video_id, "roles": results,
                        "frame_count": len(index["frames"]), "sample_hz": self.config["sample_hz"],
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})
        except Exception as exc:
            self.update_job(job_id, status="error", message="ค้นหาไม่สำเร็จ", error=f"{type(exc).__name__}: {exc}")
            if self.encoder is not None:
                self.encoder.close()
                self.encoder = None
                self.encoder_role = None

    def _search_full(self, job_id, request, started):
        """Native chronological video + real audio. No synthetic captions or answer text."""
        import subprocess
        import soundfile as sf
        from PIL import Image
        from .benchmark import sample_frames, sync_gpu
        self.update_job(job_id, status="running", progress=5, message="เตรียมวิดีโอและเสียง")
        video = self.video(request.video_id)
        if request.video_id not in self.indices:
            index = sample_frames(Path(video["path"]), request.video_id, self.config["sample_hz"], self.work,
                on_progress=lambda current, total: self.update_job(job_id, progress=5+round(current/total*5), message=f"เตรียมภาพ {current}/{total}"))
            index["vectors"] = {}
            self.indices[request.video_id] = index
        index = self.indices[request.video_id]
        self.update_job(job_id, progress=12, message="โหลด EmbeddingGemma 2 พร้อม vision และ audio encoder")
        load_ms = self._load_encoder("candidate_av")
        index_ms = 0.0
        if request.video_id not in self.full_indices:
            index_started = time.perf_counter()
            audio_path = self.work / f"{request.video_id}-audio.wav"
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", video["path"]], capture_output=True, text=True, check=True)
            has_audio = bool(json.loads(probe.stdout).get("streams"))
            if has_audio:
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video["path"], "-vn", "-ac", "1", "-ar", "16000", str(audio_path)], check=True)
                waveform, sr = sf.read(audio_path, dtype="float32")
            else:
                waveform, sr = None, 16000
            frames = index["frames"]
            vectors, intervals, centers = [], [], []
            for offset in range(0, len(frames), 6):
                records = frames[offset:offset+6]
                start = records[0]["timestamp"]
                end = frames[offset+6]["timestamp"] if offset+6 < len(frames) else video["duration_sec"]
                images = []
                try:
                    for frame in records:
                        with Image.open(frame["path"]) as image:
                            images.append(image.convert("RGB"))
                    audio = waveform[round(start*sr):round(end*sr)] if has_audio else None
                    vectors.append(self.encoder.encode_clip(images, audio, sample_hz=index.get("actual_sample_hz", self.config["sample_hz"]))[0])
                finally:
                    for image in images:
                        image.close()
                intervals.append([start, end])
                centers.append(offset+len(records)//2)
                self.update_job(job_id, progress=15+round((offset+len(records))/len(frames)*75), message=f"ประมวลผลภาพหลายเฟรม{'และเสียง' if has_audio else ' (คลิปไม่มีเสียง)'} {offset+len(records)}/{len(frames)}")
            self.full_indices[request.video_id] = {"vectors": np.stack(vectors), "intervals": np.asarray(intervals), "frame_indices": centers, "has_audio": has_audio}
            index_ms = (time.perf_counter()-index_started)*1000
        full = self.full_indices[request.video_id]
        self.update_job(job_id, progress=95, message="ค้นหาในช่วงวิดีโอร่วมกับเสียง")
        sync_gpu(self.config["device"])
        query_started = time.perf_counter()
        vector = self.encoder.encode_queries([request.query.strip()])[0]
        sync_gpu(self.config["device"])
        scores = full["vectors"] @ vector
        order = np.argsort(-scores, kind="stable")[:5]
        query_ms = (time.perf_counter()-query_started)*1000
        result = {"name": "EmbeddingGemma 2 · Video + Audio" if full["has_audio"] else "EmbeddingGemma 2 · Video (ไม่มีเสียง)", "load_ms": load_ms, "index_ms": index_ms,
            "query_ms": query_ms, "moments": [{"start": float(full["intervals"][i][0]), "end": float(full["intervals"][i][1]), "frame_index": full["frame_indices"][i]} for i in order],
            "frames": [{"timestamp": index["frames"][full["frame_indices"][i]]["timestamp"], "cosine": float(scores[i]), "frame_index": full["frame_indices"][i]} for i in order]}
        self.update_job(job_id, status="done", progress=100, message="ค้นหาเสร็จแล้ว", result={"query": request.query.strip(), "video_id": request.video_id,
            "roles": {"candidate_av": result}, "frame_count": len(index["frames"]), "sample_hz": self.config["sample_hz"], "elapsed_ms": (time.perf_counter()-started)*1000,
            "window_count": len(full["vectors"]), "has_audio": full["has_audio"]})


def create_app(store: ExperimentStore | None = None) -> FastAPI:
    store = store or ExperimentStore()
    app = FastAPI(title="EmbeddingGemma 2 experiment")
    app.state.store = store
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"], allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"status": "ready", "models": ["SigLIP2", "EmbeddingGemma 2"], "loaded_model": store.encoder_role}

    @app.get("/videos")
    def videos():
        return store.public_videos()

    @app.get("/benchmark")
    def benchmark():
        for folder in sorted((store.root / store.config["output_dir"]).glob("*"), reverse=True):
            if (folder / "candidate.json").is_file() and (folder / "baseline.json").is_file():
                reports = {role: json.loads((folder / f"{role}.json").read_text(encoding="utf-8")) for role in ("baseline", "candidate")}
                return {"query_count": len(reports["baseline"]["queries"]), "roles": {role: report["metrics"] for role, report in reports.items()}}
        return None

    @app.get("/benchmark/kamakura")
    def kamakura_benchmark():
        folder = store.root / "scratch/embeddinggemma2/kamakura"
        return {profile: json.loads((folder / f"{profile}.json").read_text(encoding="utf-8")) for profile in ("siglip", "gemma_frames", "gemma_av") if (folder / f"{profile}.json").is_file()}

    @app.post("/videos/upload")
    def upload(file: UploadFile):
        import cv2
        extension = Path(file.filename or "").suffix.lower()
        if extension not in {".mp4", ".mov", ".avi", ".webm", ".m4v"}:
            raise HTTPException(415, "รองรับไฟล์ MP4 MOV AVI WEBM และ M4V")
        video_id = str(uuid.uuid4())
        folder = store.work / "uploads" / video_id
        folder.mkdir(parents=True)
        path = folder / ("video" + extension)
        size = 0
        with path.open("wb") as output:
            while block := file.file.read(1024 * 1024):
                size += len(block)
                if size > 1024**3:
                    output.close()
                    path.unlink(missing_ok=True)
                    raise HTTPException(413, "ไฟล์มีขนาดเกิน 1 GB")
                output.write(block)
        capture = cv2.VideoCapture(str(path))
        fps, frames = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        if fps <= 0 or frames <= 0:
            path.unlink(missing_ok=True)
            raise HTTPException(422, "ไม่สามารถอ่านวิดีโอนี้ได้ กรุณาลองไฟล์อื่น")
        record = {"id": video_id, "name": Path(file.filename or "Video").name, "path": str(path),
                  "duration_sec": frames / fps, "source": "upload", "examples": ["คนกำลังเดิน", "person walking", "คนกำลังนั่ง"]}
        (folder / "video.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        with store.state_lock:
            store.catalog[video_id] = record
        return {k: v for k, v in record.items() if k != "path"} | {"indexed_models": [], "frame_count": 0}

    @app.get("/videos/{video_id}/stream")
    def stream(video_id: str):
        record = store.video(video_id)
        return FileResponse(record["path"])

    @app.get("/videos/{video_id}/frames/{frame_index}")
    def frame(video_id: str, frame_index: int):
        store.video(video_id)
        with store.state_lock:
            frames = store.indices.get(video_id, {}).get("frames", [])
            if not 0 <= frame_index < len(frames):
                raise HTTPException(404, "ไม่พบภาพตัวอย่าง")
            path = frames[frame_index]["path"]
        return FileResponse(path, media_type="image/jpeg" if Path(path).suffix.lower() in {".jpg", ".jpeg"} else "image/png")

    @app.post("/search", status_code=202)
    def search(request: SearchRequest):
        return {"job_id": store.submit(request)}

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        with store.state_lock:
            if job_id not in store.jobs:
                raise HTTPException(404, "ไม่พบคำค้นนี้")
            return dict(store.jobs[job_id])

    @app.on_event("shutdown")
    def shutdown():
        store.executor.shutdown(wait=True)
        if store.encoder is not None:
            store.encoder.close()

    return app


app = create_app()
