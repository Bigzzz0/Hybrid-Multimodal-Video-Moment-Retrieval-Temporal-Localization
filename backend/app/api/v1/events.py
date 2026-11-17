"""Recorded-video screening API, with durable jobs and separate human reviews."""
import os
import shutil
import hashlib
import subprocess
import uuid
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.gemma2.media import ffmpeg
from app.gemma2.service import get_service
from app.gemma2.store import ReindexRequired
from app.gemma2 import normality

router = APIRouter()

class RegisterRequest(BaseModel):
    path: str
    video_id: str | None = None
    captions: list[dict] = Field(default_factory=list)
    name: str | None = Field(default=None, max_length=250)

class JobRequest(BaseModel):
    video_id: str
    kind: str = "screen"
    event_types: list[str] = Field(default_factory=lambda: ["pushing_punching", "falling"])
    resume_id: str | None = None
    mode: str = "specific"
    profile_id: str | None = None

class ReviewRequest(BaseModel):
    decision: str
    note: str = Field(default="", max_length=2000)

def translate(operation):
    try:
        return operation()
    except KeyError as exc:
        raise HTTPException(404, "Not found") from exc
    except ReindexRequired as exc:
        raise HTTPException(409, {"code": "reindex_required", "message": str(exc)}) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc

class ProfileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    context: str = Field(min_length=1, max_length=3000)

class ReferenceRange(BaseModel):
    video_id: str
    t_start: float
    t_end: float
    split: str = "development"
    verified_normal: bool = False

class ReferenceRequest(BaseModel):
    ranges: list[ReferenceRange]
    expected_version: str | None = None

class NormalReviewRequest(BaseModel):
    verified_normal: bool = False
    split: str = "development"

@router.get("/profiles")
def profiles():
    return normality.profiles(get_service().store)

@router.post("/profiles", status_code=201)
def create_profile(request: ProfileRequest):
    return translate(lambda: normality.create_profile(get_service().store, request.name, request.context))

@router.get("/profiles/{profile_id}")
def profile(profile_id: str):
    return translate(lambda: normality.profile(get_service().store, profile_id))

@router.get("/profiles/{profile_id}/bank")
def reference_bank(profile_id: str, version: str | None = None):
    record = translate(lambda: normality.bank(get_service().store, profile_id, version))
    if record:
        record = {**record, "references": [{k: v for k, v in r.items() if k != "channels"} for r in record["references"]]}
    return record

@router.post("/profiles/{profile_id}/references")
def add_references(profile_id: str, request: ReferenceRequest):
    result = translate(lambda: normality.add_references(get_service(), profile_id,
        [r.model_dump() for r in request.ranges], request.expected_version))
    return {"profile_id": profile_id, "version": result["version"], "reference_count": len(result["references"])}

@router.post("/profiles/{profile_id}/normal-events/{event_id}")
def add_normal_event(profile_id: str, event_id: str, request: NormalReviewRequest):
    service = get_service()
    event = translate(lambda: service.store.get("events", event_id))
    if event.get("review") != "rejected" or not request.verified_normal:
        raise HTTPException(422, "Reject and explicitly verify normal activity before adding it")
    result = translate(lambda: normality.add_references(service, profile_id, [{"video_id": event["video_id"],
        "t_start": event["t_start"], "t_end": event["t_end"], "verified_normal": True,
        "split": request.split, "review_event_id": event_id}]))
    return {"version": result["version"], "reference_count": len(result["references"])}

@router.get("/videos")
def videos():
    return get_service().public_videos()

@router.post("/videos/register")
def register(request: RegisterRequest):
    return translate(lambda: get_service().register(request.path, request.video_id, request.captions, request.name))

@router.post("/videos/upload")
def upload(file: UploadFile):
    from app.core.config import settings
    from app.api.v1.upload import run_progressive_pipeline
    from app.gemma2.media import windows
    import threading

    service = get_service()
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4", ".mkv", ".mov", ".avi", ".webm"):
        raise HTTPException(422, "Unsupported video format")

    video_id = str(uuid.uuid4())
    saved_filename = f"{video_id}{suffix}"
    saved_path = settings.RAW_VIDEOS_DIR / saved_filename
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    with open(saved_path, "wb") as stream:
        shutil.copyfileobj(file.file, stream)

    filename = Path(file.filename or "video").name
    record = service.register(saved_path, video_id=video_id, name=filename)

    # Launch background progressive pipeline once; it owns LanceDB/Gemma2 index writes
    # so we don't insert duplicate `videos` rows or mark `phase1_ready` prematurely.
    worker_thread = threading.Thread(
        target=run_progressive_pipeline,
        args=(video_id, str(saved_path.resolve()), filename),
        daemon=True
    )
    worker_thread.start()

    total_windows = max(
        1,
        len(windows(record.get("duration", 0.0), service.config.window_seconds, service.config.stride_seconds)),
    )
    return {
        "id": video_id,
        "name": filename,
        "duration": record.get("duration", 0.0),
        "has_audio": record.get("has_audio", False),
        "caption_count": 0,
        "index_status": "indexing",
        "completed_windows": 0,
        "total_windows": total_windows,
    }

@router.post("/jobs", status_code=202)
def start_job(request: JobRequest):
    return translate(lambda: get_service().submit(request.kind, request.video_id, request.event_types, request.resume_id, request.mode, request.profile_id))

@router.get("/jobs")
def jobs():
    return get_service().store.list("jobs")

@router.get("/jobs/{job_id}")
def progress(job_id: str):
    return translate(lambda: get_service().store.get("jobs", job_id))

@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    return translate(lambda: get_service().cancel(job_id))

@router.get("/list/{video_id}")
def events(video_id: str, include_history: bool = False):
    service = get_service()
    video = translate(lambda: service.store.get("videos", video_id))
    rows = [row for row in service.store.list("events", video_id) if row.get("kind") != "shouting"]
    if not include_history:
        rows = [row for row in rows if row["namespace"] == service.namespace(video)]
        # A new run is a separate observation; retain past model results and
        # human reviews in history, including when the new run finds zero hits.
        latest = {}
        for job in sorted(service.store.list("jobs"), key=lambda item: item.get("created_at", 0)):
            if (job.get("video_id") == video_id and job.get("namespace") == service.namespace(video)
                    and job.get("screening_policy")):
                for kind in job.get("event_types", []):
                    latest[kind] = job["id"]
        rows = [row for row in rows if row["kind"] not in latest
                or row.get("screening_job_id") == latest[row["kind"]]]
    for row in rows:
        row["preview_frame_path"] = f"/api/v1/events/{row['id']}/preview"
    return sorted(rows, key=lambda row: (row["t_start"], row["kind"]))

@router.get("/{event_id}/preview")
def event_preview(event_id: str):
    service = get_service()
    event = translate(lambda: service.store.get("events", event_id))
    frames = event.get("verification_frames") or []
    if frames and Path(frames[0]["path"]).exists():
        return FileResponse(frames[0]["path"], media_type="image/jpeg")
    preview_path = str(event.get("preview_frame_path", ""))
    if preview_path and Path(preview_path).exists() and Path(preview_path).is_file():
        return FileResponse(preview_path, media_type="image/jpeg")
    window_ids = event.get("window_ids") or []
    if window_ids and event.get("video_id"):
        try:
            video = service.store.get("videos", event["video_id"])
            rows = service.store.require(event["video_id"], service.namespace(video))
            row = next((r for r in rows if r["id"] == window_ids[0]), None)
            if row and row.get("frame_paths"):
                mid_frame = Path(row["frame_paths"][len(row["frame_paths"]) // 2])
                if mid_frame.exists():
                    return FileResponse(mid_frame, media_type="image/jpeg")
        except Exception:
            pass
    try:
        from app.db.connection import db_manager
        tbl = db_manager.get_table("video_frames_v2")
        vid_id = event.get("video_id")
        t_start = float(event.get("t_start", 0))
        t_end = float(event.get("t_end", t_start + 5))
        mid = (t_start + t_end) / 2
        rows = [r for r in tbl.to_arrow().to_pylist() if r.get("video_id") == vid_id]
        if rows:
            best = min(rows, key=lambda r: abs(float(r.get("timestamp", 0)) - mid))
            best_path = Path(best.get("frame_path", ""))
            if best_path.exists():
                return FileResponse(best_path, media_type="image/jpeg")
    except Exception:
        pass
    raise HTTPException(404, "Preview image unavailable")

@router.get("/{event_id}/frames/{slot}")
def verification_frame(event_id: str, slot: int):
    event = translate(lambda: get_service().store.get("events", event_id))
    frames = event.get("verification_frames", [])
    if slot < 0 or slot >= len(frames):
        raise HTTPException(404, "Verification frame unavailable")
    frame_path = Path(frames[slot]["path"])
    if not frame_path.exists():
        raise HTTPException(404, "Verification frame file missing")
    return FileResponse(frame_path, media_type="image/jpeg")

@router.post("/{event_id}/review")
def review(event_id: str, request: ReviewRequest):
    return translate(lambda: get_service().store.review(event_id, request.decision, request.note))

@router.get("/videos/{video_id}/stream")
def stream(video_id: str, request: Request):
    from app.api.v1.video import range_requests_response
    service = get_service()
    video_path = None
    try:
        record = service.store.get("videos", video_id)
        video_path = record.get("path")
    except Exception:
        pass
    if not video_path or not os.path.exists(str(video_path)):
        try:
            from app.db.connection import db_manager
            tbl = db_manager.get_table("videos")
            rows = [r for r in tbl.to_arrow().to_pylist() if str(r.get("id")) == video_id]
            if rows:
                video_path = rows[0]["filepath"]
        except Exception:
            pass
    if not video_path or not os.path.exists(str(video_path)):
        raise HTTPException(404, "Video file not found")
    return range_requests_response(request, video_path)

@router.get("/videos/{video_id}/frames/{window_id}")
def frame(video_id: str, window_id: int):
    service = get_service()
    video = translate(lambda: service.store.get("videos", video_id))
    rows = translate(lambda: service.store.require(video_id, service.namespace(video)))
    row = next((row for row in rows if row["id"] == window_id), None)
    if row is None:
        raise HTTPException(404, "Frame unavailable")
    return FileResponse(row["frame_paths"][len(row["frame_paths"])//2])

@router.get("/{event_id}/audio")
def audio(event_id: str):
    service = get_service()
    event = translate(lambda: service.store.get("events", event_id))
    video = translate(lambda: service.store.get("videos", event["video_id"]))
    if not video["has_audio"]:
        raise HTTPException(404, "No audio evidence")
    if not Path(video["path"]).exists():
        raise HTTPException(404, "Video file not found")
    cache_key = hashlib.sha256(f"{event_id}:{video['fingerprint']}:{event['context_t_start']}:{event['context_t_end']}".encode()).hexdigest()
    path = service.store.root / "event-audio" / (cache_key + ".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(event["context_t_start"]),
            "-i", video["path"], "-t", str(event["context_t_end"]-event["context_t_start"]), "-vn", "-ac", "1", "-ar", "16000", str(path)],
            check=True, capture_output=True)
    return FileResponse(path, media_type="audio/wav")

@router.get("/health")
def health():
    service = get_service()
    try:
        response = requests.get(service.client.url + "/health", timeout=3)
        response.raise_for_status()
        worker = response.json()
    except requests.RequestException as exc:
        worker = {"status": "unavailable", "error": str(exc)}
    return {"backend": "gemma2", "config": service.config.json(), "worker": worker,
            "threshold_status": service.thresholds()[2], "screening": "experimental"}
