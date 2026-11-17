from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.v1 import events
from test_gemma2_events import service_with_index

def client_for(service, monkeypatch):
    monkeypatch.setattr(events, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(events.router, prefix="/events")
    return TestClient(app)

def test_review_keeps_model_status_and_human_note(tmp_path, monkeypatch):
    service, namespace = service_with_index(tmp_path)
    service.store.save_event({"id": "e", "video_id": "v", "namespace": namespace, "model_status": "unverified", "review": "pending", "kind": "falling", "t_start": 0, "t_end": 3})
    api = client_for(service, monkeypatch)
    assert api.post("/events/e/review", json={"decision": "confirmed", "note": "ตรวจคลิปแล้ว"}).status_code == 200
    row = api.get("/events/list/v").json()[0]
    assert row["model_status"] == "unverified" and row["review"] == "confirmed"
    assert api.post("/events/e/review", json={"decision": "unknown"}).status_code == 422
    assert api.get("/events/jobs/missing").status_code == 404

def test_invalid_job_and_index_mismatch_are_explicit(tmp_path, monkeypatch):
    service, namespace = service_with_index(tmp_path)
    api = client_for(service, monkeypatch)
    assert api.post("/events/jobs", json={"video_id": "v", "kind": "live-camera"}).status_code == 422
    service.store.begin("v", "new", {}, 1)
    service.store.save_window("new", {"id": 0}, {}, 768)
    service.store.activate("v", "new")
    assert api.get("/events/videos/v/frames/0").status_code == 409
    assert api.get("/events/e/frames/0").status_code == 404

def test_new_screening_run_retains_history_and_empty_result_is_current(tmp_path, monkeypatch):
    service, namespace = service_with_index(tmp_path)
    api = client_for(service, monkeypatch)
    service.store.save_event({"id": "old", "video_id": "v", "namespace": namespace,
                             "kind": "falling", "t_start": 0, "t_end": 6, "review": "confirmed"})
    service.store.put("jobs", "new", {"id": "new", "video_id": "v", "namespace": namespace,
                      "event_types": ["falling"], "screening_policy": "bounded-six-second-v1", "created_at": 2})
    assert api.get("/events/list/v").json() == []
    assert api.get("/events/list/v?include_history=true").json()[0]["review"] == "confirmed"
    service.store.save_event({"id": "pending", "video_id": "v", "namespace": namespace, "kind": "falling",
                             "t_start": 0, "t_end": 6, "screening_job_id": "new", "model_status": "pending"})
    # Updating an older run later must not make it the newest run.
    service.store.put("jobs", "older", {"id": "older", "video_id": "v", "namespace": namespace,
                      "event_types": ["falling"], "screening_policy": "old", "created_at": 1, "status": "complete"})
    service.store.put("jobs", "new", {**service.store.get("jobs", "new"), "status": "interrupted"})
    assert [row["id"] for row in api.get("/events/list/v").json()] == ["pending"]


def test_events_upload_does_not_mark_ready_before_pipeline(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.api.v1 import upload as upload_api
    from app.gemma2.service import EventService
    from app.gemma2.store import Store

    service = EventService(Store(tmp_path / "gemma"))
    monkeypatch.setattr(settings, "RAW_VIDEOS_DIR", tmp_path / "raw")
    monkeypatch.setattr(service, "register", lambda path, video_id=None, name=None: {
        "id": video_id, "name": name, "path": str(path), "duration": 12.0, "fps": 25.0, "has_audio": False
    })
    launched = []
    monkeypatch.setattr(upload_api, "run_progressive_pipeline", lambda vid, path, name: launched.append((vid, path, name)))
    api = client_for(service, monkeypatch)

    resp = api.post("/events/videos/upload", files={"file": ("clip.mp4", b"fake-bytes", "video/mp4")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["index_status"] == "indexing"
    assert body["completed_windows"] == 0
    assert len(launched) == 1 and launched[0][0] == body["id"]


def test_native_fallback_reports_unverified_and_preserves_job_history(tmp_path, monkeypatch):
    from app.gemma2.service import EventService
    from app.gemma2.store import Store
    from app.retrieval import search_engine as se_mod

    store = Store(tmp_path / "gemma")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixture")
    store.put("videos", "v", {
        "id": "v", "name": "video.mp4", "path": str(source),
        "source_stat": [source.stat().st_size, source.stat().st_mtime_ns],
        "duration": 10.0, "fingerprint": "sha", "captions_hash": "c", "captions": [], "has_audio": False,
    })
    service = EventService(store)

    fake_moment = SimpleNamespace(
        t_start=1.0, t_end=4.0, score=0.55, raw_score=0.55, display_score=0.55,
        caption_full="Person falling", caption_preview="Person falling",
        context_t_start=0.0, context_t_end=7.0, modality_breakdown={"visual": 0.55, "caption": 0.2},
    )
    monkeypatch.setattr(
        se_mod.search_engine,
        "search_moments",
        lambda **kwargs: SimpleNamespace(moments=[fake_moment], warnings=[]),
    )
    api = client_for(service, monkeypatch)

    store.put("jobs", "j1", {"id": "j1", "kind": "screen", "video_id": "v", "event_types": ["falling"], "status": "queued", "cancel_requested": False, "created_at": 1})
    service.run_job("j1")
    store.put("jobs", "j2", {"id": "j2", "kind": "screen", "video_id": "v", "event_types": ["falling"], "status": "queued", "cancel_requested": False, "created_at": 2})
    service.run_job("j2")

    j2 = store.get("jobs", "j2")
    assert j2["status"] == "complete" and j2["screening_policy"] == "siglip2-native-fallback"

    current = api.get("/events/list/v").json()
    history = api.get("/events/list/v?include_history=true").json()
    assert len(current) == 1 and len(history) == 2
    assert current[0]["model_status"] == "unverified"
    assert current[0]["threshold_status"] == "experimental_unvalidated"
    assert current[0]["screening_policy"] == "siglip2-native-fallback"


def test_index_job_failure_marks_failed_not_complete(tmp_path, monkeypatch):
    service, _ = service_with_index(tmp_path)
    monkeypatch.setattr(service, "index", lambda job_id, video_id: (_ for _ in ()).throw(RuntimeError("worker offline")))
    service.store.put("jobs", "j-idx", {"id": "j-idx", "kind": "index", "video_id": "v", "status": "queued", "cancel_requested": False})
    service.run_job("j-idx")
    job = service.store.get("jobs", "j-idx")
    assert job["status"] == "failed" and "worker offline" in job["error"]
