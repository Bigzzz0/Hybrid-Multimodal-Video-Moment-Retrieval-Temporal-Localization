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
