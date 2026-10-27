"""API integration checks using a deterministic encoder, without loading weights."""
import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from .prepare import load_config
from .service import ExperimentStore, create_app


@pytest.fixture
def client(tmp_path):
    config = load_config()
    config["device"] = "cpu"
    (tmp_path / "evaluation/datasets").mkdir(parents=True)
    (tmp_path / config["dataset"]).write_text(json.dumps({"videos": []}), encoding="utf-8")
    store = ExperimentStore(root=tmp_path, config=config)

    class FakeEncoder:
        def encode_images(self, images):
            vectors = np.zeros((len(images), config["embedding_dim"]), dtype=np.float32)
            vectors[:, 0] = 1
            return vectors

        def encode_queries(self, queries):
            return self.encode_images(queries)

        def encode_clip(self, frames, audio, **kwargs):
            assert len(frames) <= 6
            return self.encode_images([frames])

        def close(self):
            pass

    def load_encoder(role):
        store.encoder = FakeEncoder()
        store.encoder_role = role
        return 0.0

    store._load_encoder = load_encoder
    with TestClient(create_app(store)) as api:
        yield api, store, tmp_path


def upload_video(api, tmp_path):
    path = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    for i in range(50):
        writer.write(np.full((48, 64, 3), i * 4, dtype=np.uint8))
    writer.release()
    with path.open("rb") as media:
        response = api.post("/videos/upload", files={"file": ("../test.avi", media, "video/x-msvideo")})
    assert response.status_code == 200
    return response.json()


def test_upload_search_stream_and_frame_boundaries(client):
    api, store, tmp_path = client
    video = upload_video(api, tmp_path)
    video_id = video["id"]
    assert "path" not in video
    assert store.video(video_id)["path"].startswith(str(store.work / "uploads"))
    response = api.get(f"/videos/{video_id}/stream", headers={"Range": "bytes=0-31"})
    assert response.status_code == 206
    assert len(response.content) == 32
    assert api.get(f"/videos/{video_id}/frames/0").status_code == 404
    submitted = api.post("/search", json={"video_id": video_id, "query": " คนเดิน ", "mode": "both"})
    assert submitted.status_code == 202
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = api.get(f"/jobs/{submitted.json()['job_id']}").json()
        if job["status"] in {"done", "error"}:
            break
        time.sleep(0.02)
    assert job["status"] == "done", job
    assert job["progress"] == 100
    assert job["result"]["query"] == "คนเดิน"
    assert set(job["result"]["roles"]) == {"baseline", "candidate"}
    assert job["result"]["frame_count"] == 10
    assert api.get(f"/videos/{video_id}/frames/0").headers["content-type"] == "image/png"
    assert api.get(f"/videos/{video_id}/frames/-1").status_code == 404
    assert api.get(f"/videos/{video_id}/frames/10").status_code == 404
    assert api.get("/videos").json()[0]["indexed_models"] == ["baseline", "candidate"]


def test_invalid_inputs_and_queue_limit(client):
    api, store, tmp_path = client
    assert api.post("/videos/upload", files={"file": ("bad.txt", b"invalid")}).status_code == 415
    assert api.post("/videos/upload", files={"file": ("bad.mp4", b"invalid")}).status_code == 422
    assert api.get("/videos/missing/stream").status_code == 404
    assert api.get("/jobs/missing").status_code == 404
    video = upload_video(api, tmp_path)
    payload = {"video_id": video["id"], "query": " "}
    assert api.post("/search", json=payload).status_code == 422
    assert api.post("/search", json=payload | {"query": "hello", "mode": "unknown"}).status_code == 422
    store.jobs = {str(i): {"status": "queued"} for i in range(8)}
    assert api.post("/search", json=payload | {"query": "hello"}).status_code == 429


def test_encoder_failure_becomes_visible_job_error(client):
    api, store, tmp_path = client
    video = upload_video(api, tmp_path)
    def fail(role):
        raise RuntimeError("Model unavailable")
    store._load_encoder = fail
    response = api.post("/search", json={"video_id": video["id"], "query": "hello", "mode": "candidate"})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = api.get(f"/jobs/{response.json()['job_id']}").json()
        if job["status"] == "error":
            break
        time.sleep(0.02)
    assert job["status"] == "error"
    assert "Model unavailable" in job["error"]
    assert job["result"] is None


def test_native_video_mode_on_silent_upload(client):
    api, store, tmp_path = client
    video = upload_video(api, tmp_path)
    response = api.post("/search", json={"video_id": video["id"], "query": "hello", "mode": "candidate_av"})
    assert response.status_code == 202
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        job = api.get(f"/jobs/{response.json()['job_id']}").json()
        if job["status"] in {"done", "error"}:
            break
        time.sleep(0.02)
    assert job["status"] == "done", job
    assert job["result"]["has_audio"] is False
    assert job["result"]["window_count"] == 2
    assert set(job["result"]["roles"]) == {"candidate_av"}
    assert len(job["result"]["roles"]["candidate_av"]["moments"]) == 2
