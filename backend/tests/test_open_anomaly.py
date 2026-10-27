import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from app.gemma2 import normality as n
from app.gemma2.open_screening import merge_telemetry, screen_open, snapshot_policy
from app.gemma2.service import Cancelled
from app.gemma2.store import ReindexRequired
from evaluation.embeddinggemma2.open_anomaly_eval import audit, metrics
from evaluation.embeddinggemma2.worker import checked_verdict
from test_gemma2_events import service_with_index, unit
from test_gemma2_api import client_for


def reference(service, profile_id, **changes):
    return n.add_references(service, profile_id, [{"video_id": "v", "t_start": 0., "t_end": 6.,
        "split": "development", "verified_normal": True, **changes}])


def test_reference_versions_are_immutable_and_require_verified_development(tmp_path):
    service, _ = service_with_index(tmp_path)
    p = n.create_profile(service.store, "Camera A", "Pedestrian corridor")
    for changes in ({"split": "test"}, {"split": "validation"}, {"verified_normal": False}):
        with pytest.raises(ValueError):
            reference(service, p["id"], **changes)
    first = reference(service, p["id"])
    again = reference(service, p["id"])
    assert first["version"] == again["version"]
    assert n.bank(service.store, p["id"], first["version"])["references"] == first["references"]
    with pytest.raises(ValueError, match="complete indexed"):
        reference(service, p["id"], t_end=2.)
    service.config = replace(service.config, dimension=256)
    with pytest.raises(ReindexRequired):
        snapshot_policy(service, p["id"])


def test_snapshot_stays_missing_when_bank_added_later(tmp_path):
    service, _ = service_with_index(tmp_path)
    p = n.create_profile(service.store, "A", "A corridor")
    frozen = snapshot_policy(service, p["id"])
    reference(service, p["id"])
    assert frozen["reference_version"] is None
    assert snapshot_policy(service, p["id"])["reference_version"] is not None


def test_three_nearest_reference_windows_not_duplicate_slots():
    config = {"video": {"novelty": .5, "margin": 2., "similarity": 1.}}
    refs = [{"key": str(i), "video_id": "v", "window_id": i, "fingerprint": "f", "t_start": 0, "t_end": 6,
             "channels": {"video": np.repeat(unit(768, slot), repetitions, axis=0).tolist()}}
            for i, slot, repetitions in ((0, 0, 30), (1, 1, 1), (2, 2, 1))]
    result = n.score_channels({"video": (np.array([4]), unit())}, {"references": refs}, (unit(slot=3), unit()), config, 768)
    score = result[4]["video"][0]
    assert score["novelty"] == pytest.approx(2/3)
    assert score["selected"] and len(score["references"]) == 3


def test_semantic_fallback_and_nonfinite_vectors():
    thresholds = copy.deepcopy(n.DEFAULT_THRESHOLDS)
    scores = n.score_channels({"video": (np.array([0]), unit())}, None, (unit(), unit(slot=1)), thresholds, 768)
    assert scores[0]["video"][0]["reference_status"] == "reference_missing"
    assert scores[0]["video"][0]["triggers"] == ["semantic_concern"]
    with pytest.raises(ValueError):
        n.score_channels({"video": (np.array([0]), np.full((1, 768), np.nan))}, None, (unit(), unit()), thresholds, 768)


def test_merge_and_split_preserve_tail_and_source_evidence():
    timeline = [{"id": i, "t_start": i*3., "t_end": min(i*3.+6, 25.)} for i in range(9)]
    evidence = {i: {"video": [{"selected": True}]} for i in range(9)}
    result = n.segments(timeline, evidence)
    assert [(r["t_start"], r["t_end"]) for r in result] == [(0, 12), (12, 24), (24, 25)]
    assert all(r["window_ids"] for r in result)


def test_context_schema_unknown_is_not_normal():
    data = {"event_present": False, "assessment": "uncertain", "start_frame_index": 0, "end_frame_index": 2,
            "confidence": .4, "reason": "Occluded", "observations": "Person behind wall", "deviation": "",
            "alternative_explanation": "May be walking", "uncertainty": "Body hidden"}
    reply = checked_verdict({"raw_text": json.dumps(data)}, 12, n.POLICY)
    assert reply["event_present"] is None
    for changes in ({"assessment": "suspicious"}, {"observations": None}, {"confidence": float("nan")}):
        with pytest.raises(ValueError):
            checked_verdict({"raw_text": json.dumps({**data, **changes})}, 12, n.POLICY)


def test_worker_failure_preserves_candidates_and_review(tmp_path, monkeypatch):
    service, _ = service_with_index(tmp_path)
    p = n.create_profile(service.store, "A", "A corridor")
    policy = snapshot_policy(service, p["id"])
    for role in policy["prompts"]:
        for text in policy["prompts"][role]:
            service.query_cache[(service.config.revision, 768, text + " Scene context: " + policy["context"])] = unit(slot=0 if role == "suspicious" else 1)[0]
    service.store.put("jobs", "j", {"id": "j", "video_id": "v", "status": "running", "cancel_requested": False, "open_policy": policy})
    monkeypatch.setattr(n, "sample_frames", lambda *a, **k: [{"timestamp": 0., "path": "a"}, {"timestamp": 3., "path": "b"}])
    screen_open(service, "j")
    event = service.store.list("events", "v")[0]
    assert event["assessment"] == "uncertain" and event["model_status"] == "unverified"
    assert event["reference_status"] == "reference_missing"
    assert service.store.get("jobs", "j")["qwen_calls_unknown"] == 1
    service.store.review(event["id"], "confirmed", "Human checked")
    screen_open(service, "j")
    assert service.store.get("events", event["id"])["review"] == "confirmed"
    service.update("j", cancel_requested=True)
    with pytest.raises(Cancelled):
        screen_open(service, "j")


def test_profile_api_review_is_explicit_and_bank_hides_vectors(tmp_path, monkeypatch):
    service, ns = service_with_index(tmp_path)
    api = client_for(service, monkeypatch)
    p = api.post("/events/profiles", json={"name": "A", "context": "A corridor"}).json()
    service.store.save_event({"id": "e", "video_id": "v", "t_start": 0, "t_end": 6, "review": "pending"})
    url = f"/events/profiles/{p['id']}/normal-events/e"
    assert api.post(url, json={"verified_normal": True}).status_code == 422
    service.store.review("e", "rejected", "Normal walking")
    assert api.post(url, json={"verified_normal": False}).status_code == 422
    assert api.post(url, json={"verified_normal": True}).status_code == 200
    response = api.get(f"/events/profiles/{p['id']}/bank").json()
    assert "channels" not in response["references"][0]


def row(identifier="a", **changes):
    return {"id": identifier, "recording_id": identifier, "dataset": "d", "scene_id": "s", "sha256": identifier,
            "split": "development", "official_split": "training", "source_url": "https://example.org", "license": "research",
            "duration": 20., "events": [], **changes}


def test_dataset_audit_rejects_official_test_and_recording_leakage():
    with pytest.raises(ValueError, match="Official test"):
        audit([row(official_split="test")], check_files=False)
    with pytest.raises(ValueError, match="leakage"):
        audit([row(), row("b", recording_id="a", split="validation")], check_files=False)
    with pytest.raises(ValueError, match="Test access"):
        audit([row(split="test", official_split="test")], check_files=False)


def test_event_matching_unknowns_and_normal_exposure():
    rows = [row(events=[{"t_start": 2, "t_end": 4}, {"t_start": 8, "t_end": 10}]), row("normal", duration=3600.)]
    predictions = {"a": [{"t_start": 2, "t_end": 4, "assessment": "suspicious"},
                          {"t_start": 8, "t_end": 10, "assessment": "uncertain"}],
                   "normal": [{"t_start": 3, "t_end": 5, "assessment": "suspicious"}]}
    result = metrics(rows, predictions)
    assert result["recall"] == .5 and result["precision"] == .5
    assert result["uncertain_units"] == 1 and result["false_alerts_per_hour"] == 1.


def test_open_prompt_keeps_normal_reference_outside_event_chronology(monkeypatch):
    from app.inference.contracts import VerifyRequest
    from inference_worker.vlm_service import VariantVLMService
    service = VariantVLMService()
    seen = []
    monkeypatch.setattr(service, "_generate_frames", lambda paths, prompt, budget, frame_labels:
                        seen.append((paths, frame_labels)) or "{}")
    service._generate(VerifyRequest(query="any anomaly", frame_paths=["a", "b"], timestamps=[1., 2.],
        scene_context="corridor", reference_frames=[{"path": "normal"}], verification_mode=n.POLICY), "prompt", 512)
    paths, labels = seen[0]
    assert paths == ["a", "b", "normal"]
    assert "timestamp 1.000" in labels[0] and "separate recording" in labels[-1]


def test_open_broker_accepts_twelve_frames_and_preserves_uncertainty(monkeypatch):
    from types import SimpleNamespace
    from evaluation.embeddinggemma2 import worker
    data = {"event_present": False, "assessment": "uncertain", "start_frame_index": 0, "end_frame_index": 11,
            "confidence": .4, "reason": "Occluded", "observations": "A person behind a wall", "deviation": "",
            "alternative_explanation": "Walking", "uncertainty": "Motion hidden"}
    requests = []
    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"raw_text": json.dumps(data)})
    monkeypatch.setattr(worker.requests, "post", post)
    monkeypatch.setattr(worker, "unload", lambda: None)
    monkeypatch.setattr(worker, "release_qwen", lambda: None)
    monkeypatch.setattr(worker, "telemetry", lambda: {})
    monkeypatch.setattr(worker, "ResourceSampler", lambda: SimpleNamespace(thread=SimpleNamespace(start=lambda: None), finish=lambda: {}))
    event = {"id": "e", "kind": "open_anomaly", "verification_mode": n.POLICY,
             "frame_paths": [str(i) for i in range(12)], "timestamps": list(range(12)),
             "generate_caption": False, "scene_context": "corridor", "reference_frames": [{"path": "normal"}]}
    result = worker.verify_batch(worker.VerifyBatch(events=[event]))["results"][0]
    assert result["status"] == "unverified" and result["verdict"]["event_present"] is None
    assert requests[0]["scene_context"] == "corridor" and len(requests) == 1


def test_real_frame_sampling_is_chronological_and_bounded(tmp_path):
    import cv2
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10., (64, 48))
    for index in range(200):
        writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
    writer.release()
    frames = n.sample_frames(path, 6., 12., 20., tmp_path / "frames")
    stamps = [f["timestamp"] for f in frames]
    assert len(stamps) == 12 and stamps == sorted(set(stamps))
    assert sum(6 <= t < 12 for t in stamps) == 8
    assert sum(t < 6 for t in stamps) == 2 and sum(t >= 12 for t in stamps) == 2


def test_metric_tail_rounding_tolerates_only_submicrosecond_difference():
    rows = [row(duration=15.366666666667)]
    result = metrics(rows, {"a": [{"t_start": 12., "t_end": 15.366667, "assessment": "suspicious"}]})
    assert result["fp"] == 1
    with pytest.raises(ValueError, match="outside video"):
        metrics(rows, {"a": [{"t_start": 12., "t_end": 15.37, "assessment": "suspicious"}]})


def test_verification_telemetry_keeps_peaks_across_batches_and_retries():
    first = {"qwen_resources": {"samples": 2, "sampled_whole_gpu_peak_bytes": 100, "qwen_process_tree_peak_rss_bytes": 70}, "verification_batch_seconds": 3.}
    second = {"qwen_resources": {"samples": 4, "sampled_whole_gpu_peak_bytes": 90, "qwen_process_tree_peak_rss_bytes": 80}, "verification_batch_seconds": 4.}
    result = merge_telemetry(first, second)
    assert result["qwen_resources"]["samples"] == 6
    assert result["qwen_resources"]["sampled_whole_gpu_peak_bytes"] == 100
    assert result["qwen_resources"]["qwen_process_tree_peak_rss_bytes"] == 80
    assert result["verification_batch_seconds"] == 7.


def test_known_validation_recording_cannot_be_relabelled_development(tmp_path):
    service, _ = service_with_index(tmp_path)
    p = n.create_profile(service.store, "A", "A corridor")
    video = service.store.get("videos", "v")
    video["dataset_split"] = "validation"
    service.store.put("videos", "v", video)
    with pytest.raises(ValueError, match="Known validation/test"):
        reference(service, p["id"])


def test_metrics_require_explicit_prediction_coverage():
    with pytest.raises(ValueError, match="Missing predictions"):
        metrics([row()], {})
    assert metrics([row()], {"a": []})["fp"] == 0


def test_multiscene_selection_is_deterministic_and_keeps_official_splits():
    from evaluation.embeddinggemma2.prepare_ubnormal import select_recordings
    groups = {("normal", "training"): [f"normal_scene_{s}_scenario_{i}" for s in (3, 4) for i in (6, 2, 8)],
              ("normal", "validation"): [f"normal_scene_{s}_scenario_1" for s in (3, 4)],
              ("abnormal", "validation"): [f"abnormal_scene_{s}_scenario_{i}" for s in (3, 4) for i in (5, 3)]}
    chosen = select_recordings(groups, [4, 3, 4], 2)
    assert len(chosen) == 10
    assert [r[0] for r in chosen[:2]] == ["normal_scene_3_scenario_2", "normal_scene_3_scenario_6"]
    assert all(official == ("training" if split == "development" else "validation")
               for _, split, official, _ in chosen)
    with pytest.raises(ValueError, match="lacks independent"):
        select_recordings(groups, [12], 2)
    with pytest.raises(ValueError, match="Positive"):
        select_recordings(groups, [3], 0)


def test_stage_diagnostics_separates_screening_and_verification_losses():
    from evaluation.embeddinggemma2.open_anomaly_eval import stage_diagnostics
    rows = [row(events=[{"t_start": 2., "t_end": 4.}]), row("b", events=[{"t_start": 2., "t_end": 4.}])]
    screened = {"a": [{"t_start": 2., "t_end": 4., "assessment": "suspicious"}], "b": []}
    final = {"a": [{"t_start": 2., "t_end": 4., "assessment": "uncertain"}], "b": []}
    result = stage_diagnostics(rows, screened, final)
    assert result["counts"] == {"retained": 0, "boundary_recovered": 0, "verification_loss": 1, "screening_miss": 1}
    assert result["recordings"][0]["final"]["uncertain_units"] == 1


def test_index_reuse_never_copies_prior_predictions_profiles_or_reviews(tmp_path):
    from app.gemma2.store import Store
    from evaluation.embeddinggemma2.run_open_anomaly import reuse_indexes
    service, _ = service_with_index(tmp_path / "source")
    reference(service, n.create_profile(service.store, "Prior experiment", "corridor")["id"])
    with service.store.db() as db:
        db.execute("INSERT INTO events VALUES ('old','v','{}')")
        db.execute("INSERT INTO reviews(event_id,decision,note) VALUES ('old','rejected','old run')")
    source_hash = file_hash_for_test(service.store.path)
    destination = Store(tmp_path / "destination")
    reuse_indexes(destination, service.store.path, ["v"])
    assert destination.get("videos", "v")["id"] == "v"
    assert destination.windows(service.namespace(service.store.get("videos", "v")))
    with destination.db() as db:
        for table in ("events", "reviews", "jobs", "normality_profiles", "normality_banks", "vector_cache"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert file_hash_for_test(service.store.path) == source_hash


def file_hash_for_test(path):
    from app.gemma2.store import file_hash
    return file_hash(path)
