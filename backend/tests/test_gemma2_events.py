import json
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.gemma2.config import IndexConfig
from app.gemma2.media import windows
from app.gemma2.ranking import merge_candidates, rank_channels, rrf, screen
from app.gemma2.service import EventService
from app.gemma2.service import verification_frames
from app.gemma2.store import ReindexRequired, Store, vectors

def unit(dim=768, slot=0):
    array = np.zeros((1, dim), dtype=np.float32)
    array[0, slot] = 1
    return array

def test_partial_overlapping_tail():
    assert windows(7) == [(0, 6), (3, 7), (6, 7)]
    assert windows(.3) == [(0, .3)]
    assert windows(6) == [(0, 6), (3, 6)]

@pytest.mark.parametrize("values", [[[np.nan]*768], [[0]*768], [[1]*512]])
def test_invalid_vectors_refuse_ranking(values):
    with pytest.raises(ValueError):
        vectors(values, 768)

def test_dimension_and_vision_budgets_are_identity():
    config = IndexConfig()
    keys = {config.namespace("a"), replace(config, dimension=512).namespace("a"),
            config.namespace("b"), config.namespace("a", "captions"), replace(config, video_tokens=280).namespace("a")}
    assert len(keys) == 5
    with pytest.raises(ValueError):
        replace(config, video_tokens=1120)

def test_activation_only_complete_and_atomic(tmp_path):
    store = Store(tmp_path)
    store.begin("v", "old", {}, 1)
    store.save_window("old", {"id": 0}, {"image": unit()}, 768)
    store.activate("v", "old")
    store.begin("v", "new", {}, 2)
    store.save_window("new", {"id": 0}, {"video": unit()}, 768)
    with pytest.raises(ReindexRequired):
        store.activate("v", "new")
    assert store.require("v", "old") == [{"id": 0}]
    with pytest.raises(ReindexRequired):
        store.require("v", "new")
    store.save_window("new", {"id": 1}, {"video": unit()}, 768)
    store.activate("v", "new")
    with pytest.raises(ReindexRequired):
        store.require("v", "old")

def test_failed_window_is_not_partially_written(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        store.save_window("x", {"id": 0}, {"image": unit(), "audio": [[float("nan")]*768]}, 768)
    assert store.windows("x") == []
    assert store.embeddings("x", 768) == {}

def test_rrf_group_images_and_do_not_treat_similarity_as_probability():
    ranks, evidence = rank_channels(unit()[0], {"image": (np.array([0, 0, 1]), np.concatenate([unit(), unit(), unit(slot=1)])),
                                              "video": (np.array([0, 1]), np.concatenate([unit(slot=1), unit()]))})
    order, values = rrf(ranks)
    assert order == [0, 1]
    assert values[0] == pytest.approx(1/61 + 1/62)
    assert evidence[0]["image"] == 1

def test_shouting_needs_audio_but_falling_does_not():
    prompts = {"shouting": (unit(), unit(slot=1)), "falling": (unit(), unit(slot=1))}
    channels = {"video": (np.array([0]), unit())}
    assert [item["kind"] for item in screen(channels, prompts, ["shouting", "falling"])] == ["falling"]

def test_hard_negative_filters_candidate():
    prompts = {"falling": (unit(), unit())}
    assert screen({"video": (np.array([0]), unit())}, prompts, ["falling"]) == []

def test_merge_deduplicates_same_type_only():
    timeline = [{"id": 0, "t_start": 0, "t_end": 6}, {"id": 1, "t_start": 9, "t_end": 12}, {"id": 2, "t_start": 16, "t_end": 20}]
    candidates = [{"kind": "falling", "window_id": i, "evidence": {}} for i in range(3)]
    merged = merge_candidates(candidates, timeline)
    assert [(row["t_start"], row["t_end"]) for row in merged] == [(0, 12), (16, 20)]

def test_bounded_candidates_preserve_local_motion_and_missing_intervals():
    timeline = [{"id": i, "t_start": i*3, "t_end": i*3+6} for i in range(90)]
    candidates = [{"kind": "pushing_punching", "window_id": i, "evidence": {"video": {"similarity": .7}}}
                  for i in range(90) if i not in (20, 21, 22)]
    merged = merge_candidates(candidates, timeline, gap=0, max_seconds=6)
    assert len(merged) == 87
    assert all(row["t_end"]-row["t_start"] <= 6 for row in merged)
    assert {i for row in merged for i in row["window_ids"]} == {row["window_id"] for row in candidates}
    assert not any(row["t_start"] < 66 < row["t_end"] for row in merged)

def test_oversized_window_cannot_silently_break_verification_budget():
    with pytest.raises(ValueError, match="duration budget"):
        merge_candidates([{"kind": "falling", "window_id": 0, "evidence": {}}],
                         [{"id": 0, "t_start": 0, "t_end": 60}], max_seconds=6)

def test_review_survives_retry_and_is_separate_from_model(tmp_path):
    store = Store(tmp_path)
    item = {"id": "e", "video_id": "v", "model_status": "unverified", "review": "pending"}
    store.save_event(item)
    store.review("e", "confirmed", "watched clip")
    store.save_event({**item, "review": "pending", "model_status": "verified"})
    assert store.get("events", "e")["review"] == "confirmed"
    assert store.get("events", "e")["model_status"] == "verified"

class FakeClient:
    last_telemetry = {"allocated_bytes": 0}
    def embed(self, kind, config, **inputs):
        return np.repeat(unit(config.dimension), len(inputs.get("texts", inputs.get("frame_paths", [0]))) if kind in ("image", "query", "caption") else 1, axis=0)
    def post(self, path, payload, **kwargs):
        raise RuntimeError("Qwen offline")

def service_with_index(tmp_path, audio=False):
    store = Store(tmp_path)
    config = IndexConfig()
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixture")
    video = {"id": "v", "path": str(source), "source_stat": [source.stat().st_size, source.stat().st_mtime_ns],
             "duration": 6., "fingerprint": "sha", "captions_hash": "c", "captions": [], "has_audio": audio}
    store.put("videos", "v", video)
    namespace = config.namespace("sha", "c")
    store.begin("v", namespace, config.json(), 1)
    window = {"id": 0, "t_start": 0., "t_end": 6., "captions": [{"text": "Full original caption"}],
              "frame_paths": ["a", "b"], "timestamps": [0., 3.], "audio_path": "sound.wav" if audio else None,
              "audio_status": "available" if audio else "no_audio", "frame_count": 2}
    store.save_window(namespace, window, {"image": unit(), "video": unit()}, 768)
    store.activate("v", namespace)
    return EventService(store, FakeClient(), config), namespace

def test_search_preserves_full_caption_and_no_audio(tmp_path):
    service, _ = service_with_index(tmp_path)
    result = service.search("v", "ภาษาไทย")
    assert result["backend"] == "gemma2" and not result["calibrated"]
    assert result["moments"][0]["caption_full"] == "Full original caption"
    assert "no_audio_evidence" in result["warnings"]
    assert service.search("v", "ภาษาไทย")["cache_hits"]["query"]

def test_accurate_failure_keeps_fast_results(tmp_path):
    service, _ = service_with_index(tmp_path, audio=True)
    def fail(*args, **kwargs):
        raise TimeoutError("budget")
    service.client.embed = fail
    service.query_cache[(service.config.revision, 768, "fall")] = unit()[0]
    result = service.search("v", "fall", profile="accurate")
    assert result["moments"] and result["stage_status"]["joint"] == "incomplete"

def test_source_replacement_requires_reindex(tmp_path):
    service, _ = service_with_index(tmp_path)
    (tmp_path / "video.mp4").write_bytes(b"different video")
    with pytest.raises(ReindexRequired):
        service.search("v", "hello")

def test_restart_marks_interrupted_without_losing_windows(tmp_path):
    service, namespace = service_with_index(tmp_path)
    service.store.put("jobs", "j", {"id": "j", "status": "running", "video_id": "v"})
    recovered = EventService(service.store, FakeClient())
    assert recovered.store.get("jobs", "j")["status"] == "interrupted"
    assert len(recovered.store.windows(namespace)) == 1

def test_qwen_failure_retains_event_as_unverified(tmp_path, monkeypatch):
    service, _ = service_with_index(tmp_path)
    service.store.put("jobs", "j", {"id": "j", "kind": "screen", "video_id": "v", "status": "running", "cancel_requested": False})
    monkeypatch.setattr(service, "prompt_vectors", lambda kinds: {"falling": (unit(), unit(slot=1))})
    monkeypatch.setattr(service, "refine", lambda *args: args[0])
    service.screen_video("j", "v", ["falling"])
    events = service.store.list("events", "v")
    assert len(events) == 1 and events[0]["model_status"] == "unverified" and events[0]["review"] == "pending"

def test_cancel_does_not_delete_evidence(tmp_path):
    service, namespace = service_with_index(tmp_path)
    service.store.put("jobs", "j", {"id": "j", "status": "running", "cancel_requested": False})
    service.cancel("j")
    assert service.store.get("jobs", "j")["cancel_requested"]
    assert len(service.store.windows(namespace)) == 1

def test_short_tail_uses_last_real_frame_with_explicit_evidence(tmp_path):
    import cv2
    from app.gemma2.media import extract
    path = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    for index in range(8):
        writer.write(np.full((48, 64, 3), index*20, dtype=np.uint8))
    writer.release()
    result = extract(path, 1.59, 1.6, tmp_path / "frames", IndexConfig(), False)
    assert result["frame_count"] == 1 and result["held_last_frame"] == [True]
    assert result["timestamps"][0] < result["requested_timestamps"][0]

def test_silent_audio_cannot_screen_as_shouting(tmp_path, monkeypatch):
    service, namespace = service_with_index(tmp_path, audio=True)
    window = service.store.windows(namespace)[0]
    window["audio_status"] = "silent"
    service.store.save_window(namespace, window, {"audio": unit()}, 768)
    service.store.put("jobs", "j", {"id": "j", "status": "running", "cancel_requested": False})
    monkeypatch.setattr(service, "prompt_vectors", lambda kinds: {"shouting": (unit(), unit(slot=1))})
    service.screen_video("j", "v", ["shouting"])
    assert service.store.list("events", "v") == []

def test_qwen_budget_covers_unique_instants_across_overlapping_windows():
    lookup = {0: {"timestamps": [0, 1, 2, 3], "frame_paths": ["a", "b", "c", "d"]},
              1: {"timestamps": [2, 3, 4, 5], "frame_paths": ["c-copy", "d-copy", "e", "f"]}}
    frames = verification_frames([0, 1], lookup, 0, 6)
    assert [stamp for stamp, _ in frames] == [0, 1, 2, 3, 4, 5]
    assert frames[-1][1] == "f"

def test_job_telemetry_does_not_inherit_previous_qwen_samples(tmp_path, monkeypatch):
    service, _ = service_with_index(tmp_path)
    service.client.last_telemetry = {"qwen_resources": {"samples": 99}}
    service.store.put("jobs", "j", {"id": "j", "kind": "screen", "event_types": ["falling"],
                                  "video_id": "v", "status": "queued", "cancel_requested": False})
    monkeypatch.setattr(service, "prompt_vectors", lambda kinds: {"falling": (unit(), unit())})
    service.run_job("j")
    job = service.store.get("jobs", "j")
    assert job["status"] == "complete" and job["qwen_calls"] == 0
    assert "qwen_resources" not in job["telemetry"]

def test_qwen_resources_aggregate_across_batches_instead_of_last_batch(tmp_path, monkeypatch):
    service, namespace = service_with_index(tmp_path)
    base = service.store.windows(namespace)[0]
    for i in range(1, 12):
        service.store.save_window(namespace, {**base, "id": i, "t_start": i*3, "t_end": i*3+6,
                                            "timestamps": [i*3, i*3+3]}, {"video": unit()}, 768)
    monkeypatch.setattr(service, "prompt_vectors", lambda kinds: {"falling": (unit(), unit(slot=1))})
    batches = []
    def post(path, payload, **kwargs):
        batches.append(payload)
        peak = 100 if len(batches) == 1 else 50
        return {"results": [{"id": e["id"], "status": "verified", "calls": 1,
                              "verdict": {"event_present": False}} for e in payload["events"]],
                "telemetry": {"qwen_resources": {"sampled_whole_gpu_peak_bytes": peak,
                              "qwen_process_tree_peak_rss_bytes": peak*2, "samples": 3}, "verification_batch_seconds": 2}}
    monkeypatch.setattr(service.client, "post", post)
    service.store.put("jobs", "j", {"id": "j", "kind": "screen", "event_types": ["falling"],
                                  "video_id": "v", "status": "queued", "cancel_requested": False})
    service.run_job("j")
    job = service.store.get("jobs", "j")
    assert job["status"] == "complete" and job["qwen_calls"] == 12
    assert [len(batch["events"]) for batch in batches] == [8, 4]
    assert job["telemetry"]["qwen_resources"] == {"sampled_whole_gpu_peak_bytes": 100,
                      "qwen_process_tree_peak_rss_bytes": 200, "samples": 6, "batches": 2}
    assert job["telemetry"]["verification_batch_seconds"] == 4
