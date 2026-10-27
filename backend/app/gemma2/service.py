import hashlib
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import requests

from .config import DATA, WORKER_URL, configured_index
from .media import extract, has_audio, metadata, windows
from .ranking import EVENTS, merge_candidates, rank_channels, rrf, screen
from .store import ReindexRequired, Store, file_hash, vectors

SCREENING_POLICY = "bounded-six-second-v1"
VERIFICATION_SECONDS = 6.0

class Cancelled(Exception):
    pass

def verification_frames(window_ids, lookup, start, end):
    # Overlapping windows save separate JPEG paths for the same captured instant.
    # Deduplicate by timestamp before spending Qwen's eight-frame budget.
    unique = {round(stamp, 6): path for window_id in window_ids for stamp, path in
              zip(lookup[window_id]["timestamps"], lookup[window_id]["frame_paths"]) if start <= stamp < end}
    frames = sorted(unique.items())
    slots = np.linspace(0, len(frames)-1, min(8, len(frames))).astype(int)
    return [frames[slot] for slot in slots]

class WorkerClient:
    def __init__(self, url=WORKER_URL):
        self.url = url
        self.last_telemetry = {}
        self.last_inputs = {}

    def post(self, path, payload, timeout=240):
        response = requests.post(self.url + path, json=payload, timeout=max(0.1, timeout))
        response.raise_for_status()
        result = response.json()
        self.last_telemetry = result.get("telemetry", {})
        return result

    def embed(self, kind, config, *, timeout=240, **inputs):
        result = self.post("/embed", {"modality": kind, "dimension": config.dimension,
            "backend": "siglip2" if config.model_id.startswith("google/siglip2") else "gemma2",
            "video_tokens": config.video_tokens, "image_tokens": config.image_tokens,
            "fps": config.video_fps, **inputs}, timeout)
        if result["model_id"] != config.model_id or result["revision"] != config.revision:
            raise ReindexRequired("Worker model revision mismatch")
        checked = vectors(result["embeddings"], config.dimension)
        expected = len(inputs.get("texts", [])) if kind in ("query", "caption") else len(inputs.get("frame_paths", [])) if kind == "image" else 1
        if len(checked) != expected:
            raise ValueError("Worker dropped input records")
        self.last_inputs[kind] = result.get("inputs", [])
        return checked

class EventService:
    def __init__(self, store=None, client=None, config=None):
        self.store = store or Store(DATA)
        self.client = client or WorkerClient()
        self.config = config or configured_index()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="offline-events")
        self.gpu_lock = threading.RLock()
        self.running = set()
        self.query_cache = {}
        self.state_lock = threading.RLock()
        # Do not re-run cancelled/restarted verification silently.
        for job in self.store.list("jobs"):
            if job.get("status") in ("queued", "running", "cancelling"):
                job.update(status="interrupted", error="Process restarted; resume explicitly")
                self.store.put("jobs", job["id"], job)

    def register(self, path, video_id=None, captions=None, name=None):
        path = Path(path).resolve(strict=True)
        info = metadata(path)
        identifier = video_id or str(uuid.uuid4())
        captions = captions or []
        for caption in captions:
            if not 0 <= caption["t_start"] < caption["t_end"] <= info["duration"] + 0.1 or not caption.get("text"):
                raise ValueError("Invalid caption timeline")
        record = {"id": identifier, "name": name or path.name, "path": str(path), **info,
                  "source_stat": [path.stat().st_size, path.stat().st_mtime_ns],
                  "fingerprint": file_hash(path), "captions": captions,
                  "captions_hash": hashlib.sha256(json.dumps(captions, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                  "has_audio": has_audio(path)}
        self.store.put("videos", identifier, record)
        return record

    def namespace(self, video):
        return self.config.namespace(video["fingerprint"], video["captions_hash"])

    def validate_source(self, video):
        # Never reuse embeddings after replacing a video at the same path.
        path = Path(video["path"])
        if [path.stat().st_size, path.stat().st_mtime_ns] != video.get("source_stat"):
            raise ReindexRequired("Video fingerprint changed; register and reindex")

    def public_videos(self):
        result = []
        for video in self.store.list("videos"):
            namespace = self.namespace(video)
            try:
                count = len(self.store.require(video["id"], namespace))
                status = "ready"
            except ReindexRequired:
                count, status = len(self.store.windows(namespace)), "reindex_required"
            result.append({key: value for key, value in video.items() if key not in ("path", "captions")} |
                          {"index_status": status, "completed_windows": count,
                           "total_windows": len(windows(video["duration"], self.config.window_seconds, self.config.stride_seconds)), "caption_count": len(video["captions"])})
        return result

    def submit(self, kind, video_id, event_types=None, resume_id=None, mode="specific", profile_id=None):
        self.store.get("videos", video_id)
        if kind not in ("index", "screen") or (not resume_id and mode != "open_anomaly" and not set(event_types or []) <= EVENTS.keys()):
            raise ValueError("Invalid job type/events")
        if mode not in ("specific", "open_anomaly"):
            raise ValueError("Unknown screening mode")
        snapshot = None
        if mode == "open_anomaly" and kind == "screen" and not resume_id:
            from .open_screening import snapshot_policy
            snapshot = snapshot_policy(self, profile_id)
        with self.state_lock:
            if video_id in self.running:
                raise ValueError("A job is already running for this video")
            if resume_id:
                job = self.store.get("jobs", resume_id)
                if job["video_id"] != video_id or job["kind"] != kind or job["status"] not in ("failed", "cancelled", "interrupted"):
                    raise ValueError("This job cannot be resumed")
                job.update(status="queued", cancel_requested=False, error="")
                job.setdefault("attempt_history", []).append({key: job.get(key) for key in ("started_at", "finished_at", "elapsed_seconds", "completed")})
            else:
                job = {"id": str(uuid.uuid4()), "kind": kind, "video_id": video_id, "event_types": event_types or list(EVENTS),
                       "status": "queued", "phase": "queued", "completed": 0, "total": 0,
                       "cancel_requested": False, "error": "", "created_at": time.time(), "telemetry": {}}
            if not resume_id:
                job.update(mode=mode, profile_id=profile_id, open_policy=snapshot)
                if mode == "open_anomaly":
                    job["event_types"] = ["open_anomaly"]
            self.store.put("jobs", job["id"], job)
            self.running.add(video_id)
            self.executor.submit(self.run_job, job["id"])
            return job

    def update(self, job_id, **fields):
        with self.state_lock:
            job = self.store.get("jobs", job_id)
            job.update(fields)
            self.store.put("jobs", job_id, job)
            return job

    def check_cancel(self, job_id):
        if self.store.get("jobs", job_id)["cancel_requested"]:
            raise Cancelled()

    def cancel(self, job_id):
        with self.state_lock:
            job = self.store.get("jobs", job_id)
            if job["status"] not in ("queued", "running", "cancelling"):
                return job
            return self.update(job_id, cancel_requested=True, status="cancelling")

    def run_job(self, job_id):
        started = time.perf_counter()
        # A cached/empty screening job must not inherit Qwen resource samples
        # from the previous video.
        self.client.last_telemetry = {}
        job = self.update(job_id, status="running", started_at=time.time())
        try:
            if job["kind"] == "index":
                self.index(job_id, job["video_id"])
            elif job.get("mode") == "open_anomaly":
                from .open_screening import screen_open
                screen_open(self, job_id)
            else:
                self.screen_video(job_id, job["video_id"], job["event_types"])
            self.check_cancel(job_id)
            self.update(job_id, status="complete", phase="complete")
        except Cancelled:
            self.update(job_id, status="cancelled")
        except Exception as exc:
            self.update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            self.update(job_id, elapsed_seconds=time.perf_counter()-started, telemetry=self.client.last_telemetry,
                        cumulative_elapsed_seconds=sum(row.get("elapsed_seconds") or 0 for row in job.get("attempt_history", [])) + time.perf_counter()-started,
                        finished_at=time.time())
            with self.state_lock:
                self.running.discard(job["video_id"])

    def encode_window(self, window, captions=None, namespace=None):
        frames = window["frame_paths"]
        channels = {}
        def cached_batch(kind, keys, payloads):
            cached = [self.store.cached_vector(namespace, key, self.config.dimension) if namespace else None for key in keys]
            missing = [slot for slot, value in enumerate(cached) if value is None]
            if missing:
                field = "frame_paths" if kind == "image" else "texts"
                encoded = self.client.embed(kind, self.config, **{field: [payloads[slot] for slot in missing]})
                for slot, value in zip(missing, encoded):
                    cached[slot] = value
                    if namespace:
                        self.store.cache_vector(namespace, keys[slot], value, self.config.dimension)
            return np.stack(cached)
        if "image" in self.config.modalities:
            slots = window["image_slots"]
            channels["image"] = cached_batch("image", [f"image:{window['timestamps'][slot]:.6f}" for slot in slots], [frames[i] for i in slots])
        if "video" in self.config.modalities:
            channels["video"] = self.client.embed("video", self.config, frame_paths=frames)
        if "audio" in self.config.modalities and window["audio_path"]:
            channels["audio"] = self.client.embed("audio", self.config, audio_path=window["audio_path"])
        if "caption" in self.config.modalities and captions:
            texts = [row["text"] for row in captions]
            channels["caption"] = cached_batch("caption", ["caption:" + hashlib.sha256(text.encode()).hexdigest() for text in texts], texts)
        window["input_verification"] = getattr(self.client, "last_inputs", {}).copy()
        return channels

    def index(self, job_id, video_id):
        video = self.store.get("videos", video_id)
        self.validate_source(video)
        namespace = self.namespace(video)
        timeline = windows(video["duration"], self.config.window_seconds, self.config.stride_seconds)
        self.store.begin(video_id, namespace, self.config.json(), len(timeline))
        finished = {row["id"] for row in self.store.windows(namespace)}
        self.update(job_id, phase="index", namespace=namespace, total=len(timeline), completed=len(finished))
        previous = self.store.get("jobs", job_id)
        decode_seconds, encode_seconds = previous.get("decode_seconds", 0.), previous.get("encode_seconds", 0.)
        for identifier, (start, end) in enumerate(timeline):
            self.check_cancel(job_id)
            if identifier in finished:
                continue
            before = time.perf_counter()
            folder = self.store.root / "media" / namespace / str(identifier)
            window = extract(video["path"], start, end, folder, self.config, video["has_audio"])
            window["id"] = identifier
            window["captions"] = [row for row in video["captions"] if row["t_start"] < end and row["t_end"] > start]
            decode_seconds += time.perf_counter()-before
            before = time.perf_counter()
            with self.gpu_lock:
                channels = self.encode_window(window, window["captions"], namespace)
            encode_seconds += time.perf_counter()-before
            self.check_cancel(job_id)
            self.store.save_window(namespace, window, channels, self.config.dimension)
            finished.add(identifier)
            self.update(job_id, completed=len(finished), telemetry=self.client.last_telemetry,
                        decode_seconds=decode_seconds, encode_seconds=encode_seconds)
        self.validate_source(video)
        self.check_cancel(job_id)
        self.store.activate(video_id, namespace)

    def query_vector(self, query, timeout=240):
        key = (self.config.revision, self.config.dimension, query)
        if key in self.query_cache:
            return self.query_cache[key], True
        vector = self.client.embed("query", self.config, texts=[query], timeout=timeout)[0]
        if len(self.query_cache) >= 256:
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[key] = vector
        return vector, False

    @contextmanager
    def gpu_session(self, deadline=None):
        if deadline is None:
            acquired = self.gpu_lock.acquire()
        else:
            acquired = self.gpu_lock.acquire(timeout=max(0., deadline-time.perf_counter()))
        if not acquired:
            raise TimeoutError("GPU queue exceeded Accurate budget")
        try:
            yield
        finally:
            self.gpu_lock.release()

    def search(self, video_id, query, top_k=5, profile="fast", budget_seconds=60):
        try:
            return self._search(video_id, query, top_k, profile, budget_seconds)
        except (TimeoutError, requests.Timeout):
            video = self.store.get("videos", video_id)
            return {"query": query, "video_id": video_id, "moments": [], "timeline_heatmap": [],
                "total_duration": video["duration"], "latency_ms": budget_seconds*1000, "top_k": top_k,
                "profile": profile, "calibrated": False, "backend": "gemma2", "index_version": self.namespace(video),
                "warnings": ["accurate_incomplete_budget", "query_not_completed"], "stage_status": {"query": "incomplete"}}

    def _search(self, video_id, query, top_k=5, profile="fast", budget_seconds=60):
        started = time.perf_counter()
        video = self.store.get("videos", video_id)
        self.validate_source(video)
        namespace = self.namespace(video)
        timeline = self.store.require(video_id, namespace)
        lookup = {row["id"]: row for row in timeline}
        warnings = ["uncalibrated_similarity"]
        if not video["has_audio"]:
            warnings.append("no_audio_evidence")
        # Fast uses only stored vectors; Accurate is bounded by a wall-clock deadline.
        deadline = started + budget_seconds
        cached_query = (self.config.revision, self.config.dimension, query) in self.query_cache
        session = nullcontext() if profile == "fast" and cached_query else self.gpu_session(deadline if profile == "accurate" else None)
        with session:
            query_vector, cache_hit = self.query_vector(query, max(0.1, deadline-time.perf_counter()) if profile == "accurate" else 240)
            channels = self.store.embeddings(namespace, self.config.dimension)
            channels.pop("joint", None)
            rankings, evidence = rank_channels(query_vector, channels)
            order, fused = rrf(rankings)
            checked = 0
            if profile == "accurate":
                cached = self.store.embeddings(namespace, self.config.dimension).get("joint")
                joint = {} if cached is None else dict(zip(cached[0].tolist(), cached[1]))
                # Only the ten candidates share an AV rerank; do not give them extra RRF
                # votes against candidates that have not been checked.
                candidates = order[:10]
                for key in candidates:
                    row = lookup[key]
                    remaining = deadline-time.perf_counter()
                    if remaining <= 0:
                        warnings.append("accurate_incomplete_budget")
                        break
                    if row["audio_path"]:
                        try:
                            if key not in joint:
                                value = self.client.embed("joint", self.config, frame_paths=row["frame_paths"],
                                    audio_path=row["audio_path"], timeout=remaining)
                                self.store.save_joint(namespace, key, value, self.config.dimension)
                                joint[key] = value[0]
                            evidence[key]["joint"] = float(joint[key] @ query_vector)
                        except Exception as exc:
                            warnings.append("accurate_incomplete:" + type(exc).__name__)
                            break
                        checked += 1
                expected = sum(bool(lookup[key]["audio_path"]) for key in candidates)
                complete = checked == expected and not any("incomplete" in warning for warning in warnings)
                if complete and all(key in joint for key in candidates):
                    order = sorted(candidates, key=lambda key: (-evidence[key]["joint"], -fused[key], key)) + order[10:]
            else:
                complete = True
            moments = []
            for occurrence, key in enumerate(order[:top_k], 1):
                row = lookup[key]
                caption = "\n\n".join(item["text"] for item in row["captions"])
                moments.append({"t_start": row["t_start"], "t_end": row["t_end"], "score": fused[key],
                    "raw_score": fused[key], "display_score": fused[key], "occurrence_index": occurrence,
                    "preview_frame_path": f"/api/v1/events/videos/{video_id}/frames/{key}",
                    "caption_preview": caption[:220] or None, "caption_full": caption or None,
                    "context_t_start": max(0, row["t_start"]-5), "context_t_end": min(video["duration"], row["t_end"]+5),
                    "modality_breakdown": evidence[key], "backend": "siglip2" if self.config.model_id.startswith("google/siglip2") else "gemma2", "modalities": list(evidence[key]),
                    "evidence": {"audio_status": row["audio_status"], "timestamps": row["timestamps"],
                                 "frame_count": row["frame_count"], "window_id": key}})
            heatmap = [0.0] * int(np.ceil(video["duration"]))
            for row in timeline:
                for second in range(int(row["t_start"]), min(len(heatmap), int(np.ceil(row["t_end"])))):
                    heatmap[second] = max(heatmap[second], fused.get(row["id"], 0.0))
            peak = max(heatmap, default=0)
            heatmap = [value/peak if peak else 0 for value in heatmap]
        return {"query": query, "video_id": video_id, "moments": moments, "timeline_heatmap": heatmap,
            "total_duration": video["duration"], "latency_ms": (time.perf_counter()-started)*1000, "top_k": top_k,
            "profile": profile, "calibrated": False, "index_version": namespace, "backend": "siglip2" if self.config.model_id.startswith("google/siglip2") else "gemma2",
            "model_revision": self.config.revision, "dimension": self.config.dimension,
            "warnings": warnings, "models_used": [self.config.model_id], "strategy_used": "rrf_multimodal",
            "cache_hits": {"query": cache_hit}, "stage_status": {"joint": "not_applicable_no_audio" if profile == "accurate" and not video["has_audio"] else "complete" if complete else "incomplete"},
            "caption_status": "ready" if video["captions"] else "unavailable", "accurate_checked": checked}

    def prompt_vectors(self, kinds):
        prompts = {}
        for kind in kinds:
            spec = EVENTS[kind]
            positive = np.array([self.query_vector(text)[0] for text in spec["positive"]])
            negative = np.array([self.query_vector(text)[0] for text in spec["negative"]])
            prompts[kind] = positive, negative
        return prompts

    def thresholds(self):
        # Unvalidated defaults are deliberately marked experimental in every event.
        path = self.store.root / "validation-lock.json"
        if path.exists():
            lock = json.loads(path.read_text())
            if lock.get("config") != json.loads(json.dumps(self.config.json())) or not lock.get("validation_passed"):
                raise ValueError("Validation lock does not match configuration")
            return lock["threshold"], lock["margin"], "validation_locked"
        return 0.35, 0.02, "experimental_unvalidated"

    def refine(self, event, video, prompts, threshold, margin, job_id):
        rows, hits = [], []
        local = replace(self.config, window_seconds=3.0, stride_seconds=1.5)
        root = self.store.root / "refinement" / event["id"]
        for slot, (relative_start, relative_end) in enumerate(windows(event["t_end"]-event["t_start"], 3.0, 1.5)):
            self.check_cancel(job_id)
            start, end = event["t_start"]+relative_start, event["t_start"]+relative_end
            row = extract(video["path"], start, end, root / str(slot), local, video["has_audio"])
            row["id"] = slot
            rows.append(row)
            modalities = EVENTS[event["kind"]]["channels"]
            channels = {}
            for modality in modalities:
                if modality == "audio" and row["audio_path"]:
                    value = self.client.embed("audio", self.config, audio_path=row["audio_path"])
                elif modality == "video":
                    value = self.client.embed("video", self.config, frame_paths=row["frame_paths"])
                elif modality == "image":
                    value = self.client.embed("image", self.config, frame_paths=[row["frame_paths"][i] for i in row["image_slots"]])
                else:
                    continue
                channels[modality] = (np.array([slot] * len(value)), value)
            hits.extend(screen(channels, prompts, [event["kind"]], threshold, margin))
        if hits:
            hit_ids = {item["window_id"] for item in hits}
            event["refined_intervals"] = [{"t_start": row["t_start"], "t_end": row["t_end"]} for row in rows if row["id"] in hit_ids]
            event["t_start"] = min(item["t_start"] for item in event["refined_intervals"])
            event["t_end"] = max(item["t_end"] for item in event["refined_intervals"])
            event["refinement_status"] = "complete"
        else:
            event["refinement_status"] = "inconclusive"
        event["context_t_start"] = max(0, event["t_start"]-5)
        event["context_t_end"] = min(video["duration"], event["t_end"]+5)
        return event

    def screen_video(self, job_id, video_id, kinds):
        video = self.store.get("videos", video_id)
        self.validate_source(video)
        namespace = self.namespace(video)
        timeline = self.store.require(video_id, namespace)
        self.update(job_id, phase="screen", total=len(timeline), completed=0)
        threshold, margin, calibration = self.thresholds()
        with self.gpu_lock:
            prompts = self.prompt_vectors(kinds)
        channels = self.store.embeddings(namespace, self.config.dimension)
        candidates = screen(channels, prompts, kinds, threshold, margin)
        acoustic = {row["id"] for row in timeline if row["audio_status"] == "available"}
        candidates = [row for row in candidates if row["kind"] != "shouting" or row["window_id"] in acoustic]
        # Each decision sees local motion. Do not bridge missing candidates or
        # spread eight frames over minutes of unrelated activity.
        merged = merge_candidates(candidates, timeline, gap=0, max_seconds=VERIFICATION_SECONDS)
        self.update(job_id, screening_policy=SCREENING_POLICY, event_types=kinds,
                    namespace=namespace, verification_max_seconds=VERIFICATION_SECONDS)
        pending = []
        lookup = {row["id"]: row for row in timeline}
        # Persist candidates BEFORE any refinement/Qwen so failures retain evidence.
        for item in merged:
            key = hashlib.sha256(f"{namespace}:{job_id}:{item['kind']}:{item['window_ids']}".encode()).hexdigest()[:24]
            event = {**item, "id": key, "video_id": video_id, "namespace": namespace, "label": EVENTS[item["kind"]]["label"],
                     "model_status": "pending", "review": "pending", "audio_verified": False,
                     "screening_policy": SCREENING_POLICY, "screening_job_id": job_id,
                     "threshold_status": calibration, "threshold": threshold, "margin": margin,
                     "audio_status": "available" if video["has_audio"] else "no_audio", "caption_full": "",
                     "context_t_start": max(0, item["t_start"]-5), "context_t_end": min(video["duration"], item["t_end"]+5),
                     "preview_frame_path": f"/api/v1/events/videos/{video_id}/frames/{item['window_ids'][0]}",
                     "refinement_status": "pending", "screening_t_start": item["t_start"], "screening_t_end": item["t_end"]}
            self.store.save_event(event)
            pending.append(event)
        self.update(job_id, candidates=len(candidates), merged_events=len(pending), completed=len(timeline), total=len(timeline))
        # Local Gemma refinement does not validate an action. Keep the complete
        # short window for Qwen, including the before/after transition.
        for event in pending:
            event["refinement_status"] = "bounded_window"
            self.store.save_event(event)
        baseline_calls = sum(1 + (kind != "shouting") for row in timeline for kind in kinds
                             if kind != "shouting" or row["audio_status"] == "available")
        self.update(job_id, phase="verify", completed=0, total=len(pending), qwen_calls=0, qwen_calls_unknown=0, qwen_baseline_calls=baseline_calls)
        observed_calls, unknown_calls = 0, 0
        verification_telemetry = {"qwen_resources": {"samples": 0, "batches": 0}, "verification_batch_seconds": 0.0}
        for offset in range(0, len(pending), 8):
            self.check_cancel(job_id)
            batch = pending[offset:offset+8]
            payload = []
            for event in batch:
                # Choose across the entire event, not all frames from its first window.
                selected = verification_frames(event["window_ids"], lookup, event["t_start"], event["t_end"])
                event["verification_frames"] = [{"timestamp": stamp, "path": path} for stamp, path in selected]
                self.store.save_event(event)
                payload.append({"id": event["id"], "kind": event["kind"], "label": event["label"],
                                "timestamps": [pair[0] for pair in selected], "frame_paths": [pair[1] for pair in selected]})
            try:
                with self.gpu_lock:
                    response = self.client.post("/verify-batch", {"events": payload}, timeout=8*480+30)
                results = response["results"]
                measured = response.get("telemetry", {})
                resources = measured.get("qwen_resources", {})
                aggregate = verification_telemetry["qwen_resources"]
                for field in ("sampled_whole_gpu_peak_bytes", "qwen_process_tree_peak_rss_bytes"):
                    aggregate[field] = max(aggregate.get(field) or 0, resources.get(field) or 0) or None
                aggregate["samples"] += resources.get("samples", 0)
                aggregate["batches"] += 1
                for field in ("interval_seconds", "scope"):
                    if field in resources:
                        aggregate[field] = resources[field]
                verification_telemetry["verification_batch_seconds"] += measured.get("verification_batch_seconds", 0)
                self.client.last_telemetry = {**measured, **verification_telemetry}
                mapping = {row["id"]: row for row in results}
            except Exception as exc:
                mapping = {event["id"]: {"status": "unverified", "error": str(exc)} for event in batch}
            for event in batch:
                result = mapping.get(event["id"], {"status": "unverified", "error": "Missing worker result"})
                event["model_status"] = result["status"]
                event["verification"] = result
                caption = result.get("caption", {})
                event["caption_full"] = caption.get("raw_text") or caption.get("text") or (json.dumps(caption, ensure_ascii=False, indent=2) if caption else "")
                self.store.save_event(event)
                if "calls" in result:
                    observed_calls += result["calls"]
                elif result["status"] in ("verified", "visual_context_only"):
                    observed_calls += 1 + (event["kind"] != "shouting")
                else:
                    unknown_calls += 1
            # A caption and a verifier are separate model calls; report both honestly.
            logical = min(offset+len(batch), len(pending))
            self.update(job_id, completed=logical, qwen_calls=observed_calls, qwen_calls_unknown=unknown_calls,
                        qwen_candidate_windows=logical, qwen_baseline_calls=baseline_calls,
                        telemetry=self.client.last_telemetry)

_service = None
def get_service():
    global _service
    if _service is None:
        _service = EventService()
    return _service
