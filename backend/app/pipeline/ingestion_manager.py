import os
import uuid
import datetime
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Callable, Optional, Dict, Any
from app.core.config import settings
from app.core.logger import logger
from app.db.connection import db_manager
from app.pipeline.video_decoder import GPUVideoDecoder
from app.pipeline.scene_detector import AdaptiveSceneDetector
from app.pipeline.keyframe_filter import SSIMKeyframeFilter, compute_pixel_motion
from app.pipeline.visual_encoder import SigLIP2VisualEncoder
from app.pipeline.dense_captioner import QwenVLDenseCaptioner
from app.retrieval.grounding import grounding_orchestrator
from app.retrieval.vlm_artifacts import vlm_artifact_store
from app.inference.vlm_registry import get_variant
from app.inference.contracts import CaptionRequest, CaptionResponse

class ProgressiveIngestionManager:
    """Orchestrates Progressive Two-Phase Video Ingestion and Visual-Centric Feature Extraction."""

    def __init__(self):
        self.scene_detector = AdaptiveSceneDetector()
        self.keyframe_filter = SSIMKeyframeFilter()
        self.visual_encoder = SigLIP2VisualEncoder()
        self.dense_captioner = QwenVLDenseCaptioner()

    def process_video_phase1(
        self,
        video_id: str,
        video_path: str,
        filename: str,
        progress_callback: Optional[Callable[[str, int, str, str, Dict[str, Any]], None]] = None
    ):
        """
        Phase 1: Fast Ingestion (~45-60s) - Makes video searchable immediately.
        Extracts visual keyframes and raw SigLIP 2 NaFlex embeddings into LanceDB.
        """
        logger.info(f"=== Starting Phase 1 Ingestion for Video ID: {video_id} ({filename}) ===")
        if progress_callback:
            progress_callback(video_id, 10, "Initializing Decord GPU Decoder & Metadata...", "decoding", {})

        # 1. Decode Video Metadata
        decoder = GPUVideoDecoder(video_path)
        fps = decoder.fps
        duration = decoder.duration_sec
        resolution = decoder.resolution
        total_frames = decoder.total_frames
        logger.info(f"Video specs: {duration:.2f}s, {fps:.1f} fps, {resolution}, {total_frames} frames")

        # 2. Scene Detection
        if progress_callback:
            progress_callback(
                video_id, 20,
                f"Detecting Scene Cuts (Adaptive Content Detection)...",
                "scene_detect",
                {"duration_sec": duration, "fps": fps, "resolution": resolution, "sub_percent": 30}
            )
        scenes = self.scene_detector.detect_scenes(video_path)

        if progress_callback:
            progress_callback(
                video_id, 35,
                f"Detected {len(scenes)} scenes. Sampling Keyframes with SSIM...",
                "keyframe_ssim",
                {"scene_count": len(scenes), "sub_percent": 0}
            )

        # 3. Keyframe Sampling & SSIM Filtering
        video_keyframe_dir = settings.KEYFRAMES_DIR / video_id
        video_keyframe_dir.mkdir(parents=True, exist_ok=True)

        scene_records = []
        frame_records = []
        all_sampled_images = []
        all_sampled_meta = []
        total_scenes = max(1, len(scenes))

        for s_idx, (t_start, t_end) in enumerate(scenes):
            scene_id = str(uuid.uuid4())
            scene_duration = t_end - t_start
            
            # Keep scene boundaries and a strict <=1s visual gap. Static scenes
            # may later be compacted, but boundary/transition frames remain.
            sample_count = max(1, int(np.ceil(scene_duration / 1.0)) + 1)
            if scene_duration > 1.5:
                sample_count = max(3, sample_count)
            sample_count = min(settings.MAX_FRAMES_PER_SCENE, sample_count)
            sample_timestamps = np.linspace(float(t_start), float(t_end), sample_count).tolist()
            sample_frame_indices = [int(ts * fps) for ts in sample_timestamps]

            raw_frames = decoder.get_batch_frames(sample_frame_indices)
            filtered_frames, filtered_ts = self.keyframe_filter.filter_keyframes(raw_frames, sample_timestamps)

            # SSIM filtering is allowed to remove redundant frames, but it must
            # never remove the temporal anchors required by the v2 contract.
            # Keep scene start/end for every scene and add the midpoint for
            # scenes longer than 1.5 seconds.  This preserves boundaries even
            # when a scene is visually static.
            mandatory_indices = [0, len(raw_frames) - 1]
            if scene_duration > 1.5:
                mandatory_indices.insert(1, len(raw_frames) // 2)
            kept_timestamps = {round(float(ts), 6) for ts in filtered_ts}
            for mandatory_index in mandatory_indices:
                if not (0 <= mandatory_index < len(raw_frames)):
                    continue
                mandatory_ts = float(sample_timestamps[mandatory_index])
                if round(mandatory_ts, 6) not in kept_timestamps:
                    filtered_frames.append(raw_frames[mandatory_index])
                    filtered_ts.append(mandatory_ts)
                    kept_timestamps.add(round(mandatory_ts, 6))
            ordered = sorted(zip(filtered_ts, filtered_frames), key=lambda pair: pair[0])
            filtered_ts = [pair[0] for pair in ordered]
            filtered_frames = [pair[1] for pair in ordered]

            scene_records.append({
                "id": scene_id,
                "video_id": video_id,
                "scene_index": s_idx,
                "t_start": float(t_start),
                "t_end": float(t_end),
                "keyframe_count": len(filtered_frames),
                "caption": "",
                "caption_status": "unavailable",
                "transition_energy": 0.0,
                "embedding_model": settings.SIGLIP2_MODEL_ID,
                "caption_model": settings.QWEN_VL_MODEL_ID,
            })

            for frame_index, (f_img, f_ts) in enumerate(zip(filtered_frames, filtered_ts)):
                frame_id = str(uuid.uuid4())
                # Include scene/frame indices so adjacent scene endpoints at
                # the same timestamp never overwrite one another on disk.
                frame_filename = f"s{s_idx:04d}_f{frame_index:03d}_{f_ts:.2f}s.jpg"
                frame_save_path = video_keyframe_dir / frame_filename
                f_img.save(str(frame_save_path), "JPEG", quality=85)

                all_sampled_images.append(f_img)
                all_sampled_meta.append({
                    "id": frame_id,
                    "video_id": video_id,
                    "scene_id": scene_id,
                    "timestamp": float(f_ts),
                    "frame_path": str(frame_save_path),
                })

            if progress_callback and (s_idx % max(1, total_scenes // 15) == 0 or s_idx == total_scenes - 1):
                sub_pct = int(((s_idx + 1) / total_scenes) * 100)
                macro_pct = min(65, 35 + int(((s_idx + 1) / total_scenes) * 30))
                progress_callback(
                    video_id,
                    macro_pct,
                    f"Sampling Keyframes: Scene {s_idx + 1}/{total_scenes} ({sub_pct}%) • {len(all_sampled_images)} frames",
                    "keyframe_ssim",
                    {
                        "sub_percent": sub_pct,
                        "scene_idx": s_idx + 1,
                        "total_scenes": total_scenes,
                        "frame_count": len(all_sampled_images)
                    }
                )

        # 4. SigLIP 2 Visual Embedding
        def siglip_sub_progress(macro_pct, msg, stg, details):
            if progress_callback:
                progress_callback(video_id, macro_pct, msg, stg, details)

        if settings.INFERENCE_WORKER_ENABLED:
            try:
                from app.inference.client import inference_client
                from app.inference.contracts import EmbedImagesRequest
                embeddings_response = inference_client.embed_images(EmbedImagesRequest(
                    frame_paths=[str(meta["frame_path"]) for meta in all_sampled_meta],
                    batch_size=16,
                ))
                embeddings = embeddings_response.embeddings
                if progress_callback:
                    progress_callback(video_id, 94, f"SigLIP 2 worker embedded {len(embeddings)}/{len(all_sampled_meta)} frames", "siglip2_embedding", {"sub_percent": 100})
            except Exception as exc:
                logger.warning(f"SigLIP worker embedding fallback to local encoder: {exc}")
                embeddings = self.visual_encoder.encode_images(all_sampled_images, batch_size=16, progress_callback=siglip_sub_progress)
        else:
            embeddings = self.visual_encoder.encode_images(all_sampled_images, batch_size=16, progress_callback=siglip_sub_progress)
        
        # 5. Inter-Frame Pixel Motion Energy.  v2 stores the raw encoder output;
        # temporal context is applied at retrieval time so re-indexing is not
        # required when the proposal profile changes.
        # 5A. Compute Normalized Inter-Frame Pixel Motion
        pixel_motion_values = [0.0]
        for idx in range(1, len(all_sampled_images)):
            p_mot = compute_pixel_motion(all_sampled_images[idx - 1], all_sampled_images[idx])
            pixel_motion_values.append(p_mot)

        if len(pixel_motion_values) > 1:
            p5 = float(np.percentile(pixel_motion_values, 5))
            p95 = float(np.percentile(pixel_motion_values, 95))
            denom = max(1e-6, p95 - p5)
            norm_pixel_motion = [float(np.clip((p - p5) / denom, 0.0, 1.0)) for p in pixel_motion_values]
        else:
            norm_pixel_motion = [0.0] * len(pixel_motion_values)

        for i, meta in enumerate(all_sampled_meta):
            emb_curr = np.array(embeddings[i], dtype=np.float32)
            meta["siglip2_vector"] = (emb_curr / (np.linalg.norm(emb_curr) + 1e-6)).tolist()
            meta["embedding_model"] = settings.SIGLIP2_MODEL_ID
            meta["embedding_version"] = settings.SIGLIP2_EMBEDDING_VERSION
            meta["transition_energy"] = norm_pixel_motion[i]
            frame_records.append(meta)

        # 6. Commit to the isolated visual index v2.
        if progress_callback:
            progress_callback(
                video_id, 92,
                f"Building LanceDB IVF-PQ Vector & Full-Text Indices ({len(frame_records)} frames)...",
                "lancedb_commit",
                {"frame_count": len(frame_records), "sub_percent": 80}
            )

        # Insert Video Metadata
        tbl_videos = db_manager.get_table("videos")
        video_record = {
            "id": video_id,
            "filename": filename,
            "filepath": str(video_path),
            "duration_sec": float(duration),
            "fps": float(fps),
            "resolution": resolution,
            "total_frames": int(total_frames),
            "ingestion_phase": "phase1_ready",
            "visual_index_version": settings.VISUAL_INDEX_VERSION,
            "embedding_model": settings.SIGLIP2_MODEL_ID,
            "created_at": datetime.datetime.now().isoformat()
        }
        try:
            names = set(tbl_videos.schema.names)
            video_record = {key: value for key, value in video_record.items() if key in names}
        except Exception:
            pass
        tbl_videos.add([video_record])

        # Insert Scenes
        if scene_records:
            tbl_scenes = db_manager.get_table("scenes_v2")
            tbl_scenes.add(scene_records)

        # Insert Frames
        if frame_records:
            tbl_frames = db_manager.get_table("video_frames_v2")
            tbl_frames.add(frame_records)
            db_manager.mark_visual_index_v2(video_id)

        # Create Indices
        db_manager.create_indices()

        if progress_callback:
            progress_callback(
                video_id, 100,
                f"Phase 1 Ready: Indexed {len(frame_records)} raw v2 visual keyframes. Instant Search is Active!",
                "complete",
                {"duration_sec": duration, "keyframes": len(frame_records)}
            )
        logger.info(f"Phase 1 Visual Ingestion Complete for {video_id}.")

    def process_video_phase2_background(
        self,
        video_id: str,
        progress_callback: Optional[Callable[[str, int, str, str, Dict[str, Any]], None]] = None
    ):
        """
        Legacy phase 2 path retained for old data, but production uploads use
        ``process_video_primary_captions`` with CapRL Q6 artifacts below.
        """
        logger.info(f"=== Starting Phase 2 Background Captioning for Video ID: {video_id} ===")
        if progress_callback:
            progress_callback(video_id, 10, "Generating Qwen3-VL Dense Visual Captions (Background)...", "dense_visual_caption", {})

        try:
            from collections import defaultdict
            tbl_frames = db_manager.get_table("video_frames_v2")
            
            # Fetch frames for this video natively using PyArrow / LanceDB
            try:
                all_frames = tbl_frames.search().where(f"video_id = '{video_id}'").limit(5000).to_list()
            except Exception:
                all_frames = [r for r in tbl_frames.to_arrow().to_pylist() if r.get("video_id") == video_id]

            if not all_frames:
                return

            # Group frames by scene_id
            scenes_grouped = defaultdict(list)
            for f in all_frames:
                scenes_grouped[f.get("scene_id")].append(f)

            total_scene_groups = max(1, len(scenes_grouped))
            for s_idx, (scene_id, group) in enumerate(scenes_grouped.items()):
                group = sorted(group, key=lambda row: float(row.get("timestamp", 0.0)))
                frame_paths = [g.get("frame_path") for g in group if g.get("frame_path")]
                valid_paths = [str(fp) for fp in frame_paths if os.path.exists(str(fp))]
                images = []
                if settings.INFERENCE_WORKER_ENABLED:
                    timestamps = [float(g.get("timestamp", 0.0)) for g in group if g.get("frame_path") and os.path.exists(str(g.get("frame_path")))]
                    caption = self.dense_captioner.generate_scene_caption_from_paths(
                        valid_paths,
                        timestamps=timestamps,
                        max_frames=settings.QWEN_MAX_FRAMES_PER_CANDIDATE,
                    )
                else:
                    images = [Image.open(fp) for fp in valid_paths]
                    caption = self.dense_captioner.generate_scene_caption(images) if images else ""
                if valid_paths:
                    caption_status = "generated" if caption else getattr(self.dense_captioner, "last_status", "unavailable")
                    try:
                        db_manager.get_table("scenes_v2").update(
                            where=f"id = '{scene_id}'",
                            values={
                                "caption": caption,
                                "caption_status": caption_status,
                                "caption_model": settings.QWEN_VL_MODEL_ID if caption else "",
                            }
                        )
                    except Exception as up_err:
                        logger.debug(f"Update scene caption error: {up_err}")
                    try:
                        analysis_table = db_manager.get_table("scene_analysis_v1")
                        analysis_row = {
                            "scene_id": str(scene_id),
                            "video_id": str(video_id),
                            "caption_text": str(caption or ""),
                            "structured_json": json.dumps(getattr(self.dense_captioner, "last_structured", {}) or {}, ensure_ascii=False),
                            "model_id": settings.QWEN_VL_MODEL_ID,
                            "caption_version": settings.CAPTION_VERSION,
                            "prompt_version": "classroom-caption-v1",
                            "status": caption_status,
                            "created_at": datetime.datetime.now().isoformat(),
                        }
                        existing = analysis_table.search().where(f"scene_id = '{scene_id}'").limit(1).to_list()
                        if existing:
                            analysis_table.update(where=f"scene_id = '{scene_id}'", values=analysis_row)
                        else:
                            analysis_table.add([analysis_row])
                    except Exception as analysis_err:
                        logger.debug(f"Persist scene analysis error: {analysis_err}")
                for image in images:
                    try:
                        image.close()
                    except Exception:
                        pass

                if progress_callback:
                    sub_pct = int(((s_idx + 1) / total_scene_groups) * 100)
                    progress_callback(
                        video_id,
                        sub_pct,
                        f"Generating Qwen3-VL Action Captions: Scene {s_idx + 1}/{total_scene_groups} ({sub_pct}%)",
                        "dense_visual_caption",
                        {
                            "sub_percent": sub_pct,
                            "scene_idx": s_idx + 1,
                            "total_scenes": total_scene_groups
                        }
                    )

            # Update video phase status
            tbl_videos = db_manager.get_table("videos")
            tbl_videos.update(
                where=f"id = '{video_id}'",
                values={"ingestion_phase": "phase2_complete"}
            )
            
            # Refresh the scene-caption FTS index only after captions exist.
            try:
                db_manager.get_table("scenes_v2").create_fts_index("caption", replace=True)
            except Exception as fts_err:
                logger.debug(f"Caption FTS refresh deferred: {fts_err}")

            if progress_callback:
                progress_callback(video_id, 100, "Phase 2 Complete: Deep Action Captions Generated!", "complete", {})
            logger.info(f"Phase 2 Captioning Complete for {video_id}.")

        except Exception as e:
            logger.error(f"Error in Phase 2 background captioning: {e}")

    def process_video_primary_captions(
        self,
        video_id: str,
        progress_callback: Optional[Callable[[str, int, str, str, Dict[str, Any]], None]] = None,
    ):
        """Create the active Q6 scene-caption version after Phase 1.

        CapRL Q6 remains resident for the whole video. A scene-level failure
        stays unresolved for a later resumable retry; it never switches to a
        second VLM silently.
        The old scene caption columns are left untouched until the new version
        has a complete artifact set.
        """
        from collections import defaultdict
        from app.inference.client import inference_client

        primary = get_variant(settings.CAPTION_PRIMARY_BACKEND)
        fallback_backend = str(settings.CAPTION_FALLBACK_BACKEND or "").strip()
        fallback = get_variant(fallback_backend) if fallback_backend else None
        frames_table = db_manager.get_table("video_frames_v2")
        scenes_table = db_manager.get_table("scenes_v2")
        try:
            frames = frames_table.search().where(f"video_id = '{video_id}'").limit(200000).to_list()
            scenes = scenes_table.search().where(f"video_id = '{video_id}'").limit(10000).to_list()
        except Exception:
            frames = [row for row in frames_table.to_arrow().to_pylist() if str(row.get("video_id")) == video_id]
            scenes = [row for row in scenes_table.to_arrow().to_pylist() if str(row.get("video_id")) == video_id]
        scenes = sorted(scenes, key=lambda row: float(row.get("t_start", 0.0)))
        if not scenes:
            return

        previous_metadata = vlm_artifact_store.metadata(video_id, primary.backend) or {}
        active_version = str(previous_metadata.get("active_artifact_version") or "")
        if not active_version and previous_metadata.get("status") == "ready":
            active_version = settings.VLM_ARTIFACT_VERSION
        pending_version = str(previous_metadata.get("pending_artifact_version") or "")
        if pending_version and previous_metadata.get("build_status", previous_metadata.get("status")) in {"running", "error"}:
            job_version = pending_version
        elif active_version:
            # Never overwrite the active scene rows while rebuilding.  The
            # active pointer changes only after every scene in this version
            # has a valid Q6/fallback artifact.
            job_version = f"{settings.VLM_ARTIFACT_VERSION}:{uuid.uuid4().hex[:10]}"
        else:
            job_version = settings.VLM_ARTIFACT_VERSION

        # Ingestion may resume from a metadata row still marked running/error;
        # retrieval uses the default require_ready=True and never exposes this
        # partial set to Fast Search.
        existing = vlm_artifact_store.caption_rows(
            video_id, settings.CAPTION_PRIMARY_BACKEND, require_ready=False, artifact_version=job_version
        )
        existing_ids = {str(row.get("id")) for row in existing}
        started_at = datetime.datetime.now().isoformat()
        vlm_artifact_store.set_metadata(
            video_id, primary, len(scenes), len(existing_ids),
            "ready" if active_version else "running",
            started_at=started_at, active_artifact_version=active_version,
            pending_artifact_version=job_version, build_status="running",
        )
        active_backend = ""
        fallback_count = sum(1 for row in existing if fallback and row.get("vlm_backend") == fallback.backend)

        def choose_frames(scene_id: str):
            group = sorted(
                [row for row in frames if str(row.get("scene_id")) == str(scene_id) and os.path.exists(str(row.get("frame_path", "")))],
                key=lambda row: float(row.get("timestamp", 0.0)),
            )
            if not group:
                return [], []
            count = min(len(group), max(1, int(settings.CAPTION_MAX_FRAMES_PER_SCENE)))
            indexes = np.linspace(0, len(group) - 1, count).round().astype(int).tolist()
            selected = [group[index] for index in indexes]
            return [str(row["frame_path"]) for row in selected], [float(row.get("timestamp", 0.0)) for row in selected]

        def caption_with_backend(backend: str, paths: list[str], timestamps: list[float], max_frames: int):
            request = CaptionRequest(
                scene_id="",
                frame_paths=paths[:max_frames],
                timestamps=timestamps[:max_frames],
                prompt=(
                    "Analyze only visible visual content in these chronological CCTV frames. "
                    "Return exactly one compact JSON object with keys summary, objects, attributes, "
                    "actions, relations, temporal_events, uncertainty. Do not use OCR, subtitles, "
                    "audio, names, identities or face recognition. Keep summary under 20 words, "
                    "each array to at most 4 short items, and use empty arrays when absent. "
                    "Do not include timestamps, long explanations, markdown or extra keys."
                ),
                prompt_version=settings.CAPTION_PROMPT_VERSION,
                max_new_tokens=settings.CAPTION_MAX_NEW_TOKENS,
                vlm_backend=backend,
                release_after=False,
                max_frames=max_frames,
            )
            if settings.INFERENCE_WORKER_ENABLED:
                return inference_client.vlm_caption(request, timeout_sec=240.0)
            raise RuntimeError("CapRL Q6 caption requires the local inference worker")

        try:
            for index, scene in enumerate(scenes):
                scene_id = str(scene.get("id"))
                if scene_id in existing_ids:
                    continue
                paths, timestamps = choose_frames(scene_id)
                if not paths:
                    continue
                response = None
                backend_used = active_backend or primary.backend
                try:
                    response = caption_with_backend(backend_used, paths, timestamps, len(paths))
                    if not getattr(response, "summary", "") and not getattr(response, "text", ""):
                        raise RuntimeError("empty caption response")
                    if backend_used == primary.backend and not bool(getattr(response, "json_valid", False)):
                        raise RuntimeError("primary caption returned invalid JSON")
                except Exception as first_error:
                    logger.warning("Primary caption failed for scene %s: %s", scene_id, first_error)
                    if backend_used == primary.backend and fallback and fallback.backend != primary.backend:
                        try:
                            inference_client.vlm_unload(primary.backend)
                        except Exception:
                            pass
                        active_backend = fallback.backend
                        try:
                            response = caption_with_backend(fallback.backend, paths, timestamps, min(len(paths), settings.CAPTION_RETRY_FRAMES))
                            fallback_count += 1
                        except Exception as fallback_error:
                            logger.error("Fallback caption failed for scene %s: %s", scene_id, fallback_error)
                            response = None
                    else:
                        response = None
                if response is not None:
                    variant_used = get_variant(getattr(response, "vlm_backend", backend_used) or backend_used)
                    vlm_artifact_store.save_caption(
                        video_id, variant_used, "scene", scene_id,
                        float(scene.get("t_start", 0.0)), float(scene.get("t_end", 0.0)), timestamps, response,
                        settings.CAPTION_PROMPT_VERSION, artifact_version=job_version,
                    )
                    existing_ids.add(scene_id)
                if progress_callback:
                    percent = int(((index + 1) / max(1, len(scenes))) * 100)
                    progress_callback(video_id, percent, f"CapRL Q6 captions: Scene {index + 1}/{len(scenes)}", "dense_visual_caption", {"scene_idx": index + 1, "total_scenes": len(scenes), "fallback_count": fallback_count})
            complete = len(existing_ids) == len(scenes)
            status = "ready" if complete else ("ready" if active_version else "error")
            vlm_artifact_store.set_metadata(
                video_id, primary, len(scenes), len(existing_ids), status,
                error_message="" if complete else "one or more scenes failed",
                started_at=started_at,
                completed_at=datetime.datetime.now().isoformat() if complete else "",
                fallback_count=fallback_count,
                active_artifact_version=job_version if complete else active_version,
                pending_artifact_version="" if complete else job_version,
                build_status="ready" if complete else "error",
            )
            if complete:
                try:
                    db_manager.get_table("videos").update(where=f"id = '{video_id}'", values={"ingestion_phase": "caption_ready"})
                except Exception:
                    pass
                if progress_callback:
                    progress_callback(video_id, 100, "Caption artifacts ready: Fast and Accurate search can use them.", "dense_visual_caption", {"caption_backend": primary.backend, "fallback_count": fallback_count})
        except Exception as exc:
            logger.exception("Primary captioning failed for %s", video_id)
            vlm_artifact_store.set_metadata(
                video_id, primary, len(scenes), len(existing_ids),
                "ready" if active_version else "error", error_message=str(exc),
                started_at=started_at, fallback_count=fallback_count,
                active_artifact_version=active_version,
                pending_artifact_version=job_version,
                build_status="error",
            )
        finally:
            for backend in {primary.backend, fallback.backend if fallback else ""}:
                if not backend:
                    continue
                try:
                    if settings.INFERENCE_WORKER_ENABLED:
                        inference_client.vlm_unload(backend)
                except Exception:
                    pass

    def process_video_phase3_background(
        self,
        video_id: str,
        progress_callback: Optional[Callable[[str, int, str, str, Dict[str, Any]], None]] = None,
    ):
        """Phase 3: optional SAM 3.1 generic classroom grounding cache."""
        if progress_callback:
            progress_callback(video_id, 5, "Preparing SAM 3.1 grounding cache...", "sam_grounding", {})
        if not settings.INFERENCE_WORKER_ENABLED:
            if progress_callback:
                progress_callback(video_id, 100, "SAM 3.1 worker disabled; on-demand grounding remains available after worker startup.", "sam_unavailable", {"skipped": True})
            return
        try:
            tbl_frames = db_manager.get_table("video_frames_v2")
            tbl_scenes = db_manager.get_table("scenes_v2")
            try:
                frames = tbl_frames.search().where(f"video_id = '{video_id}'").limit(200000).to_list()
                scenes = tbl_scenes.search().where(f"video_id = '{video_id}'").limit(10000).to_list()
            except Exception:
                frames = [row for row in tbl_frames.to_arrow().to_pylist() if str(row.get("video_id")) == video_id]
                scenes = [row for row in tbl_scenes.to_arrow().to_pylist() if str(row.get("video_id")) == video_id]
            concepts = ["person", "backpack or school bag", "book", "laptop or computer", "mobile phone"]
            total = max(1, len(scenes))
            for index, scene in enumerate(sorted(scenes, key=lambda row: float(row.get("t_start", 0.0)))):
                candidate = [{"t_start": float(scene.get("t_start", 0.0)), "t_end": float(scene.get("t_end", 0.0))}]
                _, warnings, _ = grounding_orchestrator.ground_candidates(
                    video_id,
                    candidate,
                    frames,
                    concepts,
                    limit=1,
                    fps=settings.SAM_PRECOMPUTE_FPS,
                )
                if warnings:
                    logger.warning("SAM grounding warnings for scene {}: {}", scene.get("id"), warnings)
                if progress_callback:
                    percent = int(((index + 1) / total) * 100)
                    progress_callback(video_id, percent, f"SAM 3.1 Grounding: Scene {index + 1}/{total}", "sam_grounding", {"sub_percent": percent, "scene_idx": index + 1, "total_scenes": total})
            try:
                db_manager.get_table("videos").update(where=f"id = '{video_id}'", values={"ingestion_phase": "phase3_complete"})
            except Exception:
                pass
            if progress_callback:
                progress_callback(video_id, 100, "Phase 3 Complete: SAM 3.1 Grounding Cache Ready", "complete", {})
        except Exception as exc:
            logger.error(f"Error in Phase 3 SAM grounding: {exc}")
            if progress_callback:
                progress_callback(video_id, 100, f"SAM grounding unavailable: {exc}", "sam_unavailable", {})

ingestion_manager = ProgressiveIngestionManager()
