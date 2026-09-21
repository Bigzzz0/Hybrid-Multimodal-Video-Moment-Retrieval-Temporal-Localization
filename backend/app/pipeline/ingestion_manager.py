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
from app.inference.vlm_registry import all_variants
from app.retrieval.vlm_artifacts import vlm_artifact_store

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
        Phase 2: Deep Context Ingestion (Background) - Generates Qwen3-VL-2B Spatiotemporal Action Captions.
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

    def process_video_phase3_background(
        self,
        video_id: str,
        progress_callback: Optional[Callable[[str, int, str, str, Dict[str, Any]], None]] = None,
    ):
        """Phase 3: optional SAM 3.1 generic classroom grounding cache."""
        if progress_callback:
            progress_callback(video_id, 5, "Preparing SAM 3.1 grounding cache...", "sam_grounding", {})
        if not settings.INFERENCE_WORKER_ENABLED or settings.VLM_LAB_DISABLE_SAM:
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

    def process_video_vlm_ablation_background(
        self,
        video_id: str,
        progress_callback: Optional[Callable[[str, int, str, str, Dict[str, Any]], None]] = None,
        backend_filter: Optional[str] = None,
    ):
        """Build A--E caption artifacts after Fast indexing is available.

        This job is intentionally sequential by backend. The inference worker
        owns the GPU and keeps one selected VLM resident while it processes a
        backend's scenes; a worker restart or a failed variant records an
        error and allows the following variant to continue.
        """
        if not settings.VLM_LAB_ENABLED:
            return
        from app.inference.client import inference_client
        from app.inference.contracts import CaptionRequest
        import datetime as _datetime
        from collections import defaultdict

        try:
            frames_table = db_manager.get_table("video_frames_v2")
            scenes_table = db_manager.get_table("scenes_v2")
            try:
                frames = frames_table.search().where(f"video_id = '{video_id}'").limit(200000).to_list()
                scenes = scenes_table.search().where(f"video_id = '{video_id}'").limit(10000).to_list()
            except Exception:
                frames = [row for row in frames_table.to_arrow().to_pylist() if str(row.get("video_id")) == video_id]
                scenes = [row for row in scenes_table.to_arrow().to_pylist() if str(row.get("video_id")) == video_id]
            if not frames or not scenes:
                logger.warning("VLM ablation skipped for {}: no v2 scenes/frames", video_id)
                return
            try:
                video_rows = db_manager.get_table("videos").search().where(f"id = '{video_id}'").limit(1).to_list()
                source_video_path = str(video_rows[0].get("filepath", "")) if video_rows else ""
            except Exception:
                source_video_path = ""
            by_scene = defaultdict(list)
            for frame in frames:
                by_scene[str(frame.get("scene_id"))].append(frame)
            ordered_scenes = sorted(scenes, key=lambda row: float(row.get("t_start", 0.0)))

            variants = [variant for variant in all_variants() if not backend_filter or variant.backend == backend_filter]
            for variant_index, variant in enumerate(variants):
                sources = []
                if variant.input_mode == "video_chunk":
                    duration = max(float(row.get("t_end", 0.0)) for row in ordered_scenes)
                    cursor = 0.0
                    while cursor < duration:
                        end = min(duration, cursor + settings.VLM_VIDEO_CHUNK_SEC)
                        sources.append(("video_chunk", f"chunk_{cursor:.3f}", cursor, end, [f for f in frames if cursor <= float(f.get("timestamp", 0.0)) <= end]))
                        cursor += settings.VLM_VIDEO_CHUNK_STRIDE_SEC
                else:
                    sources = [
                        ("scene", str(scene.get("id")), float(scene.get("t_start", 0.0)), float(scene.get("t_end", 0.0)), by_scene.get(str(scene.get("id")), []))
                        for scene in ordered_scenes
                    ]
                expected = len(sources)
                started_at = _datetime.datetime.now().isoformat()
                existing = vlm_artifact_store.caption_rows(video_id, variant.backend)
                completed = {str(row.get("id")) for row in existing}
                vlm_artifact_store.set_metadata(video_id, variant, expected, len(completed), "running", started_at=started_at)
                if not inference_client.enabled:
                    vlm_artifact_store.set_metadata(video_id, variant, expected, len(completed), "error", "inference worker is disabled", started_at, _datetime.datetime.now().isoformat())
                    continue
                error_message = ""
                for source_kind, source_id, t_start, t_end, source_frames in sources:
                    if source_id in completed:
                        continue
                    source_frames = sorted(source_frames, key=lambda row: float(row.get("timestamp", 0.0)))
                    cap = settings.VLM_VIDEO_MAX_FRAMES if variant.input_mode == "video_chunk" else settings.VLM_FRAME_MAX_FRAMES
                    if len(source_frames) > cap:
                        positions = np.linspace(0, len(source_frames) - 1, cap).astype(int).tolist()
                        source_frames = [source_frames[index] for index in positions]
                    paths = [str(row.get("frame_path")) for row in source_frames if row.get("frame_path") and os.path.exists(str(row.get("frame_path")))]
                    timestamps = [float(row.get("timestamp", 0.0)) for row in source_frames if row.get("frame_path") and os.path.exists(str(row.get("frame_path")))]
                    if not paths:
                        continue
                    prompt_version = settings.VLM_VIDEO_PROMPT_VERSION if variant.input_mode == "video_chunk" else settings.VLM_CAPTION_PROMPT_VERSION
                    request = CaptionRequest(
                        scene_id=source_id, frame_paths=paths, timestamps=timestamps,
                        prompt_version=prompt_version, max_new_tokens=settings.VLM_CAPTION_MAX_NEW_TOKENS,
                        vlm_backend=variant.backend, release_after=False,
                        video_path=source_video_path if variant.input_mode == "video_chunk" else "", t_start=t_start, t_end=t_end,
                        sample_fps=settings.VLM_VIDEO_SAMPLE_FPS if variant.input_mode == "video_chunk" else 1.0,
                        max_frames=cap,
                        prompt=("Analyze only visible visual content in these chronological frames. Return exactly one compact JSON object with keys summary, objects, attributes, actions, relations, temporal_events, uncertainty. Keep summary under 40 words, each array under 6 items, and do not use markdown, OCR, subtitles, audio, names or identity.")
                    )
                    try:
                        response = inference_client.vlm_caption(request)
                        vlm_artifact_store.save_caption(video_id, variant, source_kind, source_id, t_start, t_end, timestamps, response, prompt_version)
                        completed.add(source_id)
                    except Exception as exc:
                        error_message = str(exc)
                        logger.warning("VLM ablation {} source {} failed: {}", variant.backend, source_id, exc)
                        break
                    if progress_callback:
                        overall = (variant_index + ((len(completed) / max(1, expected)))) / max(1, len(variants)) * 100
                        progress_callback(video_id, int(overall), f"VLM ablation {variant.label}: {len(completed)}/{expected}", f"vlm_ablation_{variant.backend}", {"backend": variant.backend, "completed": len(completed), "expected": expected})
                    vlm_artifact_store.set_metadata(video_id, variant, expected, len(completed), "running", error_message, started_at)
                status = "ready" if len(completed) == expected and not error_message else "error"
                vlm_artifact_store.set_metadata(video_id, variant, expected, len(completed), status, error_message, started_at, _datetime.datetime.now().isoformat() if status == "ready" else "")
                try:
                    inference_client.vlm_unload(variant.backend)
                except Exception:
                    pass
            if progress_callback:
                progress_callback(video_id, 100, "VLM A–E artifacts complete", "vlm_ablation_complete", {})
        except Exception as exc:
            logger.error("Error in VLM ablation background for {}: {}", video_id, exc)

ingestion_manager = ProgressiveIngestionManager()
