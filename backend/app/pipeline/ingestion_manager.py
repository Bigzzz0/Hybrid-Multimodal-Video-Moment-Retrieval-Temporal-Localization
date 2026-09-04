import os
import uuid
import datetime
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
        Extracts Decord frames, ASR transcripts, and SigLIP 2 visual embeddings into LanceDB.
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
            
            # Sample at 1 fps within scene
            sample_count = max(1, min(settings.MAX_FRAMES_PER_SCENE, int(scene_duration * settings.KEYFRAME_SAMPLE_INTERVAL_SEC)))
            sample_timestamps = [t_start + (i / max(1, sample_count)) * scene_duration for i in range(sample_count)]
            sample_frame_indices = [int(ts * fps) for ts in sample_timestamps]

            raw_frames = decoder.get_batch_frames(sample_frame_indices)
            filtered_frames, filtered_ts = self.keyframe_filter.filter_keyframes(raw_frames, sample_timestamps)

            scene_records.append({
                "id": scene_id,
                "video_id": video_id,
                "scene_index": s_idx,
                "t_start": float(t_start),
                "t_end": float(t_end),
                "keyframe_count": len(filtered_frames)
            })

            for f_img, f_ts in zip(filtered_frames, filtered_ts):
                frame_id = str(uuid.uuid4())
                frame_filename = f"f_{f_ts:.2f}s.jpg"
                frame_save_path = video_keyframe_dir / frame_filename
                f_img.save(str(frame_save_path), "JPEG", quality=85)

                all_sampled_images.append(f_img)
                all_sampled_meta.append({
                    "id": frame_id,
                    "video_id": video_id,
                    "scene_id": scene_id,
                    "timestamp": float(f_ts),
                    "frame_path": str(frame_save_path),
                    "vlm_caption": "",
                    "has_dense_caption": False
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

        embeddings = self.visual_encoder.encode_images(all_sampled_images, batch_size=16, progress_callback=siglip_sub_progress)
        
        # 5. Inter-Frame Pixel Motion Energy & Dual-Scale Temporal Context Hierarchy
        import numpy as np
        num_frames = len(embeddings)

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

        # 5B. Dual-Scale Context Hierarchy Fusion (UniTime NeurIPS 2025)
        for i, meta in enumerate(all_sampled_meta):
            emb_curr = np.array(embeddings[i], dtype=np.float32)

            if getattr(settings, "ENABLE_DUAL_SCALE_CONTEXT", True) and num_frames >= 4:
                # Local Context (W=3)
                emb_prev = np.array(embeddings[i - 1], dtype=np.float32) if i > 0 else emb_curr
                emb_next = np.array(embeddings[i + 1], dtype=np.float32) if i < num_frames - 1 else emb_curr
                v_local = 0.60 * emb_curr + 0.20 * emb_prev + 0.20 * emb_next
                v_local /= (np.linalg.norm(v_local) + 1e-6)

                # Global Context (W=7 with Gaussian decay kernel sigma=2.0)
                win_start = max(0, i - 3)
                win_end = min(num_frames, i + 4)
                tau_indices = list(range(win_start, win_end))
                weights = [np.exp(-((idx - i) ** 2) / (2.0 * (2.0 ** 2))) for idx in tau_indices]
                w_sum = sum(weights)
                v_global = sum((w / w_sum) * np.array(embeddings[idx], dtype=np.float32) for idx, w in zip(tau_indices, weights))
                v_global /= (np.linalg.norm(v_global) + 1e-6)

                # Hierarchical Convex Combination & Spherical Projection
                alpha_l = getattr(settings, "DUAL_SCALE_LOCAL_WEIGHT", 0.65)
                alpha_g = getattr(settings, "DUAL_SCALE_GLOBAL_WEIGHT", 0.35)
                v_hierarchical = alpha_l * v_local + alpha_g * v_global
                v_hierarchical /= (np.linalg.norm(v_hierarchical) + 1e-6)
            else:
                v_hierarchical = emb_curr / (np.linalg.norm(emb_curr) + 1e-6)

            meta["siglip2_vector"] = v_hierarchical.tolist()
            meta["pixel_motion"] = norm_pixel_motion[i]
            frame_records.append(meta)

        # 6. Commit to LanceDB (Phase 1 Ready - Pure Visual SOTA)
        if progress_callback:
            progress_callback(
                video_id, 92,
                f"Building LanceDB IVF-PQ Vector & Full-Text Indices ({len(frame_records)} frames)...",
                "lancedb_commit",
                {"frame_count": len(frame_records), "sub_percent": 80}
            )

        # Insert Video Metadata
        tbl_videos = db_manager.get_table("videos")
        tbl_videos.add([{
            "id": video_id,
            "filename": filename,
            "filepath": str(video_path),
            "duration_sec": float(duration),
            "fps": float(fps),
            "resolution": resolution,
            "total_frames": int(total_frames),
            "ingestion_phase": "phase1_ready",
            "created_at": datetime.datetime.now().isoformat()
        }])

        # Insert Scenes
        if scene_records:
            tbl_scenes = db_manager.get_table("scenes")
            tbl_scenes.add(scene_records)

        # Insert Frames
        if frame_records:
            tbl_frames = db_manager.get_table("video_frames")
            tbl_frames.add(frame_records)

        # Create Indices
        db_manager.create_indices()

        if progress_callback:
            progress_callback(
                video_id, 100,
                f"Phase 1 Ready: Indexed {len(frame_records)} visual keyframes with Temporal Context. Instant Search is Active!",
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
        Phase 2: Deep Context Ingestion (Background) - Generates Qwen2.5-VL-7B Spatiotemporal Action Captions.
        """
        logger.info(f"=== Starting Phase 2 Background Captioning for Video ID: {video_id} ===")
        if progress_callback:
            progress_callback(video_id, 10, "Generating Qwen2.5-VL Dense Action Captions (Background)...", "vlm_caption", {})

        try:
            from collections import defaultdict
            tbl_frames = db_manager.get_table("video_frames")
            
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
                frame_paths = [g.get("frame_path") for g in group if g.get("frame_path")]
                images = [Image.open(fp) for fp in frame_paths if os.path.exists(fp)]
                
                if images:
                    caption = self.dense_captioner.generate_scene_caption(images)
                    for g in group:
                        fid = g.get("id")
                        if fid:
                            try:
                                tbl_frames.update(
                                    where=f"id = '{fid}'",
                                    values={"vlm_caption": caption, "has_dense_caption": True}
                                )
                            except Exception as up_err:
                                logger.debug(f"Update frame caption error: {up_err}")

                if progress_callback:
                    sub_pct = int(((s_idx + 1) / total_scene_groups) * 100)
                    progress_callback(
                        video_id,
                        sub_pct,
                        f"Generating Qwen2.5-VL Action Captions: Scene {s_idx + 1}/{total_scene_groups} ({sub_pct}%)",
                        "vlm_caption",
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
            
            # Refresh FTS Index
            tbl_frames.create_fts_index("vlm_caption", replace=True)

            if progress_callback:
                progress_callback(video_id, 100, "Phase 2 Complete: Deep Action Captions Generated!", "complete", {})
            logger.info(f"Phase 2 Captioning Complete for {video_id}.")

        except Exception as e:
            logger.error(f"Error in Phase 2 background captioning: {e}")

ingestion_manager = ProgressiveIngestionManager()
