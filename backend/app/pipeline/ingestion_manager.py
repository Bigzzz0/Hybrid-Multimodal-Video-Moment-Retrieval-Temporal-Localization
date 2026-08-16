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
from app.pipeline.keyframe_filter import SSIMKeyframeFilter
from app.pipeline.audio_asr import WhisperAudioASR
from app.pipeline.visual_encoder import SigLIP2VisualEncoder
from app.pipeline.dense_captioner import MiniCPMDenseCaptioner

class ProgressiveIngestionManager:
    """Orchestrates Progressive Two-Phase Video Ingestion and Feature Extraction."""

    def __init__(self):
        self.scene_detector = AdaptiveSceneDetector()
        self.keyframe_filter = SSIMKeyframeFilter()
        self.audio_asr = WhisperAudioASR()
        self.visual_encoder = SigLIP2VisualEncoder()
        self.dense_captioner = MiniCPMDenseCaptioner()

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
                video_id, 25,
                f"Detecting Scene Cuts (Adaptive Content Detection)...",
                "scene_detect",
                {"duration_sec": duration, "fps": fps, "resolution": resolution}
            )
        scenes = self.scene_detector.detect_scenes(video_path)

        # 3. Audio ASR Transcription (Concurrent/Sequential)
        if progress_callback:
            progress_callback(
                video_id, 45,
                "Transcribing Speech with Whisper-Large-v3-Turbo (CUDA FP16)...",
                "asr_whisper",
                {"scene_count": len(scenes)}
            )
        transcripts = self.audio_asr.transcribe(video_path)

        # 4. Keyframe Sampling & SSIM Filtering
        if progress_callback:
            progress_callback(
                video_id, 60,
                "Extracting & Filtering Keyframes with SSIM...",
                "keyframe_ssim",
                {"transcript_count": len(transcripts)}
            )
        
        video_keyframe_dir = settings.KEYFRAMES_DIR / video_id
        video_keyframe_dir.mkdir(parents=True, exist_ok=True)

        scene_records = []
        frame_records = []
        all_sampled_images = []
        all_sampled_meta = []

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

        # 5. SigLIP 2 Visual Embedding
        if progress_callback:
            progress_callback(
                video_id, 80,
                f"Generating SigLIP 2 Embeddings for {len(all_sampled_images)} keyframes...",
                "siglip2_embedding",
                {"keyframe_count": len(all_sampled_images)}
            )
        
        embeddings = self.visual_encoder.encode_images(all_sampled_images, batch_size=16)
        
        for meta, emb in zip(all_sampled_meta, embeddings):
            meta["siglip2_vector"] = emb
            frame_records.append(meta)

        # 6. Commit to LanceDB (Phase 1 Ready)
        if progress_callback:
            progress_callback(
                video_id, 95,
                "Building LanceDB IVF-PQ Vector & Full-Text Indices...",
                "lancedb_commit",
                {}
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

        # Insert Transcripts
        if transcripts:
            tbl_transcripts = db_manager.get_table("transcripts")
            transcript_rows = [{
                "id": str(uuid.uuid4()),
                "video_id": video_id,
                "t_start": float(tr["t_start"]),
                "t_end": float(tr["t_end"]),
                "speaker_tag": "speaker",
                "spoken_text": tr["spoken_text"]
            } for tr in transcripts]
            tbl_transcripts.add(transcript_rows)

        # Insert Frames
        if frame_records:
            tbl_frames = db_manager.get_table("video_frames")
            tbl_frames.add(frame_records)

        # Create Indices
        db_manager.create_indices()

        if progress_callback:
            progress_callback(
                video_id, 100,
                f"Phase 1 Ready: Indexed {len(frame_records)} frames & {len(transcripts)} transcripts. Instant Search is Active!",
                "complete",
                {"duration_sec": duration, "keyframes": len(frame_records), "transcripts": len(transcripts)}
            )
        logger.info(f"Phase 1 Ingestion Complete for {video_id}.")

    def process_video_phase2_background(
        self,
        video_id: str,
        progress_callback: Optional[Callable[[str, int, str, str, Dict[str, Any]], None]] = None
    ):
        """
        Phase 2: Deep Context Ingestion (Background) - Generates MiniCPM-V 2.6 Action Captions.
        """
        logger.info(f"=== Starting Phase 2 Background Captioning for Video ID: {video_id} ===")
        if progress_callback:
            progress_callback(video_id, 10, "Generating MiniCPM-V 2.6 Action Captions (Background)...", "minicpmv_caption", {})

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

            for scene_id, group in scenes_grouped.items():
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
