"""
Recaptioning Utility:
Runs MiniCPM-V 2.6 across all keyframes in LanceDB to populate genuine, high-quality
dense action & scene captions for all videos in the database.
"""
import os
import sys
from PIL import Image

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.connection import db_manager
from app.pipeline.dense_captioner import MiniCPMDenseCaptioner
from app.core.logger import logger

def recaption_all_frames():
    logger.info("Initializing MiniCPMDenseCaptioner for full database recaptioning...")
    captioner = MiniCPMDenseCaptioner()
    
    tbl_frames = db_manager.get_table("video_frames")
    frames = tbl_frames.to_arrow().to_pylist()
    logger.info(f"Found {len(frames)} frames to process.")
    
    # Group frames by video_id and scene_id
    scenes = {}
    for f in frames:
        key = (f.get("video_id"), f.get("scene_id"))
        if key not in scenes:
            scenes[key] = []
        scenes[key].append(f)
        
    logger.info(f"Processing {len(scenes)} unique scenes...")
    
    updated_count = 0
    for (vid, sid), frame_list in scenes.items():
        # Load keyframe images
        images = []
        valid_frames = []
        for f in frame_list:
            f_path = f.get("frame_path")
            if f_path and os.path.exists(f_path):
                try:
                    img = Image.open(f_path).convert("RGB")
                    images.append(img)
                    valid_frames.append(f)
                except Exception as e:
                    logger.debug(f"Could not load image {f_path}: {e}")
                    
        if not images:
            continue
            
        logger.info(f"Generating caption for Video {vid[:8]}... Scene {sid} ({len(images)} frames)...")
        caption = captioner.generate_scene_caption(images)
        logger.info(f"-> Generated: {caption[:80]}...")
        
        # Update all frames in this scene with the generated caption
        for f in valid_frames:
            fid = f.get("id")
            if fid:
                try:
                    tbl_frames.update(
                        where=f"id = '{fid}'",
                        values={
                            "vlm_caption": caption,
                            "has_dense_caption": True
                        }
                    )
                    updated_count += 1
                except Exception as ex:
                    logger.debug(f"Update error on frame {fid}: {ex}")
                    
    logger.info(f"Successfully recaptioned {updated_count} frames across all scenes!")

if __name__ == "__main__":
    recaption_all_frames()
