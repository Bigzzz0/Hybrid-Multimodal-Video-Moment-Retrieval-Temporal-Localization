"""
Fast Target Video Recaptioning:
Directly generates high-quality MiniCPM-V 2.6 captions for video 408691dd-ea27-4a30-9982-4c44e6b66fdc
(the 14s demo video with computer desks and orange/green polo shirts).
"""
import os
import sys
from PIL import Image

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.connection import db_manager
from app.pipeline.dense_captioner import MiniCPMDenseCaptioner
from app.core.logger import logger

def recaption_demo_video():
    target_vid = "408691dd-ea27-4a30-9982-4c44e6b66fdc"
    logger.info(f"Targeted recaptioning for video: {target_vid}")
    
    captioner = MiniCPMDenseCaptioner()
    tbl_frames = db_manager.get_table("video_frames")
    frames = tbl_frames.to_arrow().to_pylist()
    
    demo_frames = [f for f in frames if f.get("video_id") == target_vid]
    logger.info(f"Found {len(demo_frames)} frames for target video.")
    
    # Group by scene_id
    scenes = {}
    for f in demo_frames:
        sid = f.get("scene_id")
        if sid not in scenes:
            scenes[sid] = []
        scenes[sid].append(f)
        
    for sid, frame_list in scenes.items():
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
                    logger.debug(f"Image load error: {e}")
                    
        if images:
            logger.info(f"Generating caption for Scene {sid} ({len(images)} keyframes)...")
            caption = captioner.generate_scene_caption(images)
            logger.info(f"Caption generated: {caption}")
            
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
                    except Exception as ex:
                        logger.debug(f"Frame update error: {ex}")
                        
    logger.info("Demo video recaptioning complete!")

if __name__ == "__main__":
    recaption_demo_video()
