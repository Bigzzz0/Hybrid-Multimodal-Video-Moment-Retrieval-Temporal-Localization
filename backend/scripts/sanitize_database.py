"""
Database Sanitation Utility:
Cleans up any AI refusal / canned responses (e.g. "I'm sorry, but I am not able to browse...")
stored in LanceDB video_frames and scenes tables.
"""
import os
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.connection import db_manager
from app.core.logger import logger

def sanitize_lancedb():
    logger.info("Starting comprehensive LanceDB data sanitation (English + Chinese)...")
    tbl_frames = db_manager.get_table("video_frames")
    
    refusal_keywords = [
        # English
        "sorry", "browse the internet", "cannot browse", "can't browse", 
        "unable to browse", "not able to browse", "large language model", 
        "training data", "cutoff date", "as an ai", "i am an ai",
        # Chinese (MiniCPM-V defaults)
        "对不起", "抱歉", "语言模型", "无法访问", "没有访问", "作为ai",
        "作为一个ai", "你好", "提供帮助", "误解了", "javascript", "const numbers"
    ]
    
    frames = tbl_frames.to_arrow().to_pylist()
    cleaned = 0
    for f in frames:
        cap = f.get("vlm_caption") or ""
        cap_low = cap.lower()
        if any(kw in cap_low or kw in cap for kw in refusal_keywords):
            fid = f.get("id")
            if fid:
                try:
                    tbl_frames.update(
                        where=f"id = '{fid}'", 
                        values={"vlm_caption": "Visual keyframe capturing scene activity and subjects."}
                    )
                    cleaned += 1
                except Exception:
                    pass
    logger.info(f"Purged {cleaned} refusal/polluted frames.")
    
    # Rebuild FTS index if exists
    try:
        tbl_frames.create_fts_index("vlm_caption", replace=True)
    except Exception:
        pass
    logger.info("✅ LanceDB data sanitation complete.")

if __name__ == "__main__":
    sanitize_lancedb()
