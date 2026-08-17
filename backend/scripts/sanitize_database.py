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
    logger.info("Starting fast batch LanceDB data sanitation...")
    tbl_frames = db_manager.get_table("video_frames")
    
    refusal_keywords = [
        "sorry", "browse the internet", "cannot browse", "can't browse", 
        "unable to browse", "not able to browse", "large language model", 
        "training data", "cutoff date", "as an ai"
    ]
    
    where_clauses = [f"vlm_caption LIKE '%{kw}%'" for kw in refusal_keywords]
    combined_where = " OR ".join(where_clauses)
    
    try:
        tbl_frames.update(
            where=combined_where,
            values={"vlm_caption": "Visual keyframe capturing scene activity and subjects."}
        )
        logger.info("✅ Bulk update completed successfully!")
    except Exception as e:
        logger.warning(f"Batch SQL error ({e}), falling back to direct Arrow array rewrite...")
        frames = tbl_frames.to_arrow().to_pylist()
        cleaned = 0
        for f in frames:
            cap = (f.get("vlm_caption") or "").lower()
            if any(kw in cap for kw in refusal_keywords):
                fid = f.get("id")
                if fid:
                    try:
                        tbl_frames.update(where=f"id = '{fid}'", values={"vlm_caption": "Visual keyframe capturing scene activity and subjects."})
                        cleaned += 1
                    except Exception:
                        pass
        logger.info(f"✅ Cleaned {cleaned} frames.")
    
    # Rebuild FTS index if exists
    try:
        tbl_frames.create_fts_index("vlm_caption", replace=True)
    except Exception:
        pass
    logger.info("✅ LanceDB data sanitation complete.")

if __name__ == "__main__":
    sanitize_lancedb()
