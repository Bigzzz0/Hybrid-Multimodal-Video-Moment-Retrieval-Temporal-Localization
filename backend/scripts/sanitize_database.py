"""
Database Sanitation Utility:
Cleans up any AI refusal / canned responses (e.g. "I'm sorry, but I am not able to browse...")
 stored in LanceDB scene captions.
"""
import os
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.connection import db_manager
from app.core.logger import logger

def sanitize_lancedb():
    logger.info("Starting comprehensive LanceDB data sanitation (English + Chinese)...")
    tbl_scenes = db_manager.get_table("scenes_v2")
    
    refusal_keywords = [
        # English
        "sorry", "browse the internet", "cannot browse", "can't browse", 
        "unable to browse", "not able to browse", "large language model", 
        "training data", "cutoff date", "as an ai", "i am an ai",
        "对不起", "抱歉", "语言模型", "无法访问", "没有访问", "作为ai",
        "作为一个ai", "你好", "提供帮助", "误解了", "javascript", "const numbers"
    ]
    
    scenes = tbl_scenes.to_arrow().to_pylist()
    cleaned = 0
    for scene in scenes:
        cap = scene.get("caption") or ""
        cap_low = cap.lower()
        if any(kw in cap_low or kw in cap for kw in refusal_keywords):
            sid = scene.get("id")
            if sid:
                try:
                    tbl_scenes.update(
                        where=f"id = '{sid}'",
                        values={"caption": "", "caption_status": "unavailable"}
                    )
                    cleaned += 1
                except Exception:
                    pass
    logger.info(f"Purged {cleaned} refusal/polluted frames.")
    
    # Rebuild FTS index if exists
    try:
        tbl_scenes.create_fts_index("caption", replace=True)
    except Exception:
        pass
    logger.info("✅ LanceDB data sanitation complete.")

if __name__ == "__main__":
    sanitize_lancedb()
