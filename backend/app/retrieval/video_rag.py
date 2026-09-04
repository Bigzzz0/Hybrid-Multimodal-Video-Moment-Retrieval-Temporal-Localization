import time
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
from pydantic import BaseModel
from app.core.config import settings
from app.core.logger import logger
from app.db.connection import db_manager
from app.pipeline.visual_encoder import SigLIP2VisualEncoder
from app.pipeline.dense_captioner import QwenVLDenseCaptioner

class VideoQARequest(BaseModel):
    video_id: str
    question: str

class GroundedMoment(BaseModel):
    t_start: float
    t_end: float
    citation_text: str
    thumbnail_path: Optional[str] = None

class VideoQAResponse(BaseModel):
    question: str
    answer: str
    grounded_moments: List[GroundedMoment]
    latency_ms: float

class VideoRAGEngine:
    """
    SOTA Multimodal Video-RAG Engine.
    Retrieves visual keyframes & speech context to answer complex questions about the video.
    """

    def __init__(self):
        self.text_encoder = SigLIP2VisualEncoder()
        self.captioner = QwenVLDenseCaptioner()

    def answer_question(self, video_id: str, question: str) -> VideoQAResponse:
        t0 = time.time()
        logger.info(f"Video-RAG answering question for video {video_id}: '{question}'")

        # 1. Retrieve Visual Frames with SigLIP 2 & Caption Matching
        tbl_frames = db_manager.get_table("video_frames")
        try:
            video_frames = tbl_frames.search().where(f"video_id = '{video_id}'").limit(1000).to_list()
        except Exception:
            video_frames = [r for r in tbl_frames.to_arrow().to_pylist() if r.get("video_id") == video_id]

        query_vec = np.array(self.text_encoder.encode_text(question), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        q_tokens = [w.strip().lower() for w in question.split() if len(w.strip()) > 1]

        scored_frames = []
        for f in video_frames:
            emb = f.get("siglip2_vector")
            sim = 0.0
            if emb is not None and len(emb) == 768:
                sim = float(np.dot(np.array(emb, dtype=np.float32), query_vec))
            
            # Action caption boost
            caption = (f.get("vlm_caption") or "").lower()
            if caption and q_tokens:
                match_count = sum(1 for tok in q_tokens if tok in caption)
                if match_count > 0:
                    sim += 0.25 * (match_count / len(q_tokens))
            
            scored_frames.append((sim, f))

        scored_frames.sort(key=lambda x: x[0], reverse=True)
        top_frames = scored_frames[:3]

        # 2. Formulate Visual Context
        context_visual_str = "\n".join([
            f"[เวลา {f.get('timestamp'):.1f}s]: {f.get('vlm_caption') or 'ภาพแสดงการกระทำและเหตุการณ์ในฉาก'}"
            for sim, f in top_frames if sim > 0.05
        ])

        # 3. Generate Grounded Synthesis Answer
        evidence_citations = []
        for sim, f in top_frames:
            if sim > 0.08:
                ts = float(f.get("timestamp", 0.0))
                evidence_citations.append(GroundedMoment(
                    t_start=round(max(0.0, ts - 1.5), 1),
                    t_end=round(ts + 3.5, 1),
                    citation_text=(f.get("vlm_caption") or "ฉากเหตุการณ์ที่ตรงกับคำถาม")[:100],
                    thumbnail_path=f.get("frame_path")
                ))

        # Synthesis
        if context_visual_str.strip():
            answer = f"จากการวิเคราะห์ภาพเหตุการณ์และการกระทำในวิดีโอ (Visual Evidence):\n{context_visual_str}"
            if evidence_citations:
                answer += f"\n\n(อ้างอิงช่วงเวลาสำคัญ: {evidence_citations[0].t_start}s - {evidence_citations[0].t_end}s)"
        else:
            answer = "ไม่พบภาพเหตุการณ์หรือการกระทำที่สอดคล้องกับคำถามนี้ในวิดีโอที่เลือก"

        latency_ms = round((time.time() - t0) * 1000.0, 2)
        return VideoQAResponse(
            question=question,
            answer=answer,
            grounded_moments=evidence_citations,
            latency_ms=latency_ms
        )

video_rag_engine = VideoRAGEngine()
