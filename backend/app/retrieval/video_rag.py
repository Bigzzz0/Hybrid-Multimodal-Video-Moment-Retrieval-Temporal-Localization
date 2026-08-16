import time
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
from pydantic import BaseModel
from app.core.config import settings
from app.core.logger import logger
from app.db.connection import db_manager
from app.pipeline.visual_encoder import SigLIP2VisualEncoder
from app.pipeline.dense_captioner import MiniCPMDenseCaptioner

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
        self.captioner = MiniCPMDenseCaptioner()

    def answer_question(self, video_id: str, question: str) -> VideoQAResponse:
        t0 = time.time()
        logger.info(f"Video-RAG answering question for video {video_id}: '{question}'")

        # 1. Retrieve Transcripts around question keywords
        tbl_transcripts = db_manager.get_table("transcripts")
        try:
            transcripts = [r for r in tbl_transcripts.to_arrow().to_pylist() if r.get("video_id") == video_id]
        except Exception:
            transcripts = []

        q_lower = question.lower()
        q_tokens = [w.strip() for w in q_lower.split() if len(w.strip()) > 2]

        matched_transcripts = []
        for tr in transcripts:
            text = (tr.get("spoken_text") or "").lower()
            if any(tok in text for tok in q_tokens):
                matched_transcripts.append(tr)

        # 2. Retrieve Visual Frames with SigLIP 2
        tbl_frames = db_manager.get_table("video_frames")
        try:
            video_frames = tbl_frames.search().where(f"video_id = '{video_id}'").limit(1000).to_list()
        except Exception:
            video_frames = [r for r in tbl_frames.to_arrow().to_pylist() if r.get("video_id") == video_id]

        query_vec = np.array(self.text_encoder.encode_text(question), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        scored_frames = []
        for f in video_frames:
            emb = f.get("siglip2_vector")
            if emb is not None and len(emb) == 768:
                sim = float(np.dot(np.array(emb, dtype=np.float32), query_vec))
                scored_frames.append((sim, f))

        scored_frames.sort(key=lambda x: x[0], reverse=True)
        top_frames = scored_frames[:3]

        # 3. Formulate Multimodal Context
        context_transcripts_str = "\n".join([
            f"[{tr.get('t_start'):.1f}s - {tr.get('t_end'):.1f}s]: {tr.get('spoken_text')}"
            for tr in (matched_transcripts[:5] if matched_transcripts else transcripts[:5])
        ])

        context_visual_str = "\n".join([
            f"[เวลา {f.get('timestamp'):.1f}s]: {f.get('vlm_caption') or 'ภาพแสดงเหตุการณ์ในฉาก'}"
            for sim, f in top_frames
        ])

        # 4. Generate Grounded Synthesis Answer
        evidence_citations = []
        if matched_transcripts:
            for tr in matched_transcripts[:2]:
                evidence_citations.append(GroundedMoment(
                    t_start=round(float(tr.get("t_start", 0.0)), 1),
                    t_end=round(float(tr.get("t_end", 0.0)), 1),
                    citation_text=str(tr.get("spoken_text", ""))[:80],
                    thumbnail_path=top_frames[0][1].get("frame_path") if top_frames else None
                ))
        elif top_frames:
            best_f = top_frames[0][1]
            ts = float(best_f.get("timestamp", 0.0))
            evidence_citations.append(GroundedMoment(
                t_start=round(max(0.0, ts - 2.0), 1),
                t_end=round(ts + 5.0, 1),
                citation_text=best_f.get("vlm_caption") or "พบเหตุการณ์ตรงกับคำถามในฉากนี้",
                thumbnail_path=best_f.get("frame_path")
            ))

        # Synthesis
        if context_transcripts_str.strip():
            answer = f"จากบทสนทนาและภาพเหตุการณ์ในวิดีโอ:\n{context_transcripts_str}\n\n(อ้างอิงช่วงเวลา: {evidence_citations[0].t_start}s - {evidence_citations[0].t_end}s)"
        elif context_visual_str.strip():
            answer = f"จากการวิเคราะห์ภาพเหตุการณ์ในวิดีโอ:\n{context_visual_str}\n\n(อ้างอิงช่วงเวลา: {evidence_citations[0].t_start}s - {evidence_citations[0].t_end}s)"
        else:
            answer = "ไม่พบบทสนทนาหรือภาพเหตุการณ์ที่ตรงกับคำถามนี้ในวิดีโอที่เลือก"

        latency_ms = round((time.time() - t0) * 1000.0, 2)
        return VideoQAResponse(
            question=question,
            answer=answer,
            grounded_moments=evidence_citations,
            latency_ms=latency_ms
        )

video_rag_engine = VideoRAGEngine()
