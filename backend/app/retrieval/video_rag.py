import time
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logger import logger
from app.db.connection import db_manager
from app.pipeline.visual_encoder import SigLIP2VisualEncoder
from app.inference.client import inference_client
from app.inference.contracts import AnswerRequest

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
    models_used: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class VideoRAGEngine:
    """
    Visual Video-RAG Engine.
    Retrieves visual keyframes and captions to answer complex questions about the video.
    """

    def __init__(self):
        self.text_encoder = SigLIP2VisualEncoder()

    def answer_question(self, video_id: str, question: str) -> VideoQAResponse:
        t0 = time.time()
        logger.info(f"Video-RAG answering question for video {video_id}: '{question}'")

        # 1. Retrieve Visual Frames with SigLIP 2 & Caption Matching
        tbl_frames = db_manager.get_table("video_frames_v2")
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
            sim = 0.0
            if emb is not None and len(emb) == 768:
                sim = float(np.dot(np.array(emb, dtype=np.float32), query_vec))
            
            scored_frames.append((sim, f))

        scored_frames.sort(key=lambda x: x[0], reverse=True)
        top_frames = scored_frames[: max(3, int(settings.QWEN_MAX_FRAMES_PER_CANDIDATE))]

        # 2. Formulate Visual Context
        scenes = {str(row.get("id")): row for row in db_manager.get_table("scenes_v2").to_arrow().to_pylist()}
        context_visual_str = "\n".join([
            f"[เวลา {f.get('timestamp'):.1f}s]: {scenes.get(str(f.get('scene_id')), {}).get('caption') or 'ภาพแสดงการกระทำและเหตุการณ์ในฉาก'}"
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
                    citation_text=(scenes.get(str(f.get("scene_id")), {}).get("caption") or "ฉากเหตุการณ์ที่ตรงกับคำถาม")[:100],
                    thumbnail_path=f.get("frame_path")
                ))

        # Evidence-grounded CapRL Q6 answer. The isolated worker owns the only
        # production VLM; there is deliberately no in-process Qwen fallback.
        answer = ""
        models_used: List[str] = [settings.SIGLIP2_MODEL_ID]
        warnings: List[str] = []
        valid_frame_paths = [str(f.get("frame_path")) for _, f in top_frames if f.get("frame_path")]
        evidence_payload = [moment.model_dump() for moment in evidence_citations]
        if valid_frame_paths and context_visual_str.strip():
            prompt = (
                "Answer the user's question using only the supplied classroom CCTV frames and structured evidence. "
                "Do not identify people, infer identity, or use audio/OCR. If evidence is insufficient, say so. "
                f"Question: {question}. Evidence: {context_visual_str}"
            )
            if inference_client.enabled:
                try:
                    response = inference_client.answer(AnswerRequest(
                        question=question,
                        frame_paths=valid_frame_paths,
                        timestamps=[float(f.get("timestamp", 0.0)) for _, f in top_frames if f.get("frame_path")],
                        evidence=evidence_payload,
                        max_new_tokens=256,
                    ))
                    answer = response.answer.strip()
                    if response.model_id:
                        models_used.append(response.model_id)
                except Exception as exc:
                    warnings.append("caprl_q6_worker_unavailable")
                    logger.warning(f"Video VQA CapRL Q6 worker unavailable: {exc}")
            else:
                warnings.append("inference_worker_unavailable")
                logger.warning("Video VQA requires the local CapRL Q6 inference worker")

        if not answer:
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
            latency_ms=latency_ms,
            models_used=models_used,
            warnings=warnings,
        )

video_rag_engine = VideoRAGEngine()
