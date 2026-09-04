import json
import re
import os
from typing import List, Dict, Any, Optional
from PIL import Image
from app.core.config import settings
from app.core.logger import logger

class VLMStage2Verifier:
    """
    Stage 2: Spatiotemporal Action Grounding Verifier (TimeLens CVPR 2026).
    Employs Qwen2.5-VL-7B to inspect candidate moments, confirm action presence,
    and refine sub-second start/end timestamps.
    """

    def __init__(self):
        self._captioner = None

    @property
    def captioner(self):
        if self._captioner is None:
            from app.pipeline.dense_captioner import dense_captioner
            self._captioner = dense_captioner
        return self._captioner

    def verify_and_refine(
        self,
        candidate_moments: List[Dict[str, Any]],
        video_frames: List[Dict[str, Any]],
        query: str,
        top_k_verify: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Takes candidate moments from Stage 1, selects the top_k candidates,
        and uses Qwen2.5-VL to confirm physical action grounding.
        """
        if not candidate_moments:
            return []

        verified_moments = []
        candidates_to_check = candidate_moments[:top_k_verify]
        remaining_candidates = candidate_moments[top_k_verify:]

        for m in candidates_to_check:
            ts, te = m["t_start"], m["t_end"]
            
            # 1. Sample 4 equidistant frames within [ts - 0.5s, te + 0.5s]
            interval_frames = [
                f for f in video_frames
                if (ts - 0.5) <= float(f.get("timestamp", 0.0)) <= (te + 0.5)
            ]
            interval_frames.sort(key=lambda x: float(x.get("timestamp", 0.0)))

            if len(interval_frames) < 2:
                verified_moments.append(m)
                continue

            step = max(1, len(interval_frames) // 4)
            sampled_frames = interval_frames[::step][:4]
            timestamps_list = [round(float(f.get("timestamp", 0.0)), 2) for f in sampled_frames]
            images = [Image.open(f.get("frame_path")) for f in sampled_frames if f.get("frame_path") and os.path.exists(f.get("frame_path", ""))]

            if len(images) < 2:
                verified_moments.append(m)
                continue

            # 2. Construct Grounding Verification Prompt
            prompt = (
                f"You are a video moment retrieval temporal grounding judge.\n"
                f"Target action to verify: '{query}'\n"
                f"Frame timestamps provided: {timestamps_list}\n"
                f"Analyze if the action '{query}' is physically occurring in these frames.\n"
                f"Respond strictly in valid JSON format:\n"
                f"{{\n"
                f'  "action_found": true,\n'
                f'  "refined_start": {ts},\n'
                f'  "refined_end": {te},\n'
                f'  "confidence": 0.90\n'
                f"}}"
            )

            try:
                raw_response = self.captioner.generate_scene_caption(images, prompt_override=prompt)
                
                # Extract JSON using regex
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    res_dict = json.loads(json_match.group(0))
                    action_found = bool(res_dict.get("action_found", True))
                    ref_start = float(res_dict.get("refined_start", ts))
                    ref_end = float(res_dict.get("refined_end", te))
                    vlm_conf = float(res_dict.get("confidence", 0.85))

                    if action_found:
                        fused_score = round(0.50 * m["score"] + 0.50 * vlm_conf, 4)
                        m_verified = dict(m)
                        m_verified["score"] = fused_score
                        m_verified["t_start"] = round(max(0.0, ref_start), 2)
                        m_verified["t_end"] = round(max(ref_start + 1.0, ref_end), 2)
                        verified_moments.append(m_verified)
                    else:
                        # Suppress False Positive heavily
                        m_suppressed = dict(m)
                        m_suppressed["score"] = round(0.15 * m["score"], 4)
                        verified_moments.append(m_suppressed)
                else:
                    verified_moments.append(m)
            except Exception as vlm_err:
                logger.warning(f"VLM Verification fallback due to error: {vlm_err}")
                verified_moments.append(m)

        all_moments = verified_moments + remaining_candidates
        all_moments.sort(key=lambda x: x["score"], reverse=True)
        return all_moments

vlm_verifier = VLMStage2Verifier()
