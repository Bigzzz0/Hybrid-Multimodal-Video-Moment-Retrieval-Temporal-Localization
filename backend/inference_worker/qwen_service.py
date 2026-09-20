from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from app.core.config import settings
from app.inference.contracts import AnswerRequest, AnswerResponse, CaptionRequest, CaptionResponse, VerifyRequest, VerifyResponse


class Qwen3VLService:
    def __init__(self) -> None:
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

        token = settings.HF_TOKEN
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            settings.QWEN_VL_MODEL_ID,
            torch_dtype=torch.bfloat16,
            quantization_config=quant,
            device_map="auto",
            token=token,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(settings.QWEN_VL_MODEL_ID, token=token)

    def unload(self) -> None:
        self.model = None
        self.processor = None

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text or ""):
            try:
                data, _ = decoder.raw_decode(text[match.start():])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
        return {}

    @staticmethod
    def _extract_partial_verify(text: str) -> Dict[str, Any]:
        """Recover verifier fields when Qwen truncates only the final reason."""
        patterns = {
            "event_present": r'"event_present"\s*:\s*(true|false)',
            "start_frame_index": r'"start_frame_index"\s*:\s*(-?\d+)',
            "end_frame_index": r'"end_frame_index"\s*:\s*(-?\d+)',
            "confidence": r'"confidence"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))',
        }
        values: Dict[str, Any] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if not match:
                return {}
            raw = match.group(1)
            if key == "event_present":
                values[key] = raw.lower() == "true"
            elif key == "confidence":
                values[key] = float(raw)
            else:
                values[key] = int(raw)
        reason = re.search(r'"reason"\s*:\s*"([^"}]*)', text or "", flags=re.IGNORECASE)
        values["reason"] = reason.group(1).strip() if reason else ""
        return values

    def _generate(self, paths: List[str], prompt: str, max_new_tokens: int) -> str:
        import torch
        from PIL import Image

        images = [Image.open(path).convert("RGB") for path in paths if path]
        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": image} for image in images],
            {"type": "text", "text": prompt},
        ]}]
        # Keep the verifier's small structured response from being consumed by
        # a long Qwen reasoning block.
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max(16, int(max_new_tokens)), do_sample=False)
        trimmed = [ids[len(input_ids):] for input_ids, ids in zip(inputs["input_ids"], output)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def caption(self, request: CaptionRequest) -> CaptionResponse:
        prompt = request.prompt or (
            "Analyze these chronological classroom surveillance frames. Return only JSON with keys "
            "summary, people_count, objects, actions, relations, uncertainty. "
            "Describe visible facts only. Never identify or name people."
        )
        raw = self._generate(request.frame_paths, prompt, request.max_new_tokens)
        data = self._extract_json(raw)
        if not data:
            repair = self._generate(
                request.frame_paths,
                "Repair the previous visual analysis into valid JSON only with keys summary, people_count, objects, actions, relations, uncertainty.",
                min(128, request.max_new_tokens),
            )
            repaired = self._extract_json(repair)
            if repaired:
                raw, data = repair, repaired
        return CaptionResponse(
            text=raw,
            summary=str(data.get("summary", raw)),
            people_count=data.get("people_count") if isinstance(data.get("people_count"), int) else None,
            objects=[str(value) for value in data.get("objects", [])] if isinstance(data.get("objects", []), list) else [],
            actions=[str(value) for value in data.get("actions", [])] if isinstance(data.get("actions", []), list) else [],
            relations=[str(value) for value in data.get("relations", [])] if isinstance(data.get("relations", []), list) else [],
            uncertainty=[str(value) for value in data.get("uncertainty", [])] if isinstance(data.get("uncertainty", []), list) else [],
            model_id=settings.QWEN_VL_MODEL_ID,
            status="generated",
        )

    def verify(self, request: VerifyRequest) -> VerifyResponse:
        mapping = list(enumerate([round(float(value), 3) for value in request.timestamps]))
        prompt = (
            f"Determine whether the visual event in this query is present: {request.query}. "
            f"Frame mapping is {mapping}. Return only JSON with keys event_present, start_frame_index, "
            "end_frame_index, confidence, reason. Use integer frame indices, not seconds. "
            "Do not identify people. Keep reason under 12 words."
        )
        if request.caption_hint:
            prompt += f" Caption hint (verify visually): {request.caption_hint}"
        raw = self._generate(request.frame_paths, prompt, request.max_new_tokens)
        data = self._extract_json(raw)
        required = {"event_present", "start_frame_index", "end_frame_index", "confidence"}
        if not required.issubset(data):
            data = self._extract_partial_verify(raw)
        if not required.issubset(data):
            repair = self._generate(
                request.frame_paths,
                "Return one compact JSON object only. Keys: event_present, start_frame_index, end_frame_index, confidence, reason. Keep reason under 8 words.",
                max(96, min(128, request.max_new_tokens)),
            )
            repaired = self._extract_json(repair)
            if not repaired:
                repaired = self._extract_partial_verify(repair)
            if repaired:
                raw, data = repair, repaired
        count = len(request.timestamps)
        start = int(data.get("start_frame_index", 0)) if count else 0
        end = int(data.get("end_frame_index", max(0, count - 1))) if count else 0
        start = max(0, min(start, max(0, count - 1)))
        end = max(start, min(end, max(0, count - 1)))
        confidence = float(data.get("confidence", 0.0) or 0.0)
        return VerifyResponse(
            event_present=bool(data.get("event_present", False)),
            start_frame_index=start,
            end_frame_index=end,
            confidence=max(0.0, min(1.0, confidence)),
            reason=str(data.get("reason", "")),
            raw_text=raw,
            model_id=settings.QWEN_VL_MODEL_ID,
        )

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        prompt = (
            f"Answer the question using only the supplied visual frames and evidence: {request.question}. "
            "Return a concise answer. If evidence is insufficient, say so. Never identify people. "
            f"Structured evidence: {json.dumps(request.evidence, ensure_ascii=False)}"
        )
        raw = self._generate(request.frame_paths, prompt, request.max_new_tokens)
        return AnswerResponse(answer=raw, model_id=settings.QWEN_VL_MODEL_ID, status="generated")
