"""Variant-aware local VLM service for the A--E ablation.

Heavy dependencies are imported inside ``load`` so the API and Fast profile
can start even when the optional lab environment or a gated checkpoint is not
installed yet.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import base64
import mimetypes
from typing import Any, Dict, List

from app.core.config import settings
from app.inference.contracts import AnswerRequest, AnswerResponse, CaptionRequest, CaptionResponse, VerifyRequest, VerifyResponse
from app.inference.vlm_registry import VLMVariant, get_variant


class VLMBackendError(RuntimeError):
    pass


class VariantVLMService:
    def __init__(self, backend: str = "qwen3_vl_2b") -> None:
        self.variant: VLMVariant = get_variant(backend)
        self.model: Any = None
        self.processor: Any = None
        self.adapter: Any = None
        self.server_process: Any = None
        self.server_port: int = 0
        self.loaded = False

    @property
    def backend(self) -> str:
        return self.variant.backend

    def metadata(self) -> Dict[str, str]:
        return {
            "vlm_backend": self.variant.backend,
            "model_id": self.variant.model_id,
            "model_revision": self.variant.revision,
            "quantization": self.variant.quantization,
            "input_mode": self.variant.input_mode,
            "artifact_version": settings.VLM_ARTIFACT_VERSION,
        }

    def load(self) -> None:
        if self.loaded:
            return
        if self.variant.runtime == "llama_cpp":
            self._load_llama_server()
            self.loaded = True
            return

        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover - exercised on optional lab envs
            raise VLMBackendError(f"transformers VLM dependencies unavailable: {exc}") from exc

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        token = settings.HF_TOKEN
        kwargs = {
            "revision": self.variant.revision,
            "torch_dtype": torch.bfloat16,
            "quantization_config": quant,
            "device_map": "auto",
            "token": token,
        }
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.variant.model_id, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.variant.model_id,
            revision=self.variant.revision,
            token=token,
        )
        if self.variant.adapter_id:
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.variant.adapter_id,
                    revision=self.variant.adapter_revision,
                    token=token,
                ).eval()
                self.adapter = self.variant.adapter_id
            except Exception as exc:
                self.unload()
                raise VLMBackendError(f"VISE adapter could not be loaded: {exc}") from exc
        self.loaded = True

    def _load_llama_server(self) -> None:
        executable = settings.LLAMA_CPP_PATH
        model_dir = settings.VLM_GGUF_DIR
        if not executable or not model_dir:
            raise VLMBackendError(
                "llama.cpp backend is configured but LLAMA_CPP_PATH and VLM_GGUF_DIR are not set"
            )
        model_path = os.path.join(model_dir, self.variant.model_file)
        mmproj_path = os.path.join(model_dir, self.variant.mmproj_file)
        if not os.path.exists(executable) or not os.path.exists(model_path) or not os.path.exists(mmproj_path):
            raise VLMBackendError(f"GGUF/mmproj files are missing for {self.variant.backend}")
        port = 18100 + list(("caprl_qwen3vl_4b_q4", "caprl_qwen3vl_4b_q6")).index(self.variant.backend)
        self.server_port = port
        self.server_process = subprocess.Popen([
            executable, "--model", model_path, "--mmproj", mmproj_path,
            "--host", "127.0.0.1", "--port", str(port), "--n-gpu-layers", "99",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # GGUF weights are memory-mapped and uploaded to the 5070 before the
        # health endpoint changes from 503 to 200.  Thirty seconds is too short
        # on a cold Windows start, especially for the Q6 file.
        deadline = time.perf_counter() + 180.0
        last_error = ""
        try:
            import httpx
            while time.perf_counter() < deadline:
                if self.server_process.poll() is not None:
                    raise VLMBackendError("llama.cpp server exited during startup")
                try:
                    response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                    if response.status_code == 200:
                        return
                    last_error = f"HTTP {response.status_code}"
                except Exception as exc:
                    last_error = str(exc)
                time.sleep(0.25)
        except VLMBackendError:
            self.unload()
            raise
        except Exception as exc:
            self.unload()
            raise VLMBackendError(f"llama.cpp server did not become ready: {last_error or exc}") from exc
        self.unload()
        raise VLMBackendError(f"llama.cpp server did not become ready: {last_error or 'timeout'}")

    def unload(self) -> None:
        process = self.server_process
        self.server_process = None
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        self.model = None
        self.processor = None
        self.adapter = None
        self.loaded = False
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        decoder = json.JSONDecoder()
        source = (text or "").strip()
        candidates = [source]
        candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", source, flags=re.IGNORECASE))
        for candidate in candidates:
            for match in re.finditer(r"\{", candidate):
                try:
                    value, _ = decoder.raw_decode(candidate[match.start():])
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    continue
        return {}

    @staticmethod
    def _extract_partial_caption(text: str) -> Dict[str, Any]:
        """Recover useful fields when a model stops inside a JSON object.

        Small local VLMs occasionally emit a fenced or token-truncated object.
        We keep this conservative: only explicitly named schema fields are
        recovered, and the result is marked invalid by the caller.
        """
        source = re.sub(r"```(?:json)?|```", "", text or "", flags=re.IGNORECASE)
        data: Dict[str, Any] = {}
        summary = re.search(r'"summary"\s*:\s*"((?:\\.|[^"\\])*)', source, flags=re.IGNORECASE | re.DOTALL)
        if summary:
            try:
                data["summary"] = json.loads('"' + summary.group(1) + '"')
            except json.JSONDecodeError:
                data["summary"] = summary.group(1).strip()
        for key in ("objects", "attributes", "actions", "relations", "temporal_events", "uncertainty"):
            match = re.search(rf'"{key}"\s*:\s*\[([\s\S]*?)(?:\]|$)', source, flags=re.IGNORECASE)
            if not match:
                continue
            values: List[str] = []
            for item in re.finditer(r'"((?:\\.|[^"\\])*)"', match.group(1)):
                try:
                    values.append(str(json.loads('"' + item.group(1) + '"')))
                except json.JSONDecodeError:
                    values.append(item.group(1).strip())
            if values:
                data[key] = values
        return data

    @staticmethod
    def _extract_partial_verify(text: str) -> Dict[str, Any]:
        patterns = {
            "event_present": r'"event_present"\s*:\s*(true|false)',
            "start_frame_index": r'"start_frame_index"\s*:\s*(-?\d+)',
            "end_frame_index": r'"end_frame_index"\s*:\s*(-?\d+)',
            "confidence": r'"confidence"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))',
        }
        data: Dict[str, Any] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if not match:
                return {}
            raw = match.group(1)
            data[key] = raw.lower() == "true" if key == "event_present" else float(raw) if key == "confidence" else int(raw)
        reason = re.search(r'"reason"\s*:\s*"([^"}]*)', text or "", flags=re.IGNORECASE)
        data["reason"] = reason.group(1).strip() if reason else ""
        return data

    def _generate_frames(self, paths: List[str], prompt: str, max_new_tokens: int) -> str:
        if self.variant.runtime == "llama_cpp":
            if not self.server_process or self.server_process.poll() is not None:
                raise VLMBackendError("llama.cpp server is not running")
            try:
                import httpx
                content: List[Dict[str, Any]] = []
                for path in paths:
                    if not path or not os.path.exists(path):
                        continue
                    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
                    with open(path, "rb") as image_file:
                        encoded = base64.b64encode(image_file.read()).decode("ascii")
                    content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
                content.append({"type": "text", "text": prompt})
                response = None
                for _ in range(3):
                    try:
                        response = httpx.post(
                            f"http://127.0.0.1:{self.server_port}/v1/chat/completions",
                            json={"messages": [{"role": "user", "content": content}], "max_tokens": max(16, int(max_new_tokens)), "temperature": 0},
                            timeout=60.0,
                        )
                        if response.status_code < 500:
                            break
                    except Exception:
                        time.sleep(1.0)
                if response is None or response.status_code >= 400:
                    raise VLMBackendError(f"llama.cpp completion failed: {response.text if response is not None else 'no response'}")
                data = response.json()
                return str(data["choices"][0]["message"]["content"]).strip()
            except VLMBackendError:
                raise
            except Exception as exc:
                raise VLMBackendError(f"llama.cpp completion failed: {exc}") from exc
        import torch
        from PIL import Image

        images = [Image.open(path).convert("RGB") for path in paths if path and os.path.exists(path)]
        if not images:
            raise VLMBackendError("VLM request contains no readable frame paths")
        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": image} for image in images],
            {"type": "text", "text": prompt},
        ]}]
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            output = self.model.generate(**inputs, max_new_tokens=max(16, int(max_new_tokens)), do_sample=False)
        trimmed = [ids[len(input_ids):] for input_ids, ids in zip(inputs["input_ids"], output)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def _generate(self, request: Any, prompt: str, max_new_tokens: int) -> str:
        # The ingestion/search pipeline samples local frames for all variants.
        # E keeps its video_chunk identity in the contract and can additionally
        # receive video_path; frame paths remain the deterministic fallback for
        # Windows builds where qwen-vl-utils cannot decode a container.
        if (settings.VLM_VIDEO_USE_CONTAINER and self.variant.input_mode == "video_chunk"
                and getattr(request, "video_path", "") and os.path.exists(request.video_path)):
            try:
                return self._generate_video(request.video_path, prompt, max_new_tokens, float(getattr(request, "sample_fps", 2.0) or 2.0))
            except Exception:
                # Windows decoders and older qwen-vl-utils versions differ in
                # their video backend. A sampled local-frame request is a
                # deterministic fallback, never a cloud call.
                pass
        paths = list(getattr(request, "frame_paths", []) or [])
        return self._generate_frames(paths, prompt, max_new_tokens)

    def _generate_video(self, video_path: str, prompt: str, max_new_tokens: int, fps: float) -> str:
        import torch
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": max(0.1, fps)},
            {"type": "text", "text": prompt},
        ]}]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            output = self.model.generate(**inputs, max_new_tokens=max(16, int(max_new_tokens)), do_sample=False)
        trimmed = [ids[len(input_ids):] for input_ids, ids in zip(inputs["input_ids"], output)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def caption(self, request: CaptionRequest) -> CaptionResponse:
        prompt = request.prompt or (
            "Analyze only visible visual content in these chronological frames. "
            "Return exactly one compact JSON object with keys summary, objects, attributes, actions, relations, temporal_events, uncertainty. "
            "Keep summary under 40 words and each array under 6 items. Do not use markdown, OCR, subtitles, audio, names, identities or face recognition."
        )
        started = time.perf_counter()
        raw = self._generate(request, prompt, max(int(request.max_new_tokens), settings.VLM_CAPTION_MAX_NEW_TOKENS))
        data = self._extract_json(raw)
        json_repaired = False
        if not data:
            repair_raw = self._generate(
                request,
                "Return one compact valid JSON object only. Keys: summary, objects, attributes, actions, relations, temporal_events, uncertainty. No markdown.",
                min(settings.VLM_JSON_REPAIR_MAX_NEW_TOKENS, max(64, int(request.max_new_tokens))),
            )
            data = self._extract_json(repair_raw)
            if data:
                raw = repair_raw
                json_repaired = True
            else:
                data = self._extract_partial_caption(repair_raw) or self._extract_partial_caption(raw)
        meta = self.metadata()
        return CaptionResponse(
            text=raw,
            summary=str(data.get("summary", raw)),
            objects=[str(x) for x in data.get("objects", [])] if isinstance(data.get("objects", []), list) else [],
            attributes=[str(x) for x in data.get("attributes", [])] if isinstance(data.get("attributes", []), list) else [],
            actions=[str(x) for x in data.get("actions", [])] if isinstance(data.get("actions", []), list) else [],
            relations=[str(x) for x in data.get("relations", [])] if isinstance(data.get("relations", []), list) else [],
            temporal_events=[str(x) for x in data.get("temporal_events", [])] if isinstance(data.get("temporal_events", []), list) else [],
            uncertainty=[str(x) for x in data.get("uncertainty", [])] if isinstance(data.get("uncertainty", []), list) else [],
            people_count=data.get("people_count") if isinstance(data.get("people_count"), int) else None,
            model_id=meta["model_id"], status="generated", **{k: meta[k] for k in ("vlm_backend", "model_revision", "quantization", "input_mode", "artifact_version")},
            inference_ms=round((time.perf_counter() - started) * 1000, 1),
            json_valid=bool(self._extract_json(raw)),
            json_repaired=json_repaired,
        )

    def verify(self, request: VerifyRequest) -> VerifyResponse:
        mapping = list(enumerate([round(float(value), 3) for value in request.timestamps]))
        prompt = (
            f"Determine whether this visible visual event is present: {request.query}. "
            f"Semantic requirements: {request.semantic_requirements}. Frame mapping: {mapping}. "
            "Return JSON only with event_present, start_frame_index, end_frame_index, confidence, reason. "
            "Use integer frame indices, not seconds. Do not identify people or use OCR."
        )
        if request.caption_hint:
            prompt += f" Caption hint to verify visually: {request.caption_hint}"
        raw = self._generate(request, prompt, request.max_new_tokens)
        data = self._extract_json(raw)
        required = {"event_present", "start_frame_index", "end_frame_index", "confidence"}
        if not required.issubset(data):
            data = self._extract_partial_verify(raw)
        if not required.issubset(data):
            repaired = self._generate(
                request,
                "Return one compact JSON object only with event_present, start_frame_index, end_frame_index, confidence, reason.",
                max(64, min(128, request.max_new_tokens)),
            )
            repaired_data = self._extract_json(repaired) or self._extract_partial_verify(repaired)
            if repaired_data:
                raw, data = repaired, repaired_data
        count = len(request.timestamps)
        start = int(data.get("start_frame_index", 0) or 0) if count else 0
        end = int(data.get("end_frame_index", max(0, count - 1)) or 0) if count else 0
        start = max(0, min(start, max(0, count - 1)))
        end = max(start, min(end, max(0, count - 1)))
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0)))
        meta = self.metadata()
        return VerifyResponse(
            event_present=bool(data.get("event_present", False)),
            start_frame_index=start,
            end_frame_index=end,
            confidence=confidence,
            reason=str(data.get("reason", "")), raw_text=raw,
            model_id=meta["model_id"], **{k: meta[k] for k in ("vlm_backend", "model_revision", "quantization", "input_mode", "artifact_version")},
        )

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        prompt = (
            f"Answer only from visible visual evidence in the supplied frames: {request.question}. "
            "If evidence is insufficient, say so. Do not use OCR, subtitles, audio or identity. "
            f"Structured evidence: {json.dumps(request.evidence, ensure_ascii=False)}"
        )
        raw = self._generate(request, prompt, request.max_new_tokens)
        meta = self.metadata()
        return AnswerResponse(answer=raw, model_id=meta["model_id"], status="generated", **{k: meta[k] for k in ("vlm_backend", "model_revision", "quantization", "input_mode")})


class Qwen3VLService(VariantVLMService):
    """Compatibility wrapper for the pre-ablation /v1/qwen endpoints."""

    def __init__(self) -> None:
        super().__init__("qwen3_vl_2b")
