import os
import torch
from PIL import Image
from typing import List, Optional
from app.core.config import settings
from app.core.logger import logger

class QwenVLDenseCaptioner:
    """
    Dense Action, Video Understanding, and Physical State Transition Captioner
    powered by 4-bit Quantized Qwen2.5-VL-7B-Instruct (Alibaba Cloud / Qwen Team).
    Supports multi-frame sequence reasoning, temporal action descriptions, and physical state transitions.
    """

    def __init__(self, model_id: str = settings.QWEN_VL_MODEL_ID, device: str = settings.DEVICE):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.last_status = "unavailable"

    def _lazy_load(self):
        if self.model is None:
            if self.device != "cuda":
                logger.info("CUDA not available. Qwen2.5-VL captioner disabled on CPU; scene caption unavailable.")
                self.last_status = "unavailable"
                return

            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes < int(settings.VLM_MIN_FREE_VRAM_MB) * 1024 * 1024:
                    logger.warning("Insufficient free VRAM for Qwen2.5-VL; leaving captions unavailable to protect the 8GB budget.")
                    self.last_status = "unavailable"
                    return
            except Exception:
                # If the driver does not expose mem_get_info, let the guarded
                # model load below decide and fall back on failure.
                pass

            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
            token = settings.HF_TOKEN or os.environ.get("HF_TOKEN")
            logger.info(f"Loading 4-bit Quantized Qwen2.5-VL-7B ({self.model_id})...")

            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=token
                ).eval()
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    token=token
                )
                logger.info("✅ Qwen2.5-VL-7B-Instruct (4-bit) successfully loaded into GPU VRAM.")
            except Exception as e:
                logger.warning(
                    f"Failed to load Qwen2.5-VL-7B ({e}).\n"
                    f"Ensure model id '{self.model_id}' is accessible or check huggingface token in backend/.env."
                )
                self.model = None
                self.processor = None
                self.last_status = "error"

    def _is_refusal_response(self, text: str) -> bool:
        """Checks if the LLM output is a canned refusal response instead of a scene description."""
        if not text:
            return True
        t_low = text.lower()
        refusal_phrases = [
            "sorry", "cannot browse", "can't browse", "unable to browse",
            "not able to browse", "large language model", "training data",
            "cutoff date", "as an ai", "i am an ai", "don't have access",
            "not sure what you are asking", "clarify your question", "how can i help"
        ]
        return any(phrase in t_low for phrase in refusal_phrases)

    def generate_scene_caption(
        self,
        keyframes: List[Image.Image],
        prompt_override: Optional[str] = None,
        max_frames: int = 4,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Generate dense action, temporal interaction, and physical state transition caption
        for a sequence of keyframes in a video scene using Qwen2.5-VL-7B.
        """
        self._lazy_load()
        if not keyframes:
            self.last_status = "unavailable"
            return ""

        if self.model is None or self.processor is None:
            if self.last_status not in {"error", "unavailable"}:
                self.last_status = "unavailable"
            return ""

        try:
            from qwen_vl_utils import process_vision_info

            # Dense scene captioning defaults to four frames.  The Accurate
            # temporal verifier may explicitly request all of its <=8 sampled
            # frames so that Qwen can return frame-index boundaries faithfully.
            sample_count = min(max(1, int(max_frames)), len(keyframes))
            if sample_count <= 1:
                sampled = [keyframes[0]]
            else:
                step = (len(keyframes) - 1) / (sample_count - 1)
                sampled = [keyframes[int(i * step)] for i in range(sample_count)]

            # Convert images to RGB if necessary
            sampled_rgb = [img.convert("RGB") if img.mode != "RGB" else img for img in sampled]

            prompt_text = prompt_override or (
                "Analyze the chronological sequence of these video keyframes. "
                "Describe the physical actions, subject movements, body gestures, and object interactions in 2 concise sentences. "
                "Highlight specifically what actions occurred and any visible physical state change from start to finish. "
                "Focus purely on visual activities and dynamic movements without describing on-screen presentation text."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in sampled_rgb],
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            # Move tensors to model device
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max(16, int(max_new_tokens)),
                    do_sample=False
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                caption_str = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()

                if self._is_refusal_response(caption_str):
                    logger.warning(f"Qwen2.5-VL returned refusal response. Discarding: {caption_str[:60]}...")
                    self.last_status = "unavailable"
                    return ""

                self.last_status = "generated" if caption_str else "unavailable"
                return caption_str

        except Exception as ex:
            logger.error(f"Qwen2.5-VL captioning error ({ex}); scene caption unavailable")
            self.last_status = "error"
            return ""

# Global singleton
dense_captioner = QwenVLDenseCaptioner()
