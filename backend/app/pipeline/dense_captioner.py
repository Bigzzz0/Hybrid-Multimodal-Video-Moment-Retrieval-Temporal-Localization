import os
import torch
import torch.nn as nn
from PIL import Image
from typing import List, Optional
from transformers.modeling_utils import PreTrainedModel
from app.core.config import settings
from app.core.logger import logger

# Compatibility patch for Transformers >= 4.45 with MiniCPM-V remote code
def patch_transformers_for_minicpm():
    """Patches tied weights dictionary properties for MiniCPM-V compatibility with newer transformers."""
    def get_all_tied(self):
        val = self.__dict__.get("_all_tied_weights_keys", None)
        if val is None:
            return {}
        if isinstance(val, list):
            return {k: [] for k in val}
        if isinstance(val, dict):
            return val
        return {}

    def set_all_tied(self, val):
        self.__dict__["_all_tied_weights_keys"] = val

    def get_tied(self):
        val = self.__dict__.get("_tied_weights_keys", None)
        if val is None:
            return {}
        if isinstance(val, list):
            return {k: [] for k in val}
        return val

    def set_tied(self, val):
        self.__dict__["_tied_weights_keys"] = val

    setattr(PreTrainedModel, "all_tied_weights_keys", property(get_all_tied, set_all_tied))
    setattr(PreTrainedModel, "_tied_weights_keys", property(get_tied, set_tied))
    setattr(nn.Module, "all_tied_weights_keys", property(get_all_tied, set_all_tied))
    setattr(nn.Module, "_tied_weights_keys", property(get_tied, set_tied))

class MiniCPMDenseCaptioner:
    """
    Dense Action & Interaction Captioner using 4-bit Quantized MiniCPM-V 2.6 (OpenBMB).
    Supports multi-image sequential understanding for video scenes within <= 5.5GB VRAM.
    """

    def __init__(self, model_id: str = settings.MINICPMV_MODEL_ID, device: str = settings.DEVICE):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None

    def _lazy_load(self):
        if self.model is None:
            if self.device != "cuda":
                logger.info("CUDA not available. MiniCPM-V 4-bit captioner disabled on CPU; using fast descriptor.")
                return

            patch_transformers_for_minicpm()

            from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
            token = settings.HF_TOKEN or os.environ.get("HF_TOKEN")
            logger.info(f"Loading 4-bit Quantized MiniCPM-V 2.6 ({self.model_id})...")

            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=token
                ).eval()
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    token=token
                )
                logger.info("MiniCPM-V 2.6 (4-bit) successfully loaded into GPU VRAM.")
            except Exception as e:
                logger.warning(
                    f"Failed to load MiniCPM-V 2.6 ({e}).\n"
                    "TIP: If you see a 401 error, visit https://huggingface.co/openbmb/MiniCPM-V-2_6 to accept terms, "
                    "or set your HF_TOKEN in backend/.env or run `huggingface-cli login`."
                )
                self.model = None

    def _is_refusal_response(self, text: str) -> bool:
        """Checks if the LLM output is a canned refusal response instead of a scene description."""
        if not text:
            return True
        t_low = text.lower()
        refusal_phrases = [
            "sorry", "cannot browse", "can't browse", "unable to browse", 
            "not able to browse", "large language model", "training data", 
            "cutoff date", "as an ai", "i am an ai", "don't have access",
            "not sure what you are asking", "clarify your question"
        ]
        return any(phrase in t_low for phrase in refusal_phrases)

    def generate_scene_caption(self, keyframes: List[Image.Image]) -> str:
        """
        Generate dense action and interaction caption for a sequence of keyframes in a scene using MiniCPM-V 2.6.
        """
        self._lazy_load()
        if not keyframes:
            return ""

        fallback_caption = f"Scene showing keyframe visuals with {len(keyframes)} frames, subjects and ongoing activity."

        if self.model is None:
            return fallback_caption

        try:
            # Subsample up to 3 representative frames for MiniCPM-V 2.6
            sample_count = min(3, len(keyframes))
            step = len(keyframes) // sample_count if sample_count > 0 else 1
            sampled = [keyframes[i] for i in range(0, len(keyframes), max(1, step))][:sample_count]

            prompt_text = "Describe what is happening in this video scene concisely in 1-2 sentences. Focus on subjects, clothing colors, actions, and objects."
            
            # MiniCPM-V 2.6 multi-modal input format
            msgs = [{
                "role": "user",
                "content": (*sampled, prompt_text)
            }]

            with torch.no_grad():
                res = self.model.chat(
                    image=None, 
                    msgs=msgs, 
                    tokenizer=self.tokenizer,
                    system_prompt="You are a computer vision video analysis model. Directly describe the visible people, clothing, colors, and actions in the given images."
                )
                caption_str = str(res).strip()
                
                # Check for refusal responses
                if self._is_refusal_response(caption_str):
                    logger.warning(f"MiniCPM-V returned refusal/canned response. Discarding: {caption_str[:60]}...")
                    return fallback_caption
                
                return caption_str

        except Exception as ex:
            logger.error(f"MiniCPM-V 2.6 captioning error ({ex}); attempting single-frame fallback...")
            try:
                # Single frame fallback
                if sampled:
                    with torch.no_grad():
                        res = self.model.chat(
                            image=sampled[0],
                            msgs=[{"role": "user", "content": "Describe this video frame concisely in 1 sentence focusing on subjects, clothing colors, and actions."}],
                            tokenizer=self.tokenizer
                        )
                        caption_str = str(res).strip()
                        if not self._is_refusal_response(caption_str):
                            return caption_str
            except Exception as single_err:
                logger.debug(f"Single frame caption fallback failed: {single_err}")

            return fallback_caption
