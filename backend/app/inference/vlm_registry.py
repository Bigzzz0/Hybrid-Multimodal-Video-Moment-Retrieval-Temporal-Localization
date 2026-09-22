"""Allow-listed production VLM runtime.

Production uses one local multimodal model for both stored captions and
Accurate verification. The old Qwen3-VL-2B runtime is intentionally not
registered here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class VLMVariant:
    backend: str
    label: str
    model_id: str
    revision: str
    runtime: str
    input_mode: str
    quantization: str
    model_file: str = ""
    mmproj_file: str = ""
    research_only: bool = False
    enabled: bool = True

    def public_dict(self) -> dict:
        data = asdict(self)
        data["model_file"] = self.model_file or None
        data["mmproj_file"] = self.mmproj_file or None
        return data


_VARIANTS: Dict[str, VLMVariant] = {
    "caprl_qwen3vl_4b_q6": VLMVariant(
        backend="caprl_qwen3vl_4b_q6",
        label="CapRL-Qwen3VL-4B Q6",
        model_id="internlm/CapRL-Qwen3VL-4B-GGUF",
        revision="922d08bb6257875336aa138616c74902f736099c",
        runtime="llama_cpp",
        input_mode="frames",
        quantization="Q6_K",
        model_file="CapRL-Qwen3VL-4B-q6_k.gguf",
        mmproj_file="CapRL-Qwen3VL-4B-mmproj-Q8_0.gguf",
        research_only=True,
    ),
}


def all_variants() -> List[VLMVariant]:
    return list(_VARIANTS.values())


def get_variant(backend: Optional[str]) -> VLMVariant:
    key = (backend or "caprl_qwen3vl_4b_q6").strip().lower()
    if key not in _VARIANTS:
        raise ValueError(f"unsupported VLM backend: {backend}")
    variant = _VARIANTS[key]
    if not variant.enabled:
        raise ValueError(f"VLM backend is disabled: {key}")
    return variant


def is_supported_backend(backend: Optional[str]) -> bool:
    try:
        get_variant(backend)
        return True
    except ValueError:
        return False
