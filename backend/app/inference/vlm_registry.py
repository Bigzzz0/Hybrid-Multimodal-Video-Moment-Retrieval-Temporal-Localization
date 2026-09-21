"""Allow-listed VLM experiment registry.

The registry is deliberately data-only: importing it never downloads a model.
Keeping model identity and revision in one place prevents cache/artifact
contamination between the A--E ablation variants.
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
    adapter_id: str = ""
    adapter_revision: str = ""
    research_only: bool = False
    enabled: bool = True

    def public_dict(self) -> dict:
        data = asdict(self)
        data["adapter_id"] = self.adapter_id or None
        data["model_file"] = self.model_file or None
        data["mmproj_file"] = self.mmproj_file or None
        return data


_VARIANTS: Dict[str, VLMVariant] = {
    "qwen3_vl_2b": VLMVariant(
        backend="qwen3_vl_2b",
        label="A  Qwen3-VL-2B",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        revision="89644892e4d85e24eaac8bacfd4f463576704203",
        runtime="transformers_nf4",
        input_mode="frames",
        quantization="NF4",
    ),
    "qwen3_vl_2b_vise": VLMVariant(
        backend="qwen3_vl_2b_vise",
        label="B  Qwen3-VL-2B + VISE",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        revision="89644892e4d85e24eaac8bacfd4f463576704203",
        runtime="transformers_nf4_peft",
        input_mode="frames",
        quantization="NF4",
        adapter_id="shravvvv/VISE",
        adapter_revision="2fc58ecbd78ba5668d7a70e4fc868fcbb2b62edd",
    ),
    "caprl_qwen3vl_2b": VLMVariant(
        backend="caprl_qwen3vl_2b",
        label="C  CapRL-Qwen3VL-2B",
        model_id="internlm/CapRL-Qwen3VL-2B",
        revision="b838cec8b6c4791ed6169d12db21db0b47bb70e5",
        runtime="transformers_nf4",
        input_mode="frames",
        quantization="NF4",
        research_only=True,
    ),
    "caprl_qwen3vl_4b_q4": VLMVariant(
        backend="caprl_qwen3vl_4b_q4",
        label="D  CapRL-Qwen3VL-4B Q4",
        model_id="internlm/CapRL-Qwen3VL-4B-GGUF",
        revision="922d08bb6257875336aa138616c74902f736099c",
        runtime="llama_cpp",
        input_mode="frames",
        quantization="Q4_K_M",
        model_file="CapRL-Qwen3VL-4B-Q4_K_M.gguf",
        mmproj_file="CapRL-Qwen3VL-4B-mmproj-Q8_0.gguf",
        research_only=True,
    ),
    "caprl_qwen3vl_4b_q6": VLMVariant(
        backend="caprl_qwen3vl_4b_q6",
        label="D  CapRL-Qwen3VL-4B Q6",
        model_id="internlm/CapRL-Qwen3VL-4B-GGUF",
        revision="922d08bb6257875336aa138616c74902f736099c",
        runtime="llama_cpp",
        input_mode="frames",
        quantization="Q6_K",
        model_file="CapRL-Qwen3VL-4B-q6_k.gguf",
        mmproj_file="CapRL-Qwen3VL-4B-mmproj-Q8_0.gguf",
        research_only=True,
    ),
    "caprl_video_4b": VLMVariant(
        backend="caprl_video_4b",
        label="E  CapRL-Video-4B",
        model_id="internlm/CapRL-Video-4B",
        revision="d932a039f2a7f580abf9d9e07dabe60a18eb1960",
        runtime="transformers_nf4",
        input_mode="video_chunk",
        quantization="NF4",
        research_only=True,
    ),
}


def all_variants() -> List[VLMVariant]:
    return list(_VARIANTS.values())


def get_variant(backend: Optional[str]) -> VLMVariant:
    key = (backend or "qwen3_vl_2b").strip().lower()
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

