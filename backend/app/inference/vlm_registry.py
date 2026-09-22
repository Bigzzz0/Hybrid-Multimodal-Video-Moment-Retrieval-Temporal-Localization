"""Allow-listed production VLM runtimes.

The production path intentionally exposes only the Q6 captioner and the
small Qwen verifier. The A--E research variants remain on the ablation branch.
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
    "qwen3_vl_2b": VLMVariant(
        backend="qwen3_vl_2b",
        label="Qwen3-VL-2B",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        revision="89644892e4d85e24eaac8bacfd4f463576704203",
        runtime="transformers_nf4",
        input_mode="frames",
        quantization="NF4",
    ),
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
