from __future__ import annotations

import base64
from contextlib import nullcontext
import json
import inspect
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import settings
from app.inference.contracts import GroundObservation, GroundRequest, GroundResponse, GroundTrack


def _to_numpy(value: Any):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value


def _rle(mask: Any) -> Dict[str, Any]:
    """Small JSON-safe row-major RLE; frontend can decode without pycocotools."""
    import numpy as np
    array = np.asarray(_to_numpy(mask)).astype(bool)
    if array.ndim > 2:
        array = np.squeeze(array)
    flat = array.flatten(order="C").astype(np.uint8)
    counts: List[int] = []
    current = 0
    run = 0
    for item in flat:
        if int(item) == current:
            run += 1
        else:
            counts.append(run)
            current = int(item)
            run = 1
    counts.append(run)
    return {"size": [int(array.shape[0]), int(array.shape[1])], "counts": counts, "order": "C"}


def _start_sam31_session(
    predictor: Any,
    resource_path: str,
    session_id: str | None = None,
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
) -> Dict[str, str]:
    """Start a SAM 3.1 session across official API minor-version changes.

    The current SAM 3.1 multiplex model does not accept
    ``offload_state_to_cpu`` even though the shared request dispatcher still
    forwards that legacy keyword. Filter init kwargs against the actual model
    signature and register the state using the predictor's normal session map.
    """
    init_kwargs = {
        "resource_path": resource_path,
        "offload_video_to_cpu": offload_video_to_cpu,
        "offload_state_to_cpu": offload_state_to_cpu,
    }
    if hasattr(predictor, "async_loading_frames"):
        init_kwargs["async_loading_frames"] = predictor.async_loading_frames
    if hasattr(predictor, "video_loader_type"):
        init_kwargs["video_loader_type"] = predictor.video_loader_type
    valid = set(inspect.signature(predictor.model.init_state).parameters)
    inference_state = predictor.model.init_state(
        **{key: value for key, value in init_kwargs.items() if key in valid}
    )
    session_id = session_id or str(uuid.uuid4())
    predictor._all_inference_states[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": time.time(),
        "last_use_time": time.time(),
    }
    return {"session_id": session_id}


class SAM31Grounder:
    def __init__(self) -> None:
        self.predictor = None

    def load(self) -> None:
        import torch
        from sam3.model_builder import build_sam3_predictor
        import sam3.model.decoder as sam_decoder

        # RTX 5070/SM120 with the current Windows PyTorch build may expose
        # neither the flash nor memory-efficient SDPA kernels used by the
        # upstream defaults. Keep SAM usable with the portable math backend.
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(False)
            # SAM's decoder explicitly requests FLASH_ATTENTION in its
            # functional_attention helper, which is unavailable in this
            # Windows/SM120 build. Redirect that local context to MATH.
            sam_decoder.sdpa_kernel = lambda _backend: torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.MATH
            )

        # SAM 3.1 is exposed by the official package through the multiplex
        # video predictor.  The image builder downloads the SAM 3 checkpoint,
        # so using it here would silently run the wrong model.
        self.predictor = build_sam3_predictor(
            version="sam3.1",
            compile=settings.SAM_COMPILE,
            warm_up=False,
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,  # FlashAttention 3 is not available on this Windows setup.
            use_rope_real=True,
            async_loading_frames=False,
        )
        # The shared SAM dispatcher still forwards offload_state_to_cpu, but
        # the current SAM 3.1 multiplex model removed that argument.
        self.predictor.start_session = lambda **kwargs: _start_sam31_session(
            self.predictor, **kwargs
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def unload(self) -> None:
        self.predictor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _inference_context():
        """Use the mixed-precision context expected by the official SAM3 code.

        SAM3.1 prepares video tensors as float16, while parts of the tracker
        produce bfloat16 activations.  Without the official CUDA autocast
        context, a few Linear layers receive bfloat16 inputs with float32
        weights and fail with ``mat1 and mat2 must have the same dtype``.
        """
        import torch

        if torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _best_observation(output: Dict[str, Any], image_width: int, image_height: int):
        """Convert an official SAM 3.1 frame output to one best observation."""
        import numpy as np

        probs = np.asarray(_to_numpy(output.get("out_probs", []))).reshape(-1)
        masks = np.asarray(_to_numpy(output.get("out_binary_masks", [])))
        boxes = np.asarray(_to_numpy(output.get("out_boxes_xywh", [])))
        if probs.size == 0 or masks.size == 0 or boxes.size == 0:
            return None
        best = int(probs.argmax())
        if best >= len(masks) or best >= len(boxes):
            return None
        x, y, width, height = [float(value) for value in boxes[best][:4]]
        # SAM 3.1 video outputs use normalized xywh coordinates.
        if max(abs(x), abs(y), abs(width), abs(height)) <= 1.5:
            x *= image_width
            y *= image_height
            width *= image_width
            height *= image_height
        return {
            "bbox_xyxy": [x, y, x + width, y + height],
            "score": float(probs[best]),
            "mask": masks[best],
        }

    def ground(self, request: GroundRequest) -> GroundResponse:
        from PIL import Image

        paths = list(request.frame_paths)
        timestamps = list(request.timestamps)
        if self.predictor is None:
            raise RuntimeError("SAM 3.1 predictor is not loaded")
        if not paths:
            raise ValueError("SAM 3.1 worker requires sampled frame_paths")
        if len(timestamps) != len(paths):
            timestamps = [float(index) for index in range(len(paths))]

        tracks: List[GroundTrack] = []
        selected_paths = paths[: request.max_frames]
        selected_timestamps = timestamps[: request.max_frames]
        with tempfile.TemporaryDirectory(prefix="sam31_frames_") as frame_dir:
            frame_dir_path = Path(frame_dir)
            sizes = []
            for index, path in enumerate(selected_paths):
                target = frame_dir_path / f"{index:08d}.jpg"
                shutil.copyfile(path, target)
                with Image.open(target) as image:
                    sizes.append((image.width, image.height))

            for prompt in request.prompts:
                session_id = None
                frame_outputs: Dict[int, Dict[str, Any]] = {}
                try:
                    # The upstream qualitative/video examples enter this
                    # context before building and running the predictor. Keep
                    # it active while consuming the propagation generator.
                    with self._inference_context():
                        session = self.predictor.handle_request({
                            "type": "start_session",
                            "resource_path": str(frame_dir_path),
                            "offload_video_to_cpu": True,
                        })
                        session_id = session["session_id"]
                        first = self.predictor.handle_request({
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": prompt,
                        })
                        frame_outputs[0] = first.get("outputs", {})
                        for item in self.predictor.handle_stream_request({
                            "type": "propagate_in_video",
                            "session_id": session_id,
                            "propagation_direction": "forward",
                            "start_frame_index": 0,
                            "max_frame_num_to_track": len(selected_paths),
                        }):
                            frame_outputs[int(item["frame_index"])] = item.get("outputs", {})
                finally:
                    if session_id is not None:
                        self.predictor.handle_request({
                            "type": "close_session",
                            "session_id": session_id,
                            "run_gc_collect": False,
                        })

                observations: List[GroundObservation] = []
                for index in sorted(frame_outputs):
                    if index >= len(selected_paths):
                        continue
                    best = self._best_observation(frame_outputs[index], *sizes[index])
                    if best is None:
                        continue
                    observations.append(GroundObservation(
                        timestamp=float(selected_timestamps[index]),
                        bbox_xyxy=best["bbox_xyxy"],
                        score=best["score"],
                        mask_rle=_rle(best["mask"]),
                        frame_width=int(sizes[index][0]),
                        frame_height=int(sizes[index][1]),
                    ))
                if observations:
                    scores = [item.score for item in observations]
                    tracks.append(GroundTrack(
                        track_id=f"{uuid.uuid4()}-{prompt}",
                        concept=prompt,
                        t_start=min(item.timestamp for item in observations),
                        t_end=max(item.timestamp for item in observations),
                        mean_score=sum(scores) / len(scores),
                        max_score=max(scores),
                        observations=observations,
                    ))
        return GroundResponse(
            tracks=tracks,
            model_id=settings.SAM_MODEL_ID,
            grounding_version=settings.GROUNDING_VERSION,
            status="generated",
        )
