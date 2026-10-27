"""Isolated CPU FP32 AVA action evidence. Never a production verdict/probability.

Install optional dependencies into scratch/action-expert-tools, not the workers.
Full RGB clip + central-frame detected people; annotations never enter inference.
"""
import hashlib
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CHECKPOINT_SHA256 = "9bd6584a8212f91e4c7d96ec7d6bff31250e1c810fcee968471614a1b9658ac8"
AVA_IDS = {"fall": 5, "fight_hit_person": 64, "handshake": 68,
           "hug": 70, "push_person": 76}


def sample_indices(start, end, fps, frame_count):
    if not np.isfinite([start, end, fps]).all() or start < 0 or end <= start or fps <= 0:
        raise ValueError("Invalid clip interval/FPS")
    first, stop = math.ceil(start*fps), min(math.ceil(end*fps), frame_count)
    if stop-first < 32:
        raise ValueError("insufficient_unique_frames")
    indices = np.linspace(first, stop-1, 32).astype(np.int64)
    if len(np.unique(indices)) != 32:
        raise ValueError("insufficient_unique_frames")
    return indices


def decode_clip(path, start, end):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot decode action clip")
        fps = cap.get(cv2.CAP_PROP_FPS)
        indices = sample_indices(start, end, fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        wanted = set(indices.tolist()); frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
        for index in range(int(indices[0]), int(indices[-1])+1):
            if not cap.grab():
                raise ValueError("Incomplete action clip")
            if index in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError("Unreadable action frame")
                frames.append(frame)
        return np.stack(frames), (indices/fps).tolist()
    finally:
        cap.release()


def inputs(frames, xywh):
    import torch
    import torch.nn.functional as functional
    if frames.ndim != 4 or frames.shape[0] != 32 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("Expected 32 BGR uint8 frames")
    height, width = frames.shape[1:3]
    boxes = np.asarray(xywh, dtype=np.float32).reshape(-1, 4).copy()
    if not len(boxes) or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= 0):
        raise ValueError("No valid people")
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width-1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height-1)
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Degenerate clipped person box")
    # Match official short-side bilinear transform and corresponding ROI scale.
    scale = 256/min(height, width)
    out_h, out_w = int(math.floor(height*scale)), int(math.floor(width*scale))
    tensor = torch.from_numpy(frames[..., ::-1].copy()).permute(0, 3, 1, 2).float()/255.
    tensor = functional.interpolate(tensor, size=(out_h, out_w), mode="bilinear", align_corners=False)
    tensor = ((tensor-.45)/.225).permute(1, 0, 2, 3)
    boxes *= scale
    rois = torch.cat([torch.zeros(len(boxes), 1), torch.from_numpy(boxes)], dim=1)
    slow = tensor[:, torch.linspace(0, 31, 8).long()]
    return [slow.unsqueeze(0), tensor.unsqueeze(0)], rois


def action_scores(predictions):
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or predictions.shape[1] != 80 or not len(predictions) or not np.isfinite(predictions).all():
        raise ValueError("Invalid AVA prediction")
    if np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("AVA head must already apply sigmoid")
    selected = {name: float(predictions[:, identifier-1].max()) for name, identifier in AVA_IDS.items()}
    selected["target"] = max(selected["fight_hit_person"], selected["push_person"])
    selected["normal"] = max(selected["handshake"], selected["hug"])
    selected["margin"] = selected["target"]-selected["normal"]
    return selected


class ActionExpert:
    def __init__(self, checkpoint, detector_cache, tools="scratch/action-expert-tools"):
        started = time.perf_counter()
        checkpoint = Path(checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError("Unexpected action checkpoint SHA256")
        sys.path.insert(0, str(Path(tools).resolve()))
        import torch
        import torchvision
        from pytorchvideo.models.hub import slowfast_r50_detection
        from .person_region import SSDLitePersonDetector
        torch.set_num_threads(4)
        self.detector = SSDLitePersonDetector(detector_cache)
        self.model = slowfast_r50_detection(pretrained=False).cpu().float().eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model_state"], strict=True)
        self.metadata = dict(checkpoint_sha256=digest, torch=torch.__version__, torchvision=torchvision.__version__,
            pytorchvideo="0.1.5", device="cpu", dtype="float32", threads=4, slow_frames=8, fast_frames=32,
            short_side=256, mean=.45, std=.225, ava_ids=AVA_IDS, detector=self.detector.metadata,
            load_seconds=time.perf_counter()-started)

    def predict(self, path, bounds):
        import torch
        started = time.perf_counter()
        frames, stamps = decode_clip(path, *bounds)
        boxes, detector_scores = self.detector.detectMultiScale(frames[16])
        evidence = dict(bounds=bounds, timestamps=stamps, boxes=boxes, detector_scores=detector_scores,
                        center_timestamp=stamps[16], status="unverified")
        if not boxes:
            return dict(evidence, reason="no_person_detected", elapsed_seconds=time.perf_counter()-started)
        pathways, rois = inputs(frames, boxes)
        with torch.inference_mode():
            predictions = self.model(pathways, rois).cpu().numpy()
        return dict(evidence, status="scored", scores=action_scores(predictions),
                    person_predictions=predictions.tolist(), elapsed_seconds=time.perf_counter()-started)
