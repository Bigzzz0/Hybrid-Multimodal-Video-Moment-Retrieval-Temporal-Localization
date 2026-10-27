"""Label-free, CPU-only person-region proposals for a verification ablation.

One fixed padded union across the whole frame sequence avoids crop jitter.
This uses OpenCV's bundled pretrained HOG pedestrian detector, not annotation
boxes. A crop includes all detected boxes; missed people remain a limitation.
"""
import hashlib
import math
import time
from pathlib import Path

import cv2
import numpy as np


def filter_person_boxes(prediction, width, height, threshold=.5):
    boxes, scores = [], []
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if label != 1 or not np.isfinite(score) or score < threshold:
            continue
        if not np.isfinite(box).all():
            raise ValueError("Nonfinite detector box")
        x1, y1, x2, y2 = box
        x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
        x2, y2 = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2-x1, y2-y1]); scores.append(float(score))
    return boxes, scores


class SSDLitePersonDetector:
    """Pinned pretrained COCO weights, CPU FP32; never loads a GPU model."""
    def __init__(self, cache):
        import torch
        import torchvision
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
        started = time.perf_counter()
        torch.set_num_threads(4); torch.hub.set_dir(str(Path(cache).resolve()))
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.model = ssdlite320_mobilenet_v3_large(weights=weights, progress=False).cpu().float().eval()
        path = Path(torch.hub.get_dir())/"checkpoints"/Path(weights.url).name
        self.metadata = dict(method="ssdlite320-coco-person-padded-union-v1", device="cpu", dtype="float32",
            torchvision_version=torchvision.__version__, torch_version=torch.__version__,
            detector_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), weights_url=weights.url,
            settings=dict(class_id=1, score_threshold=.5, cpu_threads=4),
            cold_load_seconds=time.perf_counter()-started)

    def detectMultiScale(self, frame, **kwargs):
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()/255.
        with torch.inference_mode():
            prediction = self.model([tensor])[0]
        values = {key: value.detach().cpu().numpy() for key, value in prediction.items()}
        return filter_person_boxes(values, frame.shape[1], frame.shape[0])


def region_for_detections(detections, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Invalid canvas")
    full = [0, 0, width, height]
    boxes = [box for frame in detections for box in frame]
    if sum(bool(frame) for frame in detections) < 2:
        return full, "insufficient_detection_frames"
    for box in boxes:
        if (len(box) != 4 or not np.isfinite(box).all() or box[2] <= 0 or box[3] <= 0 or
                box[0] < 0 or box[1] < 0 or box[0]+box[2] > width or box[1]+box[3] > height):
            raise ValueError("Invalid detector box")
    left = min(box[0] for box in boxes); top = min(box[1] for box in boxes)
    right = max(box[0]+box[2] for box in boxes); bottom = max(box[1]+box[3] for box in boxes)
    cx, cy = (left+right)/2, (top+bottom)/2
    # Twenty percent context on each side; at least half the original canvas.
    rw = max((right-left)*1.4, width*.5)
    rh = max((bottom-top)*1.4, height*.5)
    rw = max(rw, rh*width/height); rh = rw*height/width
    if rw >= width or rh >= height or rw*rh >= .95*width*height:
        return full, "region_too_wide"
    x = max(0., min(cx-rw/2, width-rw)); y = max(0., min(cy-rh/2, height-rh))
    rect = [math.floor(x), math.floor(y), math.ceil(x+rw), math.ceil(y+rh)]
    if not all(rect[0] <= b[0] and rect[1] <= b[1] and rect[2] >= b[0]+b[2] and rect[3] >= b[1]+b[3] for b in boxes):
        return full, "context_does_not_fit"
    return rect, None


def prepare_region(paths, output, detector=None):
    started = time.perf_counter(); frames = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError("Unreadable verification frame: " + str(path))
        frames.append(frame)
    if not frames or any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("Region frames must share a canvas")
    height, width = frames[0].shape[:2]
    weights = cv2.HOGDescriptor_getDefaultPeopleDetector()
    if detector is None:
        detector = cv2.HOGDescriptor(); detector.setSVMDetector(weights)
    boxes = []
    for frame in frames:
        found = [] if width < 64 or height < 128 else detector.detectMultiScale(
            frame, hitThreshold=0., winStride=(8, 8), padding=(8, 8), scale=1.05)[0]
        boxes.append([list(map(int, box)) for box in found])
    rect, fallback = region_for_detections(boxes, width, height)
    result_paths = list(map(str, paths))
    if fallback is None:
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        x1, y1, x2, y2 = rect; result_paths = []
        for index, frame in enumerate(frames):
            # Match the original input canvas and preserve aspect approximately
            # (integer box rounding can differ by a pixel). No generative upscale.
            crop = cv2.resize(frame[y1:y2, x1:x2], (width, height), interpolation=cv2.INTER_CUBIC)
            target = output/f"frame-{index:02}.png"
            if not cv2.imwrite(str(target), crop):
                raise OSError("Unable to write crop")
            result_paths.append(str(target.resolve()))
    metadata = getattr(detector, "metadata", dict(method="opencv-pretrained-hog-padded-union-v1",
        detector_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
        settings=dict(hitThreshold=0., winStride=[8, 8], padding=[8, 8], scale=1.05)))
    return result_paths, dict(**metadata, cv2_version=cv2.__version__,
        region_settings=dict(minimum_canvas_fraction=.5, context_each_side=.2),
        detections=boxes, rect=rect, canvas=[width, height], fallback=fallback,
        frame_count=len(frames), elapsed_seconds=time.perf_counter()-started)
