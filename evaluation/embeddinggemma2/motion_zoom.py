"""Label-free temporal detail: find a two-second interval with most motion.
Motion is a sampling cue, never evidence that an abnormal event occurred.
"""
import math

import cv2
import numpy as np


def choose_motion_span(timestamps, energies, start, end, seconds=2.):
    stamps, values = np.asarray(timestamps, dtype=float), np.asarray(energies, dtype=float)
    if (end <= start or not np.isfinite([start, end, seconds]).all() or seconds <= 0 or
            stamps.shape != values.shape or stamps.ndim != 1 or
            not np.isfinite(stamps).all() or not np.isfinite(values).all() or
            np.any(np.diff(stamps) <= 0) or np.any(values < 0) or
            np.any(stamps < start) or np.any(stamps >= end)):
        raise ValueError("Invalid motion trace")
    if len(stamps) < 2 or values.max(initial=0.) < 1.:
        return None
    span = min(seconds, end-start)
    step = float(np.median(np.diff(stamps)))
    count = min(len(stamps), max(1, math.ceil(span/step)))
    scores = np.convolve(values, np.ones(count), mode="valid")
    # Fixed central tie-break, never source labels or target timestamps.
    centers = stamps[:len(scores)]+(count-1)*step/2
    best = max(range(len(scores)), key=lambda i: (scores[i], -abs(centers[i]-(start+end)/2), -i))
    left = float(np.clip(centers[best]-span/2, start, end-span))
    return [left, left+span]


def motion_trace(path, start, end):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Cannot decode motion source")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid motion FPS")
        first = math.ceil(start*fps); stop = math.ceil(end*fps)
        step = max(1, round(fps/8.))
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        previous = None; stamps, values = [], []
        for index in range(first, stop):
            if not capture.grab():
                break
            if (index-first) % step:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError("Motion frame decode failed")
            gray = cv2.cvtColor(cv2.resize(frame, (160, 108)), cv2.COLOR_BGR2GRAY).astype(float)
            if previous is None:
                energy = 0.
            else:
                difference = np.abs(gray-previous)
                # Suppress global brightness changes, emphasize local motion.
                difference = np.maximum(0., difference-np.median(difference))
                energy = float(np.sort(difference.ravel())[-max(1, difference.size//20):].mean())
            stamps.append(index/fps); values.append(energy); previous = gray
        return dict(timestamps=stamps, energies=values, source_fps=fps,
                    sampled_fps=fps/step, canvas=[160, 108], cue="top-five-percent-absolute-gray-difference-v1")
    finally:
        capture.release()
