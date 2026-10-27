import math
import shutil
import subprocess
import wave as wavfile
from pathlib import Path
import numpy as np

def windows(duration, size=6.0, stride=3.0):
    if duration <= 0 or size <= 0 or stride <= 0:
        raise ValueError("Invalid timeline")
    return [(round(start, 6), round(min(start + size, duration), 6))
            for start in np.arange(0, duration, stride)]

def metadata(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps <= 0 or count <= 0:
            raise ValueError("Video cannot be decoded")
        return {"duration": count / fps, "fps": fps}
    finally:
        capture.release()

def ffmpeg():
    binary = shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError("ffmpeg is required for synchronized 16 kHz audio")
    return binary

def has_audio(path):
    # Missing tooling is an error, not an assertion that the video has no audio.
    probe = shutil.which("ffprobe")
    if not probe:
        raise RuntimeError("ffprobe is required")
    result = subprocess.run([probe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return bool(result.stdout.strip())

def extract(path, start, end, folder, config, audio_available):
    import cv2
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    times = np.arange(start, end - 1e-7, 1 / config.video_fps).tolist()
    if not times:
        times = [start]
    if len(times) > config.max_frames:
        raise ValueError("Frame budget exceeded")
    capture = cv2.VideoCapture(str(path))
    frame_paths = []
    actual_times, tail_holds = [], []
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        for index, stamp in enumerate(times):
            capture.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000)
            valid, frame = capture.read()
            held = False
            if not valid:
                # Containers can end fractionally after the last coded frame.
                # Hold that last frame only within one source-frame interval;
                # retain the actual timestamp and explicit fallback evidence.
                if source_fps > 0 and source_count > 0 and 0 <= source_count/source_fps-stamp <= 1/source_fps + .02:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, source_count-1)
                    valid, frame = capture.read()
                    held = valid
                if not valid:
                    raise ValueError(f"Corrupt/missing frame at {stamp:.3f}s")
            actual = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if actual <= 0 and stamp > 0:
                actual = (source_count-1)/source_fps if held else stamp
            actual_times.append(actual)
            tail_holds.append(held)
            target = folder / f"frame-{index:02}.jpg"
            if not cv2.imwrite(str(target), frame):
                raise OSError("Unable to write frame")
            frame_paths.append(str(target.resolve()))
    finally:
        capture.release()
    wave = None
    if audio_available:
        target = folder / "audio.wav"
        subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start), "-i", str(path), "-t", str(end-start), "-vn", "-ac", "1", "-ar", "16000", str(target)], capture_output=True, check=True)
        with wavfile.open(str(target), "rb") as stream:
            rate = stream.getframerate()
            if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
                raise ValueError("Expected PCM16 mono WAV")
            data = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32)/32768
        expected = round((end-start) * 16000)
        if rate != 16000 or data.ndim != 1 or not len(data) or abs(len(data) - expected) > 1600 or not np.isfinite(data).all():
            raise ValueError("Audio alignment/sample-rate validation failed")
        wave = str(target.resolve())
    # At default 2 fps, every second chronological frame is the 1 fps image index.
    image_slots = [i for i, stamp in enumerate(times) if abs((stamp-start)*config.image_fps - round((stamp-start)*config.image_fps)) < 1e-5]
    rms = float(np.sqrt(np.mean(data**2))) if wave else None
    return {"t_start": start, "t_end": end, "frame_paths": frame_paths,
            "timestamps": actual_times, "requested_timestamps": times, "held_last_frame": tail_holds,
            "image_slots": image_slots, "audio_path": wave,
            "audio_status": ("silent" if rms < 1e-5 else "available") if wave else "no_audio", "audio_rms": rms, "frame_count": len(times),
            "video_tokens": config.video_tokens, "image_tokens": config.image_tokens}
