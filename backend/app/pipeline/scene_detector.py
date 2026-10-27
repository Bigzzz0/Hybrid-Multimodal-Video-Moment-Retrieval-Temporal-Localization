import time
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import AdaptiveDetector, ContentDetector
from app.core.logger import logger


class ProgressAdaptiveDetector(AdaptiveDetector):
    """AdaptiveDetector wrapper that tracks frame-level progress and emits live telemetry."""

    def __init__(
        self,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        total_frames: int = 1,
        duration_sec: float = 1.0,
        resolution: str = "Unknown",
        video_fps: float = 30.0,
        frame_skip: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.on_progress = on_progress
        self.total_frames = max(1, total_frames)
        self.duration_sec = duration_sec
        self.resolution = resolution
        self.video_fps = video_fps
        self.frame_skip = frame_skip
        self.processed_frames = 0
        self.cuts_found = 0
        self.start_time = time.monotonic()
        self.last_report_time = 0.0

    def process_frame(self, timecode: FrameTimecode, frame_img: np.ndarray) -> list[FrameTimecode]:
        cuts = super().process_frame(timecode=timecode, frame_img=frame_img)
        if cuts:
            self.cuts_found += len(cuts)
        self.processed_frames += (1 + self.frame_skip)

        now = time.monotonic()
        # Report every 0.35 seconds or when hitting the final frame
        curr_frame = timecode.frame_num if hasattr(timecode, "frame_num") else self.processed_frames
        if self.on_progress and (now - self.last_report_time >= 0.35 or curr_frame >= self.total_frames):
            self.last_report_time = now
            elapsed = max(0.001, now - self.start_time)
            speed_fps = round(self.processed_frames / elapsed, 1)

            curr_sec = float(timecode.seconds) if hasattr(timecode, "seconds") else (curr_frame / max(1.0, self.video_fps))
            pct = min(100, max(0, int(round((curr_frame / self.total_frames) * 100))))

            self.on_progress({
                "sub_percent": pct,
                "current_frame": curr_frame,
                "total_frames": self.total_frames,
                "current_sec": round(curr_sec, 1),
                "duration_sec": round(self.duration_sec, 1),
                "cuts_found": self.cuts_found,
                "speed_fps": speed_fps,
                "resolution": self.resolution,
                "fps": self.video_fps,
            })
        return cuts


class AdaptiveSceneDetector:
    """Content-aware adaptive scene boundary detector with live progress telemetry."""

    def __init__(
        self,
        adaptive_threshold: float = 3.0,
        min_scene_len_sec: float = 1.5,
        max_scene_len_sec: float = 30.0,
    ):
        self.adaptive_threshold = adaptive_threshold
        self.min_scene_len_sec = min_scene_len_sec
        self.max_scene_len_sec = max_scene_len_sec

    def detect_scenes(
        self,
        video_path: str,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        resolution: str = "Unknown",
        fps: float = 30.0,
        duration_sec: float = 0.0,
        total_frames: int = 0,
    ) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries with real-time progress callbacks.
        Returns a list of tuples: [(t_start_sec, t_end_sec), ...]
        """
        logger.info(f"Starting Scene Detection for: {video_path}")
        start_time = time.monotonic()
        try:
            video = open_video(video_path)
            actual_fps = float(video.frame_rate) if video.frame_rate else fps
            actual_duration = float(video.duration.seconds) if (video.duration and hasattr(video.duration, "seconds")) else duration_sec
            actual_total_frames = int(video.duration.frame_num) if (video.duration and hasattr(video.duration, "frame_num")) else (total_frames or int(actual_duration * actual_fps))

            # Adaptive frame skipping: doubles/triples speed on longer videos without losing cut detection accuracy
            if actual_duration >= 600.0:
                frame_skip = 2  # 3x speed on videos >= 10 minutes (~10 fps effective sampling)
            elif actual_duration >= 120.0:
                frame_skip = 1  # 2x speed on videos 2 - 10 minutes (~15 fps effective sampling)
            else:
                frame_skip = 0  # Frame-accurate for short clips < 2 minutes

            effective_min_scene_len = max(1, int((actual_fps * self.min_scene_len_sec) / (1 + frame_skip)))

            detector = ProgressAdaptiveDetector(
                on_progress=on_progress,
                total_frames=actual_total_frames,
                duration_sec=actual_duration,
                resolution=resolution,
                video_fps=actual_fps,
                frame_skip=frame_skip,
                adaptive_threshold=self.adaptive_threshold,
                min_scene_len=effective_min_scene_len,
            )

            scene_manager = SceneManager()
            scene_manager.add_detector(detector)
            scene_manager.detect_scenes(video=video, frame_skip=frame_skip)
            scenes = scene_manager.get_scene_list()

            if not scenes:
                # Fallback: Single scene if no cuts detected
                dur = float(video.duration.seconds) if (video.duration and hasattr(video.duration, "seconds")) else actual_duration
                logger.info(f"No scene cuts found. Using continuous timeline [0.0, {dur:.2f}]")
                scene_intervals = [(0.0, dur)]
            else:
                scene_intervals = [
                    (
                        float(scene[0].seconds if hasattr(scene[0], "seconds") else scene[0].get_seconds()),
                        float(scene[1].seconds if hasattr(scene[1], "seconds") else scene[1].get_seconds()),
                    )
                    for scene in scenes
                ]

            # 1. Merge or discard tiny ending/glitch remnants (e.g. black screen fade-out < min_scene_len_sec)
            while len(scene_intervals) > 1 and (scene_intervals[-1][1] - scene_intervals[-1][0]) < self.min_scene_len_sec:
                tail = scene_intervals.pop()
                prev = scene_intervals[-1]
                scene_intervals[-1] = (prev[0], tail[1])

            # 2. Subdivide continuous long scenes into logical temporal chunks <= max_scene_len_sec
            # Essential for CCTV, surveillance, dashcam, and single-take continuous recordings
            refined_scenes: List[Tuple[float, float]] = []
            for t_start, t_end in scene_intervals:
                scene_dur = t_end - t_start
                if scene_dur <= self.max_scene_len_sec:
                    refined_scenes.append((round(t_start, 2), round(t_end, 2)))
                else:
                    num_chunks = int(np.ceil(scene_dur / self.max_scene_len_sec))
                    chunk_dur = scene_dur / num_chunks
                    for i in range(num_chunks):
                        c_start = t_start + i * chunk_dur
                        c_end = t_start + (i + 1) * chunk_dur if i < num_chunks - 1 else t_end
                        refined_scenes.append((round(c_start, 2), round(c_end, 2)))

            scene_intervals = refined_scenes

            if on_progress:
                elapsed = max(0.001, time.monotonic() - start_time)
                on_progress({
                    "sub_percent": 100,
                    "current_frame": actual_total_frames,
                    "total_frames": actual_total_frames,
                    "current_sec": round(actual_duration, 1),
                    "duration_sec": round(actual_duration, 1),
                    "cuts_found": len(scene_intervals),
                    "speed_fps": round(actual_total_frames / elapsed, 1),
                    "resolution": resolution,
                    "fps": actual_fps,
                })

            logger.info(f"Detected {len(scene_intervals)} scenes in video ({actual_total_frames} frames in {time.monotonic() - start_time:.1f}s).")
            return scene_intervals

        except Exception as e:
            logger.error(f"Scene detection error: {e}. Falling back to default uniform chunking.")
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                f_fps = float(cap.get(cv2.CAP_PROP_FPS))
                frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                duration = frame_count / f_fps if f_fps > 0 else 0.0
            except Exception:
                duration = 0.0
            if duration <= 0.0:
                return [(0.0, 0.0)]
            return [
                (start, min(duration, start + 10.0))
                for start in [float(value) for value in range(0, int(duration), 10)]
            ]
