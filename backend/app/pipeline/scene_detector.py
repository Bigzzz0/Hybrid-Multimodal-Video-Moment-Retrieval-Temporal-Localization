from typing import List, Tuple
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector, ContentDetector
from app.core.logger import logger

class AdaptiveSceneDetector:
    """Content-aware adaptive scene boundary detector."""

    def __init__(self, adaptive_threshold: float = 3.0, min_scene_len_sec: float = 1.5):
        self.adaptive_threshold = adaptive_threshold
        self.min_scene_len_sec = min_scene_len_sec

    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries.
        Returns a list of tuples: [(t_start_sec, t_end_sec), ...]
        """
        logger.info(f"Starting Scene Detection for: {video_path}")
        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(
                AdaptiveDetector(
                    adaptive_threshold=self.adaptive_threshold,
                    min_scene_len=int(video.frame_rate * self.min_scene_len_sec)
                )
            )
            scene_manager.detect_scenes(video=video)
            scenes = scene_manager.get_scene_list()

            if not scenes:
                # Fallback: Single scene if no cuts detected
                duration = video.duration.get_seconds()
                logger.info(f"No scene cuts found. Using single scene [0.0, {duration:.2f}]")
                return [(0.0, float(duration))]

            scene_intervals = [
                (float(scene[0].get_seconds()), float(scene[1].get_seconds()))
                for scene in scenes
            ]
            logger.info(f"Detected {len(scene_intervals)} scenes in video.")
            return scene_intervals

        except Exception as e:
            logger.error(f"Scene detection error: {e}. Falling back to default uniform chunking.")
            # Default fallback: 10s uniform chunks
            return [(i * 10.0, (i + 1) * 10.0) for i in range(10)]
