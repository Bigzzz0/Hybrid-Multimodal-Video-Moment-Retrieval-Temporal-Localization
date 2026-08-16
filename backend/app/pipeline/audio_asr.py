import os
from typing import List, Dict, Any
from app.core.config import settings
from app.core.logger import logger

class WhisperAudioASR:
    """Timestamped speech-to-text extraction using Faster-Whisper (Whisper-Large-v3-Turbo)."""

    def __init__(self, model_size: str = settings.WHISPER_MODEL_SIZE, device: str = settings.DEVICE):
        self.model_size = model_size
        self.device = device
        self.model = None

    def _lazy_load(self):
        if self.model is None:
            from faster_whisper import WhisperModel
            compute_type = "float16" if self.device == "cuda" else "int8"
            logger.info(f"Loading Faster-Whisper model '{self.model_size}' on {self.device} ({compute_type})...")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)

    def transcribe(self, video_or_audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe speech from video file.
        Returns list of segment dicts: [{"t_start": float, "t_end": float, "spoken_text": str}, ...]
        """
        self._lazy_load()
        logger.info(f"Transcribing audio from: {os.path.basename(video_or_audio_path)}")

        try:
            segments, info = self.model.transcribe(
                video_or_audio_path,
                word_timestamps=True,
                beam_size=5,
                vad_filter=True
            )

            results = []
            for seg in segments:
                text = seg.text.strip()
                if text:
                    results.append({
                        "t_start": float(seg.start),
                        "t_end": float(seg.end),
                        "spoken_text": text
                    })

            logger.info(f"Transcribed {len(results)} speech segments (Language: {info.language}, Prob: {info.language_probability:.2f})")
            return results

        except Exception as e:
            logger.warning(f"Audio transcription warning/error (video may be silent): {e}")
            return []
