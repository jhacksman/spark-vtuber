"""Speech-to-text engine for Spark VTuber."""

from spark_vtuber.stt.base import BaseSTT, STTResult
from spark_vtuber.stt.parakeet import ParakeetSTT
from spark_vtuber.stt.whisper import WhisperSTT

__all__ = ["BaseSTT", "STTResult", "WhisperSTT", "ParakeetSTT"]
