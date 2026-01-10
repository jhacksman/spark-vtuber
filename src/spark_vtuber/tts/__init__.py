"""Text-to-speech engine for Spark VTuber."""

from spark_vtuber.tts.base import BaseTTS, TTSResult
from spark_vtuber.tts.streaming import StreamingTTS

__all__ = ["BaseTTS", "TTSResult", "StreamingTTS"]
