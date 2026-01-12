"""Text-to-speech engine for Spark VTuber."""

from spark_vtuber.tts.base import BaseTTS, TTSResult
from spark_vtuber.tts.cosyvoice import CosyVoiceTTS
from spark_vtuber.tts.fish_speech import FishSpeechTTS
from spark_vtuber.tts.piper import PiperTTS
from spark_vtuber.tts.streaming import StreamingTTS
from spark_vtuber.tts.styletts2 import StyleTTS2

__all__ = [
    "BaseTTS",
    "TTSResult",
    "CosyVoiceTTS",
    "FishSpeechTTS",
    "PiperTTS",
    "StreamingTTS",
    "StyleTTS2",
]
