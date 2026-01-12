"""Text-to-speech engine for Spark VTuber."""

from spark_vtuber.tts.base import BaseTTS, TTSResult
from spark_vtuber.tts.fish_speech import FishSpeechTTS
from spark_vtuber.tts.llmvox import LLMVoXTTS
from spark_vtuber.tts.streaming import StreamingTTS
from spark_vtuber.tts.styletts2 import StyleTTS2

__all__ = ["BaseTTS", "TTSResult", "FishSpeechTTS", "LLMVoXTTS", "StreamingTTS", "StyleTTS2"]
