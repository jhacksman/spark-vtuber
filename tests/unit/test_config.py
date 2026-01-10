"""Tests for configuration module."""

import pytest
from pathlib import Path

from spark_vtuber.config.settings import (
    Settings,
    LLMSettings,
    TTSSettings,
    STTSettings,
    MemorySettings,
    AvatarSettings,
    PersonalitySettings,
    ChatSettings,
    GameSettings,
    get_settings,
)


class TestLLMSettings:
    """Tests for LLM settings."""

    def test_default_values(self):
        """Test default LLM settings."""
        settings = LLMSettings()
        assert settings.model_name == "Qwen/Qwen3-30B-A3B-Instruct"
        assert settings.quantization == "awq"
        assert settings.max_tokens == 2048
        assert settings.temperature == 0.7
        assert settings.top_p == 0.9
        assert settings.context_length == 8192
        assert settings.gpu_memory_utilization == 0.85

    def test_custom_values(self):
        """Test custom LLM settings."""
        settings = LLMSettings(
            model_name="custom-model",
            quantization="8bit",
            max_tokens=4096,
        )
        assert settings.model_name == "custom-model"
        assert settings.quantization == "8bit"
        assert settings.max_tokens == 4096


class TestTTSSettings:
    """Tests for TTS settings."""

    def test_default_values(self):
        """Test default TTS settings."""
        settings = TTSSettings()
        assert settings.engine == "coqui"
        assert settings.sample_rate == 22050
        assert settings.streaming is True

    def test_custom_values(self):
        """Test custom TTS settings."""
        settings = TTSSettings(
            engine="styletts2",
            sample_rate=44100,
        )
        assert settings.engine == "styletts2"
        assert settings.sample_rate == 44100


class TestSTTSettings:
    """Tests for STT settings."""

    def test_default_values(self):
        """Test default STT settings."""
        settings = STTSettings()
        assert settings.model_size == "large-v3"
        assert settings.device == "cuda"
        assert settings.compute_type == "float16"
        assert settings.language == "en"
        assert settings.vad_enabled is True


class TestMemorySettings:
    """Tests for memory settings."""

    def test_default_values(self):
        """Test default memory settings."""
        settings = MemorySettings()
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.max_memories == 10000
        assert settings.retrieval_top_k == 5


class TestAvatarSettings:
    """Tests for avatar settings."""

    def test_default_values(self):
        """Test default avatar settings."""
        settings = AvatarSettings()
        assert settings.vtube_studio_host == "localhost"
        assert settings.vtube_studio_port == 8001
        assert settings.plugin_name == "SparkVTuber"
        assert settings.lip_sync_enabled is True


class TestPersonalitySettings:
    """Tests for personality settings."""

    def test_default_values(self):
        """Test default personality settings."""
        settings = PersonalitySettings()
        assert settings.primary_name == "Spark"
        assert settings.secondary_name == "Shadow"
        assert settings.switch_cooldown == 5.0


class TestChatSettings:
    """Tests for chat settings."""

    def test_default_values(self):
        """Test default chat settings."""
        settings = ChatSettings()
        assert settings.twitch_enabled is False
        assert settings.youtube_enabled is False
        assert settings.message_queue_size == 100
        assert settings.rate_limit_per_minute == 20


class TestGameSettings:
    """Tests for game settings."""

    def test_default_values(self):
        """Test default game settings."""
        settings = GameSettings()
        assert settings.minecraft_enabled is False
        assert settings.watch_mode_enabled is False
        assert settings.screen_capture_fps == 2


class TestSettings:
    """Tests for main settings."""

    def test_default_values(self):
        """Test default main settings."""
        settings = Settings()
        assert settings.app_name == "SparkVTuber"
        assert settings.debug is False
        assert settings.log_level == "INFO"

    def test_nested_settings(self):
        """Test nested settings access."""
        settings = Settings()
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.tts, TTSSettings)
        assert isinstance(settings.stt, STTSettings)
        assert isinstance(settings.memory, MemorySettings)
        assert isinstance(settings.avatar, AvatarSettings)
        assert isinstance(settings.personality, PersonalitySettings)
        assert isinstance(settings.chat, ChatSettings)
        assert isinstance(settings.game, GameSettings)

    def test_ensure_directories(self, tmp_path):
        """Test directory creation."""
        settings = Settings(
            data_dir=tmp_path / "data",
            memory=MemorySettings(chroma_persist_dir=tmp_path / "chroma"),
        )
        settings.ensure_directories()
        assert (tmp_path / "data").exists()
        assert (tmp_path / "chroma").exists()


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings(self):
        """Test that get_settings returns Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_singleton_behavior(self):
        """Test that get_settings returns same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
