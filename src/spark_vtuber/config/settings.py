"""
Configuration settings for Spark VTuber.

Uses Pydantic Settings for environment variable loading and validation.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM configuration settings."""

    model_name: str = Field(
        default="meta-llama/Llama-3.1-70B-Instruct",
        description="HuggingFace model name or local path",
    )
    quantization: Literal["none", "awq", "gptq", "bitsandbytes_4bit"] = Field(
        default="awq",
        description="Quantization method (awq/gptq for vLLM, bitsandbytes_4bit for transformers)",
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    context_length: int = Field(default=8192, description="Maximum context length")
    gpu_memory_utilization: float = Field(
        default=0.85,
        description="Fraction of GPU memory to use",
    )

    model_config = SettingsConfigDict(env_prefix="LLM_")


class TTSSettings(BaseSettings):
    """Text-to-speech configuration settings."""

    engine: Literal["fish_speech", "styletts2"] = Field(
        default="fish_speech",
        description="TTS engine to use (fish_speech recommended for production)",
    )
    model_name: str = Field(
        default="speech-1.5",
        description="Fish Speech model version (speech-1.5 recommended)",
    )
    voice_id: str | None = Field(default=None, description="Voice reference ID for synthesis")
    sample_rate: int = Field(default=44100, description="Audio sample rate (44100 for Fish Speech)")
    streaming: bool = Field(default=True, description="Enable streaming synthesis")
    use_api: bool = Field(
        default=True,
        description="Use Fish Audio cloud API (set False for local inference)",
    )
    api_key: str | None = Field(
        default=None,
        description="Fish Audio API key (or set FISH_API_KEY env var)",
    )

    model_config = SettingsConfigDict(env_prefix="TTS_")


class STTSettings(BaseSettings):
    """Speech-to-text configuration settings."""

    model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = Field(
        default="large-v3",
        description="Whisper model size",
    )
    device: str = Field(default="cuda", description="Device for inference")
    compute_type: Literal["float16", "int8", "int8_float16"] = Field(
        default="float16",
        description="Compute type for inference",
    )
    language: str = Field(default="en", description="Language for transcription")
    vad_enabled: bool = Field(default=True, description="Enable voice activity detection")

    model_config = SettingsConfigDict(env_prefix="STT_")


class MemorySettings(BaseSettings):
    """Memory and RAG configuration settings."""

    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"),
        description="ChromaDB persistence directory",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    max_memories: int = Field(default=10000, description="Maximum memories to store")
    retrieval_top_k: int = Field(default=5, description="Number of memories to retrieve")
    summarization_threshold: int = Field(
        default=10000,
        description="Token count to trigger summarization",
    )

    model_config = SettingsConfigDict(env_prefix="MEMORY_")


class AvatarSettings(BaseSettings):
    """Avatar control configuration settings."""

    vtube_studio_host: str = Field(default="localhost", description="VTube Studio host")
    vtube_studio_port: int = Field(default=8001, description="VTube Studio WebSocket port")
    plugin_name: str = Field(default="SparkVTuber", description="Plugin name for VTube Studio")
    plugin_developer: str = Field(default="SparkVTuber", description="Plugin developer name")
    lip_sync_enabled: bool = Field(default=True, description="Enable lip sync")
    expression_enabled: bool = Field(default=True, description="Enable expression control")

    dual_avatar_enabled: bool = Field(
        default=False,
        description="Enable dual avatar mode (requires VNet Multiplayer Collab)",
    )
    primary_avatar_port: int = Field(
        default=8001,
        description="VTube Studio port for primary avatar (Spark)",
    )
    secondary_avatar_port: int = Field(
        default=8002,
        description="VTube Studio port for secondary avatar (Shadow)",
    )
    primary_avatar_position: Literal["left", "center", "right"] = Field(
        default="left",
        description="Preferred position for primary avatar (VNet handles actual positioning)",
    )
    secondary_avatar_position: Literal["left", "center", "right"] = Field(
        default="right",
        description="Preferred position for secondary avatar (VNet handles actual positioning)",
    )

    model_config = SettingsConfigDict(env_prefix="AVATAR_")


class PersonalitySettings(BaseSettings):
    """Dual AI personality configuration settings."""

    primary_name: str = Field(default="Spark", description="Primary personality name")
    primary_lora_path: str | None = Field(
        default=None,
        description="Path to primary personality LoRA adapter",
    )
    secondary_name: str = Field(default="Shadow", description="Secondary personality name")
    secondary_lora_path: str | None = Field(
        default=None,
        description="Path to secondary personality LoRA adapter",
    )
    switch_cooldown: float = Field(
        default=5.0,
        description="Minimum seconds between personality switches",
    )

    model_config = SettingsConfigDict(env_prefix="PERSONALITY_")


class ChatSettings(BaseSettings):
    """Chat integration configuration settings."""

    twitch_enabled: bool = Field(default=False, description="Enable Twitch integration")
    twitch_channel: str = Field(default="", description="Twitch channel name")
    twitch_oauth_token: SecretStr = Field(
        default=SecretStr(""),
        description="Twitch OAuth token (stored securely)",
    )
    youtube_enabled: bool = Field(default=False, description="Enable YouTube integration")
    youtube_video_id: str = Field(default="", description="YouTube live video ID")
    youtube_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="YouTube API key (stored securely)",
    )
    message_queue_size: int = Field(default=100, description="Maximum queued messages")
    rate_limit_per_minute: int = Field(default=20, description="Max responses per minute")

    def get_twitch_token(self) -> str:
        """Get Twitch OAuth token value (never log this)."""
        return self.twitch_oauth_token.get_secret_value()

    def get_youtube_key(self) -> str:
        """Get YouTube API key value (never log this)."""
        return self.youtube_api_key.get_secret_value()

    model_config = SettingsConfigDict(env_prefix="CHAT_")


class GameSettings(BaseSettings):
    """Game integration configuration settings."""

    minecraft_enabled: bool = Field(default=False, description="Enable Minecraft integration")
    minecraft_host: str = Field(default="localhost", description="Minecraft server host")
    minecraft_port: int = Field(default=25565, description="Minecraft server port")
    minecraft_username: str = Field(default="SparkVTuber", description="Minecraft username")
    watch_mode_enabled: bool = Field(default=False, description="Enable watch/spectator mode")
    screen_capture_fps: int = Field(default=2, description="Screen capture FPS for watch mode")

    model_config = SettingsConfigDict(env_prefix="GAME_")


class Settings(BaseSettings):
    """Main application settings."""

    app_name: str = Field(default="SparkVTuber", description="Application name")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    data_dir: Path = Field(default=Path("./data"), description="Data directory")

    # Component settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    avatar: AvatarSettings = Field(default_factory=AvatarSettings)
    personality: PersonalitySettings = Field(default_factory=PersonalitySettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    game: GameSettings = Field(default_factory=GameSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings
