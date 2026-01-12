"""
CosyVoice TTS implementation for Spark VTuber.

Provides high-quality TTS with TRUE streaming support and voice cloning.
CosyVoice 3.0 (Fun-CosyVoice3-0.5B) offers state-of-the-art quality with 150ms latency.

Supports both local inference with streaming and cloud API modes.
"""

import asyncio
import os
import tempfile
import time
from typing import AsyncIterator

import numpy as np

from spark_vtuber.tts.base import BaseTTS, TTSResult


class CosyVoiceTTS(BaseTTS):
    """
    CosyVoice TTS implementation.

    Supports:
    - High-quality speech synthesis (0.81% CER, 78% speaker similarity)
    - TRUE streaming output with 150ms first-chunk latency
    - Zero-shot voice cloning from 3-15 seconds of audio
    - 100+ emotion/tone controls (happy, sad, angry, excited, calm, whisper, etc.)
    - Cross-lingual support (9 languages + 18 Chinese dialects)
    - Both local inference and cloud API modes

    Memory usage: ~8GB VRAM for 0.5B model
    Latency: ~150ms first chunk (streaming mode)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        use_api: bool = False,
        api_key: str | None = None,
        reference_audio_path: str | None = None,
        model_name: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        model_path: str | None = None,
        device: str = "cuda",
        half_precision: bool = True,
        compile_model: bool = False,
        **kwargs,
    ):
        """
        Initialize CosyVoice TTS.

        Args:
            sample_rate: Output sample rate (CosyVoice supports 22050)
            use_api: Whether to use cloud API (False for local inference - DEFAULT)
            api_key: API key if using cloud mode
            reference_audio_path: Default voice reference audio file path
            model_name: HuggingFace model ID or local path
            model_path: Path to local model weights (auto-downloads if not specified)
            device: Device to run inference on (cuda, cpu)
            half_precision: Use FP16 for faster inference (default True)
            compile_model: Use torch.compile for faster inference
            **kwargs: Additional arguments
        """
        super().__init__(sample_rate, **kwargs)
        self.use_api = use_api
        self.api_key = api_key or os.environ.get("COSYVOICE_API_KEY")
        self.reference_audio_path = reference_audio_path
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.half_precision = half_precision
        self.compile_model = compile_model
        self._client = None
        self._local_model = None
        self._cloned_voices: dict[str, str] = {}
        self._reference_audio_cache: dict[str, np.ndarray] = {}
        self._first_chunk_latency_ms: float = 0.0

    async def load(self) -> None:
        """Load the TTS model/client."""
        if self._loaded:
            self.logger.warning("TTS already loaded")
            return

        self.logger.info(f"Loading CosyVoice TTS (api_mode={self.use_api})")
        start_time = time.time()

        try:
            if self.use_api:
                # API mode (if CosyVoice offers cloud API in the future)
                if not self.api_key:
                    raise ValueError(
                        "COSYVOICE_API_KEY environment variable or api_key parameter required "
                        "for API mode. Set use_api=False for local inference."
                    )
                # API client initialization would go here
                raise NotImplementedError("CosyVoice cloud API not yet supported")
            else:
                # Load local CosyVoice model
                await self._load_local_model()

            self._loaded = True
            load_time = time.time() - start_time
            self.logger.info(f"CosyVoice TTS loaded in {load_time:.2f}s")

        except ImportError as e:
            self.logger.error(f"Failed to import CosyVoice: {e}")
            self.logger.error(
                "Install with: "
                "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git && "
                "cd CosyVoice && pip install -e ."
            )
            raise
        except Exception as e:
            self.logger.error(f"Failed to load CosyVoice TTS: {e}")
            raise

    async def _load_local_model(self) -> None:
        """Load local CosyVoice model for inference."""
        import torch

        loop = asyncio.get_event_loop()

        def _load_sync():
            try:
                # Try CosyVoice2/3 API (recommended)
                from cosyvoice.cli.cosyvoice import CosyVoice2

                # Determine model path
                model_path = self.model_path
                if model_path is None:
                    # Use HuggingFace hub to download model
                    from huggingface_hub import snapshot_download

                    self.logger.info(
                        f"Downloading CosyVoice model from HuggingFace: {self.model_name}..."
                    )
                    model_path = snapshot_download(
                        repo_id=self.model_name,
                        allow_patterns=["*.pt", "*.pth", "*.json", "*.yaml", "config.*", "*.safetensors"],
                    )

                self.logger.info(f"Loading CosyVoice model from {model_path}")

                # Load the model with streaming support
                model = CosyVoice2(
                    model_path,
                    load_jit=self.compile_model,  # Use JIT compilation if enabled
                    load_onnx=False,  # We'll use PyTorch
                    load_trt=False,  # TensorRT for production deployment
                )

                return {
                    "model": model,
                    "api": "cosyvoice2",
                    "sample_rate": model.sample_rate,
                }

            except ImportError:
                # Fallback: try original CosyVoice API
                try:
                    from cosyvoice.cli.cosyvoice import CosyVoice

                    model_path = self.model_path
                    if model_path is None:
                        from huggingface_hub import snapshot_download

                        self.logger.info(
                            f"Downloading CosyVoice model: {self.model_name}..."
                        )
                        model_path = snapshot_download(
                            repo_id=self.model_name,
                            allow_patterns=["*.pt", "*.pth", "*.json", "*.yaml", "config.*"],
                        )

                    self.logger.info(f"Loading CosyVoice model from {model_path}")

                    model = CosyVoice(model_path)

                    return {
                        "model": model,
                        "api": "cosyvoice",
                        "sample_rate": getattr(model, "sample_rate", 22050),
                    }

                except ImportError:
                    raise ImportError(
                        "CosyVoice not installed. Install with: "
                        "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git && "
                        "cd CosyVoice && pip install -e ."
                    )

        self._local_model = await loop.run_in_executor(None, _load_sync)
        self.logger.info(
            f"CosyVoice local model loaded successfully (api={self._local_model.get('api')}, "
            f"sample_rate={self._local_model.get('sample_rate')})"
        )

        # Update sample rate from model
        model_sr = self._local_model.get("sample_rate")
        if model_sr and model_sr != self.sample_rate:
            self.logger.warning(
                f"Adjusting sample rate from {self.sample_rate} to {model_sr} (model default)"
            )
            self.sample_rate = model_sr

    async def unload(self) -> None:
        """Unload the TTS model/client."""
        if not self._loaded:
            return

        self.logger.info("Unloading CosyVoice TTS")

        self._client = None
        self._local_model = None
        self._cloned_voices.clear()
        self._reference_audio_cache.clear()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        emotion: str | None = None,
    ) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice_id: Voice reference ID (path to audio file)
            speed: Speech speed multiplier (0.5-2.0)
            emotion: Emotion/instruction text (e.g., "Speak with excitement", "happy", "whisper")

        Returns:
            TTSResult with audio data
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        start_time = time.time()

        # Build instruct text from emotion if provided
        instruct_text = None
        if emotion:
            # Map simple emotion words to instruction phrases
            emotion_map = {
                "happy": "Speak with happiness and joy",
                "sad": "Speak with sadness",
                "angry": "Speak with anger",
                "excited": "Speak with excitement and energy",
                "calm": "Speak calmly and peacefully",
                "whisper": "Speak in a whisper",
                "surprised": "Speak with surprise",
                "fearful": "Speak with fear",
            }
            instruct_text = emotion_map.get(emotion.lower(), emotion)

        # Get reference audio path
        ref_audio_path = voice_id or self.reference_audio_path
        if voice_id and voice_id in self._cloned_voices:
            ref_audio_path = self._cloned_voices[voice_id]

        audio = await self._synthesize_local(text, ref_audio_path, speed, instruct_text)

        latency_ms = (time.time() - start_time) * 1000
        duration = len(audio) / self.sample_rate

        return TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            latency_ms=latency_ms,
            metadata={
                "engine": "cosyvoice",
                "model": self.model_name,
                "voice_id": voice_id,
                "emotion": emotion,
                "instruct_text": instruct_text,
            },
        )

    async def _synthesize_local(
        self,
        text: str,
        reference_audio_path: str | None,
        speed: float,
        instruct_text: str | None,
    ) -> np.ndarray:
        """
        Synthesize using local CosyVoice inference (non-streaming).

        This collects all audio chunks and returns the complete audio.
        For streaming, use synthesize_stream() instead.
        """
        if self._local_model is None:
            raise RuntimeError("Local model not loaded")

        # Collect all chunks
        audio_chunks = []
        async for chunk in self._stream_local(text, reference_audio_path, speed, instruct_text):
            audio_chunks.append(chunk)

        if not audio_chunks:
            self.logger.warning("No audio generated")
            return np.array([], dtype=np.float32)

        # Concatenate all chunks
        audio = np.concatenate(audio_chunks, axis=0)
        return audio

    async def _stream_local(
        self,
        text: str,
        reference_audio_path: str | None,
        speed: float,
        instruct_text: str | None,
    ) -> AsyncIterator[np.ndarray]:
        """
        Generate streaming audio using local CosyVoice model.

        This is where TRUE streaming happens - yields audio chunks as they're generated.
        """
        if self._local_model is None:
            raise RuntimeError("Local model not loaded")

        loop = asyncio.get_event_loop()
        model = self._local_model["model"]
        api_type = self._local_model["api"]

        def _stream_sync():
            """
            Synchronous streaming generator.
            CosyVoice supports stream=True for chunk-by-chunk generation.
            """
            import torchaudio

            try:
                # Load reference audio if provided (for zero-shot voice cloning)
                prompt_speech_16k = None
                prompt_text = ""

                if reference_audio_path and os.path.exists(reference_audio_path):
                    # Load and resample to 16kHz (CosyVoice requirement)
                    from cosyvoice.utils.file_utils import load_wav

                    prompt_speech_16k = load_wav(reference_audio_path, 16000)
                    # Use first few seconds of reference as prompt text
                    # (In production, you'd transcribe this or provide it)
                    prompt_text = ""  # Empty for zero-shot

                # Choose inference mode based on inputs
                if instruct_text and reference_audio_path:
                    # Instruct mode with voice cloning
                    stream_iter = model.inference_instruct(
                        text,
                        instruct_text,
                        prompt_speech_16k,
                        stream=True,  # CRITICAL: Enable streaming
                    )
                elif instruct_text:
                    # Instruct mode (emotion control) without voice cloning
                    stream_iter = model.inference_instruct(
                        text,
                        instruct_text,
                        None,
                        stream=True,
                    )
                elif reference_audio_path and prompt_speech_16k is not None:
                    # Zero-shot voice cloning mode
                    stream_iter = model.inference_zero_shot(
                        text,
                        prompt_text,
                        prompt_speech_16k,
                        stream=True,  # CRITICAL: Enable streaming
                    )
                else:
                    # Default inference (no cloning, no emotion)
                    stream_iter = model.inference_sft(
                        text,
                        stream=True,
                    )

                # Stream chunks as they're generated
                for i, chunk_dict in enumerate(stream_iter):
                    # chunk_dict contains 'tts_speech' tensor
                    audio_tensor = chunk_dict["tts_speech"]

                    # Convert to numpy
                    audio_np = audio_tensor.cpu().numpy().squeeze()

                    # Apply speed adjustment if needed (affects pitch too)
                    if speed != 1.0:
                        import scipy.signal
                        new_length = int(len(audio_np) / speed)
                        audio_np = scipy.signal.resample(audio_np, new_length)

                    yield audio_np.astype(np.float32)

            except Exception as e:
                self.logger.error(f"CosyVoice streaming failed: {e}")
                raise RuntimeError(f"CosyVoice inference failed: {e}")

        # Convert sync generator to async
        sync_gen = _stream_sync()
        while True:
            try:
                chunk = await loop.run_in_executor(None, lambda: next(sync_gen))
                yield chunk
            except StopIteration:
                break

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        emotion: str | None = None,
        chunk_size: int = 4096,
    ) -> AsyncIterator[np.ndarray]:
        """
        Synthesize speech with TRUE streaming output.

        CosyVoice supports real streaming with ~150ms first-chunk latency.

        Args:
            text: Text to synthesize
            voice_id: Voice reference ID (audio file path)
            speed: Speech speed multiplier
            emotion: Emotion tag or instruction text
            chunk_size: Audio chunk size in samples (not used - CosyVoice controls chunks)

        Yields:
            Audio chunks as numpy arrays as they're generated
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        start_time = time.time()
        first_chunk = True

        # Build instruct text from emotion
        instruct_text = None
        if emotion:
            emotion_map = {
                "happy": "Speak with happiness and joy",
                "sad": "Speak with sadness",
                "angry": "Speak with anger",
                "excited": "Speak with excitement and energy",
                "calm": "Speak calmly and peacefully",
                "whisper": "Speak in a whisper",
                "surprised": "Speak with surprise",
                "fearful": "Speak with fear",
            }
            instruct_text = emotion_map.get(emotion.lower(), emotion)

        # Get reference audio path
        ref_audio_path = voice_id or self.reference_audio_path
        if voice_id and voice_id in self._cloned_voices:
            ref_audio_path = self._cloned_voices[voice_id]

        # Stream audio chunks
        async for chunk in self._stream_local(text, ref_audio_path, speed, instruct_text):
            if first_chunk:
                self._first_chunk_latency_ms = (time.time() - start_time) * 1000
                self.logger.info(f"First chunk latency: {self._first_chunk_latency_ms:.1f}ms")
                first_chunk = False
            yield chunk

    async def clone_voice(
        self,
        reference_audio: np.ndarray,
        voice_id: str,
    ) -> None:
        """
        Clone a voice from reference audio.

        For CosyVoice, this saves the reference audio file for later zero-shot use.
        CosyVoice requires 3-15 seconds of clean audio for good results.

        Args:
            reference_audio: Reference audio samples (numpy array)
            voice_id: Identifier for the cloned voice
        """
        import soundfile as sf

        # Save reference audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, reference_audio, self.sample_rate)
            temp_path = f.name

        self._cloned_voices[voice_id] = temp_path
        self.logger.info(f"Voice reference saved: {voice_id} -> {temp_path}")
        self.logger.info(
            f"For best results, use 3-15 seconds of clean audio with emotional variation"
        )

    def get_available_voices(self) -> list[str]:
        """Get list of available voice IDs."""
        voices = list(self._cloned_voices.keys())

        if self.reference_audio_path:
            voices.append("default")

        return voices

    def get_first_chunk_latency(self) -> float:
        """Get the latency of the first audio chunk in milliseconds."""
        return self._first_chunk_latency_ms
