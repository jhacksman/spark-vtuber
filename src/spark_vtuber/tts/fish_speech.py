"""
Fish Speech TTS implementation for Spark VTuber.

Provides high-quality TTS with streaming support and voice cloning.
Fish Speech 1.5 (OpenAudio S1) is a state-of-the-art TTS model ranked #1 on TTS-Arena2.

Supports both local inference and cloud API modes.
"""

import asyncio
import os
import tempfile
import time
from collections.abc import AsyncIterator

import numpy as np

from spark_vtuber.tts.base import BaseTTS, TTSResult


class FishSpeechTTS(BaseTTS):
    """
    Fish Speech TTS implementation.

    Supports:
    - High-quality speech synthesis
    - Pipeline streaming with intelligent break finding
    - Voice cloning from reference audio
    - Emotion control via text markers
    - Both local inference and cloud API modes
    - ARM64 + CUDA compatible (works on DGX Spark)

    Streaming strategy:
    - API mode: True token-level streaming (~300ms first chunk)
    - Local mode: Pipeline buffering with break finder agent
      1. Break finder identifies natural first chunk boundary
      2. First chunk generates immediately (~200ms)
      3. Remaining content generates in background during playback

    Memory usage: ~12GB VRAM for local inference
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        use_api: bool = False,
        api_key: str | None = None,
        reference_id: str | None = None,
        model: str = "speech-1.5",
        model_path: str | None = None,
        device: str = "cuda",
        half_precision: bool = True,
        compile_model: bool = False,
        break_finder_enabled: bool = True,
        break_finder_api_base: str = "http://localhost:8001/v1",
        break_finder_model: str = "Qwen/Qwen3-0.5B-Instruct",
        break_finder_timeout_ms: int = 100,
        **kwargs,
    ):
        """
        Initialize Fish Speech TTS.

        Args:
            sample_rate: Output sample rate (default 44100 for Fish Speech)
            use_api: Whether to use cloud API (False for local inference - DEFAULT)
            api_key: Fish Audio API key (required if use_api=True)
            reference_id: Default voice reference ID for synthesis
            model: Model version to use (speech-1.5 recommended)
            model_path: Path to local model weights (auto-downloads if not specified)
            device: Device to run inference on (cuda, cpu)
            half_precision: Use FP16 for faster inference (default True)
            compile_model: Use torch.compile for faster inference (Linux only)
            break_finder_enabled: Enable LLM-based break finding for streaming
            break_finder_api_base: API base URL for break finder model
            break_finder_model: Model to use for break finding
            break_finder_timeout_ms: Max time to wait for break finder
            **kwargs: Additional arguments
        """
        super().__init__(sample_rate, **kwargs)
        self.use_api = use_api
        self.api_key = api_key or os.environ.get("FISH_API_KEY")
        self.reference_id = reference_id
        self.model = model
        self.model_path = model_path
        self.device = device
        self.half_precision = half_precision
        self.compile_model = compile_model
        self._client = None
        self._local_model = None
        self._tokenizer = None
        self._cloned_voices: dict[str, str] = {}
        self._reference_audio_cache: dict[str, np.ndarray] = {}
        self._first_chunk_latency_ms: float = 0.0

        # Initialize break finder for intelligent streaming
        from spark_vtuber.tts.break_finder import BreakFinder

        self._break_finder = BreakFinder(
            api_base=break_finder_api_base,
            model_name=break_finder_model,
            timeout_ms=break_finder_timeout_ms,
            enabled=break_finder_enabled,
        )

    async def load(self) -> None:
        """Load the TTS model/client."""
        if self._loaded:
            self.logger.warning("TTS already loaded")
            return

        self.logger.info(f"Loading Fish Speech TTS (api_mode={self.use_api})")
        start_time = time.time()

        try:
            if self.use_api:
                from fishaudio import FishAudio

                if not self.api_key:
                    raise ValueError(
                        "FISH_API_KEY environment variable or api_key parameter required "
                        "for API mode. Set use_api=False for local inference."
                    )
                self._client = FishAudio(api_key=self.api_key)
            else:
                # Load local Fish Speech model
                await self._load_local_model()

            self._loaded = True
            load_time = time.time() - start_time
            self.logger.info(f"Fish Speech TTS loaded in {load_time:.2f}s")

        except ImportError as e:
            self.logger.error(f"Failed to import Fish Audio SDK: {e}")
            self.logger.error("Install with: pip install fish-audio-sdk")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load Fish Speech TTS: {e}")
            raise

    async def _load_local_model(self) -> None:
        """Load local Fish Speech model for inference."""
        import torch

        loop = asyncio.get_event_loop()

        def _load_sync():
            # Try to import Fish Speech inference module
            try:
                # Try the main inference API first (fish-speech >= 1.5)
                from fish_speech.inference import load_model

                self.logger.info("Loading Fish Speech model via inference API...")
                model = load_model(
                    checkpoint_path=self.model_path,
                    device=self.device,
                    half=self.half_precision,
                    compile=self.compile_model,
                )
                return {"model": model, "api": "inference"}

            except ImportError:
                pass

            try:
                # Try tools.inference module (alternative path)
                from tools.inference import load_model

                self.logger.info("Loading Fish Speech model via tools.inference...")
                model = load_model(
                    checkpoint_path=self.model_path,
                    device=self.device,
                    half=self.half_precision,
                    compile=self.compile_model,
                )
                return {"model": model, "api": "tools"}

            except ImportError:
                pass

            try:
                # Try loading model components directly
                from fish_speech.models.text2semantic.inference import (
                    load_model as load_llama_model,
                )
                from fish_speech.models.vqgan.inference import (
                    load_model as load_vqgan_model,
                )

                # Determine model path
                model_path = self.model_path
                if model_path is None:
                    # Use HuggingFace hub to download model
                    from huggingface_hub import snapshot_download

                    self.logger.info("Downloading Fish Speech model from HuggingFace...")
                    model_path = snapshot_download(
                        repo_id="fishaudio/openaudio-s1-mini",
                        allow_patterns=["*.pth", "*.json", "*.yaml", "config.*"],
                    )

                self.logger.info(f"Loading Fish Speech model from {model_path}")

                # Load the models
                device = torch.device(self.device)
                dtype = torch.float16 if self.half_precision else torch.float32

                # Load semantic model (LLaMA-based)
                llama_model = load_llama_model(
                    checkpoint_path=f"{model_path}/llama",
                    device=device,
                    dtype=dtype,
                    compile=self.compile_model,
                )

                # Load VQGAN vocoder
                vqgan_model = load_vqgan_model(
                    checkpoint_path=f"{model_path}/vqgan",
                    device=device,
                )

                return {
                    "llama": llama_model,
                    "vqgan": vqgan_model,
                    "device": device,
                    "dtype": dtype,
                    "api": "components",
                }

            except ImportError:
                raise ImportError(
                    "Fish Speech not installed. Install with: "
                    "git clone https://github.com/fishaudio/fish-speech && "
                    "cd fish-speech && pip install -e '.[cu129]'"
                )

        self._local_model = await loop.run_in_executor(None, _load_sync)
        self.logger.info(f"Fish Speech local model loaded successfully (api={self._local_model.get('api')})")

    async def unload(self) -> None:
        """Unload the TTS model/client."""
        if not self._loaded:
            return

        self.logger.info("Unloading Fish Speech TTS")

        self._client = None
        self._local_model = None
        self._tokenizer = None
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
            text: Text to synthesize (supports emotion markers like (happy), (sad))
            voice_id: Voice reference ID (uses default if not specified)
            speed: Speech speed multiplier (0.5-2.0)
            emotion: Emotion tag to prepend (happy, sad, angry, calm, etc.)

        Returns:
            TTSResult with audio data
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        start_time = time.time()

        if emotion:
            text = f"({emotion}) {text}"

        ref_id = voice_id or self._cloned_voices.get(voice_id) or self.reference_id

        if self.use_api and self._client:
            audio = await self._synthesize_api(text, ref_id, speed)
        else:
            audio = await self._synthesize_local(text, ref_id, speed)

        latency_ms = (time.time() - start_time) * 1000
        duration = len(audio) / self.sample_rate

        return TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            latency_ms=latency_ms,
            metadata={
                "engine": "fish_speech",
                "model": self.model,
                "voice_id": ref_id,
                "emotion": emotion,
            },
        )

    async def _synthesize_api(
        self,
        text: str,
        reference_id: str | None,
        speed: float,
    ) -> np.ndarray:
        """Synthesize using Fish Audio cloud API."""
        loop = asyncio.get_event_loop()

        def _sync_convert():
            kwargs = {"text": text, "speed": speed}
            if reference_id:
                kwargs["reference_id"] = reference_id
            return self._client.tts.convert(**kwargs)

        audio_bytes = await loop.run_in_executor(None, _sync_convert)

        import io

        import soundfile as sf

        audio_data, sr = sf.read(io.BytesIO(audio_bytes))

        if sr != self.sample_rate:
            import scipy.signal

            num_samples = int(len(audio_data) * self.sample_rate / sr)
            audio_data = scipy.signal.resample(audio_data, num_samples)

        return np.array(audio_data, dtype=np.float32)

    async def _synthesize_local(
        self,
        text: str,
        reference_id: str | None,
        speed: float,
    ) -> np.ndarray:
        """
        Synthesize using local Fish Speech inference.

        Uses the locally loaded Fish Speech model for TTS generation.
        Supports voice cloning via reference audio.
        """
        if self._local_model is None:
            raise RuntimeError(
                "Local model not loaded. Ensure fish-speech is installed: "
                "git clone https://github.com/fishaudio/fish-speech && "
                "cd fish-speech && pip install -e '.[cu129]'"
            )

        loop = asyncio.get_event_loop()

        def _synthesize_sync():
            import torch

            # Get reference audio if specified
            reference_audio = None
            if reference_id and reference_id in self._reference_audio_cache:
                reference_audio = self._reference_audio_cache[reference_id]
            elif reference_id and reference_id in self._cloned_voices:
                # Load reference audio from file path
                voice_path = self._cloned_voices[reference_id]
                if os.path.exists(voice_path):
                    import soundfile as sf
                    reference_audio, _ = sf.read(voice_path)
                    self._reference_audio_cache[reference_id] = reference_audio

            api_type = self._local_model.get("api")

            try:
                if api_type == "inference":
                    # Use fish_speech.inference API
                    from fish_speech.inference import synthesize

                    audio = synthesize(
                        text=text,
                        model=self._local_model["model"],
                        reference_audio=reference_audio,
                        speed=speed,
                    )
                    return np.array(audio, dtype=np.float32)

                elif api_type == "tools":
                    # Use tools.inference module
                    from tools.inference import synthesize

                    audio = synthesize(
                        text=text,
                        model=self._local_model["model"],
                        reference_audio=reference_audio,
                        speed=speed,
                    )
                    return np.array(audio, dtype=np.float32)

                elif api_type == "components":
                    # Use model components directly
                    llama = self._local_model.get("llama")
                    vqgan = self._local_model.get("vqgan")

                    if llama is None or vqgan is None:
                        raise RuntimeError("Model components not loaded properly")

                    # Generate semantic tokens from text
                    with torch.no_grad():
                        # Tokenize and generate semantic tokens
                        # The exact API depends on fish-speech version
                        try:
                            from fish_speech.text import clean_text, g2p
                            cleaned_text = clean_text(text)
                            phonemes = g2p(cleaned_text)
                        except ImportError:
                            # Fallback: use text directly
                            phonemes = text

                        # Generate semantic tokens using LLaMA model
                        semantic_tokens = llama.generate(
                            phonemes,
                            reference_audio=reference_audio,
                            max_new_tokens=2048,
                        )

                        # Decode to audio using VQGAN
                        audio = vqgan.decode(semantic_tokens)

                    # Convert to numpy
                    audio_np = audio.cpu().numpy().squeeze()

                    # Apply speed adjustment if needed
                    if speed != 1.0:
                        import scipy.signal
                        new_length = int(len(audio_np) / speed)
                        audio_np = scipy.signal.resample(audio_np, new_length)

                    return audio_np.astype(np.float32)

                else:
                    raise RuntimeError(f"Unknown API type: {api_type}")

            except Exception as e:
                self.logger.error(f"Local synthesis failed: {e}")
                raise RuntimeError(
                    f"Fish Speech local inference failed: {e}. "
                    "Ensure fish-speech is properly installed."
                )

        audio = await loop.run_in_executor(None, _synthesize_sync)

        # Resample to target sample rate if needed (Fish Speech outputs at 44100Hz)
        fish_speech_sr = 44100
        if self.sample_rate != fish_speech_sr:
            import scipy.signal
            num_samples = int(len(audio) * self.sample_rate / fish_speech_sr)
            audio = scipy.signal.resample(audio, num_samples)

        return audio

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        emotion: str | None = None,
        chunk_size: int = 4096,
    ) -> AsyncIterator[np.ndarray]:
        """
        Synthesize speech with streaming output using intelligent pipeline buffering.

        For local inference, uses break finder agent to identify natural split points:
        1. Break finder identifies optimal first chunk boundary (interjections, short phrases)
        2. First chunk generates immediately for low latency
        3. Remaining content generates in background during playback

        For API mode, uses native streaming API.

        Args:
            text: Text to synthesize
            voice_id: Voice reference ID
            speed: Speech speed multiplier
            emotion: Emotion tag
            chunk_size: Audio chunk size in samples

        Yields:
            Audio chunks as numpy arrays
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        start_time = time.time()
        first_chunk = True

        if emotion:
            text = f"({emotion}) {text}"

        ref_id = voice_id or self._cloned_voices.get(voice_id) or self.reference_id

        if self.use_api and self._client:
            # API mode: use native streaming
            async for chunk in self._stream_api(text, ref_id, speed, chunk_size):
                if first_chunk:
                    self._first_chunk_latency_ms = (time.time() - start_time) * 1000
                    first_chunk = False
                yield chunk
        else:
            # Local mode: use pipeline buffering with break finder
            async for chunk in self._stream_local_pipelined(
                text, voice_id, speed, emotion, chunk_size, start_time
            ):
                if first_chunk:
                    self._first_chunk_latency_ms = (time.time() - start_time) * 1000
                    first_chunk = False
                yield chunk

    async def _stream_local_pipelined(
        self,
        text: str,
        voice_id: str | None,
        speed: float,
        emotion: str | None,
        chunk_size: int,
        start_time: float,
    ) -> AsyncIterator[np.ndarray]:
        """
        Stream local synthesis using intelligent pipeline buffering.

        Strategy:
        1. Use break finder to identify natural first chunk boundary
        2. Generate first chunk immediately and start yielding
        3. While yielding first chunk, generate remainder in background
        4. Continue until all content is processed

        This achieves low first-chunk latency by:
        - Finding natural break points (not rigid sentence boundaries)
        - Starting with a short, natural-sounding first chunk
        - Overlapping generation with playback
        """
        # Use break finder to identify optimal split point
        break_point = await self._break_finder.find_break(text)

        self.logger.debug(
            f"Break finder: method={break_point.method}, "
            f"confidence={break_point.confidence:.2f}, "
            f"latency={break_point.latency_ms:.1f}ms, "
            f"first_chunk='{break_point.first_chunk[:50]}...'"
        )

        first_text = break_point.first_chunk
        remainder_text = break_point.remainder

        if not first_text:
            return

        # Queue to hold pre-generated audio for remainder
        audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=2)
        generation_complete = asyncio.Event()

        async def generate_remainder() -> None:
            """Background task to generate remaining text."""
            try:
                if remainder_text.strip():
                    result = await self.synthesize(remainder_text, voice_id, speed, emotion)
                    await audio_queue.put(result.audio)
            except Exception as e:
                self.logger.error(f"Background generation error: {e}")
            finally:
                await audio_queue.put(None)  # Signal completion
                generation_complete.set()

        # Generate first chunk immediately
        self.logger.debug(f"Generating first chunk: '{first_text}'")
        first_result = await self.synthesize(first_text, voice_id, speed, emotion)
        first_audio = first_result.audio

        # Start background generation of remainder
        background_task = None
        if remainder_text.strip():
            background_task = asyncio.create_task(generate_remainder())

        # Yield first chunk audio
        for i in range(0, len(first_audio), chunk_size):
            chunk = first_audio[i : i + chunk_size]
            yield chunk
            await asyncio.sleep(0)  # Allow other tasks to run

        # Yield remainder from queue
        if background_task:
            while True:
                try:
                    audio = await asyncio.wait_for(audio_queue.get(), timeout=30.0)
                    if audio is None:
                        break  # Generation complete

                    for i in range(0, len(audio), chunk_size):
                        chunk = audio[i : i + chunk_size]
                        yield chunk
                        await asyncio.sleep(0)

                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for background generation")
                    break

            # Ensure background task is cleaned up
            if not background_task.done():
                background_task.cancel()
                try:
                    await background_task
                except asyncio.CancelledError:
                    pass

    async def _stream_api(
        self,
        text: str,
        reference_id: str | None,
        speed: float,
        chunk_size: int,
    ) -> AsyncIterator[np.ndarray]:
        """Stream synthesis using Fish Audio API."""
        import io

        import soundfile as sf

        loop = asyncio.get_event_loop()

        def _sync_stream():
            kwargs = {"text": text, "speed": speed}
            if reference_id:
                kwargs["reference_id"] = reference_id
            return self._client.tts.stream(**kwargs)

        audio_stream = await loop.run_in_executor(None, _sync_stream)

        buffer = b""
        for audio_chunk in audio_stream:
            buffer += audio_chunk.data

            while len(buffer) >= chunk_size * 2:
                chunk_bytes = buffer[: chunk_size * 2]
                buffer = buffer[chunk_size * 2 :]

                try:
                    audio_data, sr = sf.read(io.BytesIO(chunk_bytes))
                    if sr != self.sample_rate:
                        import scipy.signal

                        num_samples = int(len(audio_data) * self.sample_rate / sr)
                        audio_data = scipy.signal.resample(audio_data, num_samples)
                    yield np.array(audio_data, dtype=np.float32)
                except Exception:
                    continue

        if buffer:
            try:
                audio_data, sr = sf.read(io.BytesIO(buffer))
                if sr != self.sample_rate:
                    import scipy.signal

                    num_samples = int(len(audio_data) * self.sample_rate / sr)
                    audio_data = scipy.signal.resample(audio_data, num_samples)
                yield np.array(audio_data, dtype=np.float32)
            except Exception:
                pass

    async def clone_voice(
        self,
        reference_audio: np.ndarray,
        voice_id: str,
    ) -> None:
        """
        Clone a voice from reference audio.

        For API mode, this uploads the audio to Fish Audio and creates a voice model.
        For local mode, this saves the reference audio for later use.

        Args:
            reference_audio: Reference audio samples (numpy array)
            voice_id: Identifier for the cloned voice
        """
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, reference_audio, self.sample_rate)
            temp_path = f.name

        if self.use_api and self._client:
            try:
                loop = asyncio.get_event_loop()

                def _create_voice():
                    with open(temp_path, "rb") as audio_file:
                        return self._client.voices.create(
                            title=voice_id,
                            audio=audio_file,
                        )

                voice = await loop.run_in_executor(None, _create_voice)
                self._cloned_voices[voice_id] = voice.id
                self.logger.info(f"Voice cloned via API: {voice_id} -> {voice.id}")
            except Exception as e:
                self.logger.error(f"Failed to clone voice via API: {e}")
                self._cloned_voices[voice_id] = temp_path
        else:
            self._cloned_voices[voice_id] = temp_path
            self.logger.info(f"Voice reference saved locally: {voice_id}")

    def get_available_voices(self) -> list[str]:
        """Get list of available voice IDs."""
        voices = list(self._cloned_voices.keys())

        if self.reference_id:
            voices.append(self.reference_id)

        return voices

    def get_first_chunk_latency(self) -> float:
        """Get the latency of the first audio chunk in milliseconds."""
        return self._first_chunk_latency_ms
