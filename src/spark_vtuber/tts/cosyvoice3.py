"""
CosyVoice 3 streaming TTS implementation for Spark VTuber.

CosyVoice 3 is Alibaba's state-of-the-art streaming TTS system with:
- TRUE bi-streaming: text-in streaming + audio-out streaming
- 150ms first-chunk latency (vs 300ms for LLMVoX)
- 100+ emotion/tone controls via instruct prompts
- Zero-shot voice cloning with 15-30s reference samples
- 97% speaker similarity with just 15s of reference audio
- 9 languages + 18 Chinese dialects

Based on: https://arxiv.org/abs/2505.17589 (May 2025)
Code: https://github.com/FunAudioLLM/CosyVoice
Model: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512

Key features for VTuber streaming:
- Bi-streaming: accepts text generator from LLM for true end-to-end streaming
- Emotion controls: "(excited) Hello!" syntax parsed and converted to instruct prompts
- Fine-grained controls: [breath], [laughter], <strong></strong> tags
- Low VRAM: ~2-4GB for 0.5B model
"""

import asyncio
import re
import sys
import time
from collections.abc import AsyncIterator, Generator
from pathlib import Path

import numpy as np

from spark_vtuber.tts.base import BaseTTS, TTSResult

# Emotion tag to CosyVoice 3 instruct prompt mapping
EMOTION_TO_INSTRUCT = {
    "excited": "You are a helpful assistant. Please speak with excitement and enthusiasm.<|endofprompt|>",
    "happy": "You are a helpful assistant. Please speak happily and cheerfully.<|endofprompt|>",
    "sad": "You are a helpful assistant. Please speak sadly and with melancholy.<|endofprompt|>",
    "angry": "You are a helpful assistant. Please speak angrily and with frustration.<|endofprompt|>",
    "surprised": "You are a helpful assistant. Please speak with surprise and wonder.<|endofprompt|>",
    "fearful": "You are a helpful assistant. Please speak with fear and anxiety.<|endofprompt|>",
    "disgusted": "You are a helpful assistant. Please speak with disgust.<|endofprompt|>",
    "neutral": "You are a helpful assistant.<|endofprompt|>",
    "whisper": "You are a helpful assistant. Please speak in a soft whisper.<|endofprompt|>",
    "shout": "You are a helpful assistant. Please speak loudly as if shouting.<|endofprompt|>",
    "fast": "You are a helpful assistant. Please speak as fast as possible.<|endofprompt|>",
    "slow": "You are a helpful assistant. Please speak slowly and deliberately.<|endofprompt|>",
    "cute": "You are a helpful assistant. Please speak in a cute and adorable way.<|endofprompt|>",
    "serious": "You are a helpful assistant. Please speak seriously and formally.<|endofprompt|>",
    "playful": "You are a helpful assistant. Please speak playfully and teasingly.<|endofprompt|>",
    "sarcastic": "You are a helpful assistant. Please speak sarcastically.<|endofprompt|>",
    "loving": "You are a helpful assistant. Please speak lovingly and warmly.<|endofprompt|>",
    "nervous": "You are a helpful assistant. Please speak nervously with hesitation.<|endofprompt|>",
    "confident": "You are a helpful assistant. Please speak confidently and assertively.<|endofprompt|>",
    "tired": "You are a helpful assistant. Please speak tiredly and wearily.<|endofprompt|>",
}

# Regex pattern to extract emotion tags like "(excited)" or "(happy)"
EMOTION_PATTERN = re.compile(r"^\s*\((\w+)\)\s*")


class CosyVoice3TTS(BaseTTS):
    """
    CosyVoice 3 streaming TTS implementation.

    Provides TRUE bi-streaming TTS that generates audio as text tokens arrive,
    with 150ms first-chunk latency and 100+ emotion controls.

    Memory usage: ~2-4GB VRAM (0.5B parameters)
    Latency: ~150ms first chunk (true streaming mode)
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        model_dir: str | None = None,
        reference_audio_path: str | None = None,
        reference_text: str | None = None,
        default_instruct: str | None = None,
        device: str = "cuda",
        load_vllm: bool = False,
        load_trt: bool = False,
        fp16: bool = False,
        **kwargs,
    ):
        """
        Initialize CosyVoice 3 TTS.

        Args:
            sample_rate: Output sample rate (24000 for CosyVoice 3)
            model_dir: Path to CosyVoice 3 model directory (auto-downloads if not specified)
            reference_audio_path: Path to reference audio for zero-shot voice cloning
            reference_text: Transcript of reference audio (improves cloning quality)
            default_instruct: Default instruct prompt for emotion/style control
            device: Device to run inference on (cuda, cpu)
            load_vllm: Enable vLLM acceleration (requires vLLM 0.11+)
            load_trt: Enable TensorRT acceleration
            fp16: Use FP16 precision
            **kwargs: Additional arguments
        """
        super().__init__(sample_rate, **kwargs)
        self.model_dir = model_dir or "./models/cosyvoice3"
        self.reference_audio_path = reference_audio_path
        self.reference_text = reference_text or "You are a helpful assistant.<|endofprompt|>"
        self.default_instruct = default_instruct or "You are a helpful assistant.<|endofprompt|>"
        self.device = device
        self.load_vllm = load_vllm
        self.load_trt = load_trt
        self.fp16 = fp16

        # Model components (loaded lazily)
        self._cosyvoice = None
        self._reference_audio = None
        self._first_chunk_latency_ms: float = 0.0
        self._cloned_voices: dict[str, dict] = {}

    async def load(self) -> None:
        """Load the CosyVoice 3 model components."""
        if self._loaded:
            self.logger.warning("CosyVoice 3 TTS already loaded")
            return

        self.logger.info("Loading CosyVoice 3 TTS...")
        start_time = time.time()

        loop = asyncio.get_event_loop()

        def _load_sync():

            # Add CosyVoice to path if needed
            cosyvoice_path = Path(self.model_dir).parent / "CosyVoice"
            if cosyvoice_path.exists():
                sys.path.insert(0, str(cosyvoice_path))
                matcha_path = cosyvoice_path / "third_party" / "Matcha-TTS"
                if matcha_path.exists():
                    sys.path.insert(0, str(matcha_path))

            # Download model if not present
            model_dir = self._ensure_model()

            # Import CosyVoice after ensuring model is downloaded
            from cosyvoice.cli.cosyvoice import AutoModel

            # Register vLLM model if using vLLM
            if self.load_vllm:
                try:
                    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
                    from vllm import ModelRegistry
                    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
                except ImportError:
                    self.logger.warning("vLLM not available, falling back to standard inference")
                    self.load_vllm = False

            # Load model
            self.logger.info(f"Loading CosyVoice 3 from {model_dir}...")
            cosyvoice = AutoModel(
                model_dir=str(model_dir),
                load_trt=self.load_trt,
                load_vllm=self.load_vllm,
                fp16=self.fp16,
            )

            # Load reference audio if specified
            reference_audio = None
            if self.reference_audio_path and Path(self.reference_audio_path).exists():
                import torchaudio
                self.logger.info(f"Loading reference audio from {self.reference_audio_path}...")
                waveform, sr = torchaudio.load(self.reference_audio_path)
                if sr != cosyvoice.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, cosyvoice.sample_rate)
                reference_audio = self.reference_audio_path

            return {
                "cosyvoice": cosyvoice,
                "reference_audio": reference_audio,
                "sample_rate": cosyvoice.sample_rate,
            }

        models = await loop.run_in_executor(None, _load_sync)

        self._cosyvoice = models["cosyvoice"]
        self._reference_audio = models["reference_audio"]
        self.sample_rate = models["sample_rate"]

        self._loaded = True
        load_time = time.time() - start_time
        self.logger.info(f"CosyVoice 3 TTS loaded in {load_time:.2f}s")

    def _ensure_model(self) -> Path:
        """Ensure CosyVoice 3 model is downloaded."""
        from huggingface_hub import snapshot_download

        model_dir = Path(self.model_dir)

        # Check if model already exists
        if (model_dir / "cosyvoice3.yaml").exists():
            self.logger.info(f"CosyVoice 3 model found at {model_dir}")
            return model_dir

        # Download from HuggingFace
        self.logger.info("Downloading CosyVoice 3 model from HuggingFace...")
        model_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )

        self.logger.info(f"CosyVoice 3 model downloaded to {model_dir}")
        return model_dir

    async def unload(self) -> None:
        """Unload the CosyVoice 3 model."""
        if not self._loaded:
            return

        self.logger.info("Unloading CosyVoice 3 TTS")

        self._cosyvoice = None
        self._reference_audio = None
        self._cloned_voices.clear()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False

    def _parse_emotion(self, text: str) -> tuple[str, str]:
        """
        Parse emotion tag from text.

        Supports format: "(emotion) text" -> returns (instruct_prompt, clean_text)

        Args:
            text: Input text potentially with emotion tag

        Returns:
            Tuple of (instruct_prompt, clean_text)
        """
        match = EMOTION_PATTERN.match(text)
        if match:
            emotion = match.group(1).lower()
            clean_text = text[match.end():]
            instruct = EMOTION_TO_INSTRUCT.get(emotion, self.default_instruct)
            return instruct, clean_text
        return self.default_instruct, text

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
            text: Text to synthesize (can include emotion tags like "(excited) Hello!")
            voice_id: Voice reference ID for cloned voices
            speed: Speech speed multiplier
            emotion: Emotion override (overrides emotion tag in text)

        Returns:
            TTSResult with audio data
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        start_time = time.time()

        # Collect all audio chunks from streaming
        audio_chunks = []
        async for chunk in self.synthesize_stream(text, voice_id, speed, emotion):
            audio_chunks.append(chunk)

        audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)

        latency_ms = (time.time() - start_time) * 1000
        duration = len(audio) / self.sample_rate

        return TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            latency_ms=latency_ms,
            metadata={
                "engine": "cosyvoice3",
                "first_chunk_latency_ms": self._first_chunk_latency_ms,
                "emotion": emotion,
            },
        )

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

        CosyVoice 3 generates audio chunk-by-chunk as text is processed,
        providing true bi-streaming with ~150ms first-chunk latency.

        Args:
            text: Text to synthesize (can include emotion tags)
            voice_id: Voice reference ID for cloned voices
            speed: Speech speed multiplier
            emotion: Emotion override
            chunk_size: Audio chunk size in samples (not used, CosyVoice handles chunking)

        Yields:
            Audio chunks as numpy arrays
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        # Parse emotion from text or use override
        if emotion:
            instruct = EMOTION_TO_INSTRUCT.get(emotion.lower(), self.default_instruct)
            clean_text = text
        else:
            instruct, clean_text = self._parse_emotion(text)

        # Get reference audio for voice cloning
        reference_audio = self._reference_audio

        if voice_id and voice_id in self._cloned_voices:
            voice_info = self._cloned_voices[voice_id]
            reference_audio = voice_info.get("audio_path")

        loop = asyncio.get_event_loop()
        start_time = time.time()
        first_chunk = True

        # Capture instruct for use in nested function
        instruct_text = instruct

        def _generate_audio() -> Generator[np.ndarray, None, None]:
            """Generate audio chunks synchronously."""
            import torch

            # Choose inference method based on available reference
            if reference_audio:
                # Zero-shot with instruct for emotion control
                generator = self._cosyvoice.inference_instruct2(
                    tts_text=clean_text,
                    instruct_text=instruct_text,
                    prompt_wav=reference_audio,
                    stream=True,
                    speed=speed,
                )
            else:
                # Cross-lingual inference without reference
                generator = self._cosyvoice.inference_cross_lingual(
                    tts_text=f"<|en|>{clean_text}",
                    prompt_wav="./asset/zero_shot_prompt.wav" if Path("./asset/zero_shot_prompt.wav").exists() else None,
                    stream=True,
                    speed=speed,
                )

            for output in generator:
                speech = output["tts_speech"]
                if isinstance(speech, torch.Tensor):
                    speech = speech.cpu().numpy()
                if speech.ndim > 1:
                    speech = speech.squeeze()
                yield speech.astype(np.float32)

        # Run generator in thread pool and yield chunks
        audio_queue = asyncio.Queue()
        generation_done = asyncio.Event()

        async def _run_generation():
            """Run audio generation in background."""
            try:
                gen = await loop.run_in_executor(None, _generate_audio)
                for chunk in gen:
                    await audio_queue.put(chunk)
            except Exception as e:
                self.logger.error(f"Audio generation error: {e}")
            finally:
                generation_done.set()

        # Start generation task
        generation_task = asyncio.create_task(_run_generation())

        try:
            while not generation_done.is_set() or not audio_queue.empty():
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    if first_chunk:
                        self._first_chunk_latency_ms = (time.time() - start_time) * 1000
                        first_chunk = False
                    yield chunk
                except asyncio.TimeoutError:
                    continue
        finally:
            generation_task.cancel()
            try:
                await generation_task
            except asyncio.CancelledError:
                pass

    async def synthesize_from_text_stream(
        self,
        text_generator: AsyncIterator[str],
        voice_id: str | None = None,
        speed: float = 1.0,
        emotion: str | None = None,
    ) -> AsyncIterator[np.ndarray]:
        """
        Synthesize speech from streaming text input (bi-streaming).

        This is the TRUE end-to-end streaming mode where text arrives
        token-by-token from the LLM and audio is generated incrementally.

        Args:
            text_generator: Async generator yielding text chunks from LLM
            voice_id: Voice reference ID for cloned voices
            speed: Speech speed multiplier
            emotion: Emotion override

        Yields:
            Audio chunks as numpy arrays
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        # Parse emotion from first chunk or use override (unused for now, kept for future instruct support)
        _ = EMOTION_TO_INSTRUCT.get(emotion.lower(), self.default_instruct) if emotion else self.default_instruct

        # Get reference audio
        reference_audio = self._reference_audio
        reference_text = self.reference_text

        if voice_id and voice_id in self._cloned_voices:
            voice_info = self._cloned_voices[voice_id]
            reference_audio = voice_info.get("audio_path")
            reference_text = voice_info.get("text", self.reference_text)

        loop = asyncio.get_event_loop()
        start_time = time.time()
        first_chunk = True

        # Collect text chunks into a sync generator for CosyVoice
        text_buffer: list[str] = []

        async def _collect_text():
            """Collect text from async generator."""
            async for chunk in text_generator:
                text_buffer.append(chunk)

        # Start collecting text
        collect_task = asyncio.create_task(_collect_text())

        def _sync_text_generator() -> Generator[str, None, None]:
            """Sync generator that yields collected text."""
            idx = 0
            while True:
                if idx < len(text_buffer):
                    yield text_buffer[idx]
                    idx += 1
                elif collect_task.done():
                    break
                else:
                    time.sleep(0.01)

        def _generate_audio() -> Generator[np.ndarray, None, None]:
            """Generate audio from text stream."""
            import torch

            if reference_audio:
                generator = self._cosyvoice.inference_zero_shot(
                    tts_text=_sync_text_generator(),
                    prompt_text=reference_text,
                    prompt_wav=reference_audio,
                    stream=True,
                    speed=speed,
                )
            else:
                # Fallback to cross-lingual
                full_text = "".join(text_buffer)
                generator = self._cosyvoice.inference_cross_lingual(
                    tts_text=f"<|en|>{full_text}",
                    prompt_wav=None,
                    stream=True,
                    speed=speed,
                )

            for output in generator:
                speech = output["tts_speech"]
                if isinstance(speech, torch.Tensor):
                    speech = speech.cpu().numpy()
                if speech.ndim > 1:
                    speech = speech.squeeze()
                yield speech.astype(np.float32)

        # Run generator and yield chunks
        audio_queue = asyncio.Queue()
        generation_done = asyncio.Event()

        async def _run_generation():
            """Run audio generation in background."""
            try:
                gen = await loop.run_in_executor(None, _generate_audio)
                for chunk in gen:
                    await audio_queue.put(chunk)
            except Exception as e:
                self.logger.error(f"Audio generation error: {e}")
            finally:
                generation_done.set()

        generation_task = asyncio.create_task(_run_generation())

        try:
            while not generation_done.is_set() or not audio_queue.empty():
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    if first_chunk:
                        self._first_chunk_latency_ms = (time.time() - start_time) * 1000
                        first_chunk = False
                    yield chunk
                except asyncio.TimeoutError:
                    continue
        finally:
            collect_task.cancel()
            generation_task.cancel()
            try:
                await collect_task
            except asyncio.CancelledError:
                pass
            try:
                await generation_task
            except asyncio.CancelledError:
                pass

    async def clone_voice(
        self,
        reference_audio: np.ndarray,
        voice_id: str,
        reference_text: str | None = None,
    ) -> None:
        """
        Clone a voice from reference audio.

        CosyVoice 3 achieves 97% speaker similarity with just 15s of reference audio.

        Args:
            reference_audio: Reference audio samples (15-30s recommended)
            voice_id: Identifier for the cloned voice
            reference_text: Transcript of reference audio (improves quality)
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        import tempfile

        import soundfile as sf

        # Save reference audio to temp file
        temp_dir = Path(tempfile.gettempdir()) / "cosyvoice3_voices"
        temp_dir.mkdir(parents=True, exist_ok=True)
        audio_path = temp_dir / f"{voice_id}.wav"

        sf.write(str(audio_path), reference_audio, self.sample_rate)

        # Store voice info
        self._cloned_voices[voice_id] = {
            "audio_path": str(audio_path),
            "text": reference_text or "You are a helpful assistant.<|endofprompt|>",
        }

        # Optionally add to CosyVoice's speaker info
        if reference_text:
            try:
                self._cosyvoice.add_zero_shot_spk(
                    prompt_text=reference_text,
                    prompt_wav=str(audio_path),
                    zero_shot_spk_id=voice_id,
                )
                self.logger.info(f"Voice '{voice_id}' cloned and registered with CosyVoice")
            except Exception as e:
                self.logger.warning(f"Could not register voice with CosyVoice: {e}")

        self.logger.info(f"Voice '{voice_id}' cloned from {len(reference_audio) / self.sample_rate:.1f}s of audio")

    def get_available_voices(self) -> list[str]:
        """Get list of available voice IDs."""
        voices = list(self._cloned_voices.keys())

        # Add built-in voices from CosyVoice if available
        if self._cosyvoice:
            try:
                builtin = self._cosyvoice.list_available_spks()
                voices.extend(builtin)
            except Exception:
                pass

        return voices

    def get_first_chunk_latency(self) -> float:
        """Get the latency of the first audio chunk in milliseconds."""
        return self._first_chunk_latency_ms

    @staticmethod
    def get_supported_emotions() -> list[str]:
        """Get list of supported emotion tags."""
        return list(EMOTION_TO_INSTRUCT.keys())
