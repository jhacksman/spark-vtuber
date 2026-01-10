"""
NVIDIA Parakeet TDT STT implementation for Spark VTuber.

Provides ultra-fast, low-latency speech recognition optimized for real-time streaming.
Parakeet TDT 0.6B V2 achieves 6.05% WER with 3386x RTFx - 16x faster than Whisper Turbo.
"""

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from spark_vtuber.stt.base import BaseSTT, STTResult, STTSegment


class ParakeetSTT(BaseSTT):
    """
    NVIDIA Parakeet TDT STT implementation.

    Supports:
    - Ultra-fast inference (3386x RTFx)
    - Low-latency streaming transcription
    - Automatic punctuation and capitalization
    - Word-level timestamps

    Memory usage: ~4GB VRAM
    Latency: Ultra-low for real-time streaming
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        **kwargs,
    ):
        """
        Initialize Parakeet TDT STT.

        Args:
            model_name: HuggingFace model name or local path
            device: Device for inference (cuda/cpu)
            compute_type: Compute precision (float16/float32)
            language: Target language (English only for Parakeet)
            **kwargs: Additional arguments
        """
        super().__init__(language, **kwargs)
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._processor = None

    async def load(self) -> None:
        """Load the Parakeet TDT model."""
        if self._loaded:
            self.logger.warning("STT already loaded")
            return

        self.logger.info(f"Loading Parakeet TDT model: {self.model_name}")
        start_time = time.time()

        loop = asyncio.get_event_loop()

        def _load_sync():
            import torch

            # Try NeMo toolkit first (recommended for Parakeet)
            try:
                import nemo.collections.asr as nemo_asr

                self.logger.info("Loading Parakeet via NeMo toolkit...")
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name
                )

                if self.device == "cuda" and torch.cuda.is_available():
                    model = model.cuda()

                if self.compute_type == "float16":
                    model = model.half()

                model.eval()
                return {"model": model, "backend": "nemo"}

            except ImportError:
                self.logger.info("NeMo not available, trying transformers...")

            # Fallback to transformers/HuggingFace
            try:
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

                processor = AutoProcessor.from_pretrained(self.model_name)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.compute_type == "float16" else torch.float32,
                    device_map=self.device,
                )
                model.eval()
                return {"model": model, "processor": processor, "backend": "transformers"}

            except Exception as e:
                self.logger.warning(f"Transformers loading failed: {e}")

            # Last resort: try loading as a generic NeMo model
            try:
                import nemo.collections.asr as nemo_asr

                # Try loading from HuggingFace hub
                model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                    model_name=self.model_name
                )

                if self.device == "cuda" and torch.cuda.is_available():
                    model = model.cuda()

                model.eval()
                return {"model": model, "backend": "nemo_rnnt"}

            except Exception as e:
                raise ImportError(
                    f"Failed to load Parakeet TDT model: {e}. "
                    "Install NeMo toolkit: pip install nemo_toolkit[asr]"
                )

        try:
            result = await loop.run_in_executor(None, _load_sync)
            self._model = result["model"]
            self._processor = result.get("processor")
            self._backend = result["backend"]
            self._loaded = True

            load_time = time.time() - start_time
            self.logger.info(f"Parakeet TDT loaded in {load_time:.2f}s (backend={self._backend})")

        except Exception as e:
            self.logger.error(f"Failed to load Parakeet TDT: {e}")
            raise

    async def unload(self) -> None:
        """Unload the Parakeet TDT model."""
        if not self._loaded:
            return

        self.logger.info("Unloading Parakeet TDT")

        if self._model:
            del self._model
            self._model = None

        self._processor = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribe audio to text."""
        if not self._loaded:
            raise RuntimeError("STT not loaded")

        start_time = time.time()

        # Parakeet expects 16kHz audio
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        loop = asyncio.get_event_loop()

        def _transcribe_sync():
            import torch

            if self._backend == "nemo" or self._backend == "nemo_rnnt":
                # NeMo transcription
                with torch.no_grad():
                    # NeMo expects audio as a list of numpy arrays or file paths
                    transcriptions = self._model.transcribe([audio])

                    # Get timestamps if available
                    try:
                        # Try to get word-level timestamps
                        hypotheses = self._model.transcribe(
                            [audio],
                            return_hypotheses=True,
                        )
                        if hypotheses and len(hypotheses) > 0:
                            hyp = hypotheses[0]
                            if hasattr(hyp, 'timestep') and hyp.timestep:
                                return {
                                    "text": transcriptions[0] if transcriptions else "",
                                    "timestamps": hyp.timestep,
                                }
                    except Exception:
                        pass

                    return {
                        "text": transcriptions[0] if transcriptions else "",
                        "timestamps": None,
                    }

            elif self._backend == "transformers":
                # Transformers transcription
                inputs = self._processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                )

                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = self._model.generate(**inputs)
                    transcription = self._processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )[0]

                return {"text": transcription, "timestamps": None}

            else:
                raise RuntimeError(f"Unknown backend: {self._backend}")

        result = await loop.run_in_executor(None, _transcribe_sync)

        latency_ms = (time.time() - start_time) * 1000

        # Build segments from timestamps if available
        segments = []
        if result.get("timestamps"):
            timestamps = result["timestamps"]
            words = result["text"].split()
            for i, (word, ts) in enumerate(zip(words, timestamps)):
                segments.append(STTSegment(
                    text=word,
                    start_time=ts.get("start", i * 0.1),
                    end_time=ts.get("end", (i + 1) * 0.1),
                    confidence=ts.get("confidence", 1.0),
                ))
        else:
            # Single segment for entire transcription
            audio_duration = len(audio) / 16000
            segments.append(STTSegment(
                text=result["text"],
                start_time=0.0,
                end_time=audio_duration,
                confidence=1.0,
            ))

        return STTResult(
            text=result["text"],
            segments=segments,
            language=self.language,
            latency_ms=latency_ms,
            metadata={"backend": self._backend, "model": self.model_name},
        )

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        import scipy.signal
        num_samples = int(len(audio) * target_sr / orig_sr)
        return scipy.signal.resample(audio, num_samples).astype(np.float32)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
    ) -> AsyncIterator[STTSegment]:
        """
        Transcribe streaming audio with ultra-low latency.

        Parakeet TDT is optimized for streaming with minimal latency.
        """
        if not self._loaded:
            raise RuntimeError("STT not loaded")

        buffer = np.array([], dtype=np.float32)
        # Shorter chunks for lower latency (Parakeet is fast enough)
        chunk_duration = 1.0  # 1 second chunks for low latency
        chunk_samples = int(sample_rate * chunk_duration)
        overlap_samples = int(sample_rate * 0.2)  # 200ms overlap

        async for chunk in audio_stream:
            if sample_rate != 16000:
                chunk = self._resample(chunk, sample_rate, 16000)
                sample_rate = 16000

            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= chunk_samples:
                audio_chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples - overlap_samples:]

                result = await self.transcribe(audio_chunk, sample_rate)
                for segment in result.segments:
                    if segment.text.strip():
                        yield segment

        # Process remaining buffer
        if len(buffer) > sample_rate * 0.5:  # At least 500ms
            result = await self.transcribe(buffer, sample_rate)
            for segment in result.segments:
                if segment.text.strip():
                    yield segment
