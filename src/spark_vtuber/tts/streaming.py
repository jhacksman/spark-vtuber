"""
Streaming TTS utilities for Spark VTuber.

Provides sentence-based streaming for low-latency TTS.
"""

import asyncio
import re
from typing import AsyncIterator, Callable

import numpy as np

from spark_vtuber.tts.base import BaseTTS
from spark_vtuber.utils.logging import LoggerMixin


class StreamingTTS(LoggerMixin):
    """
    Streaming TTS wrapper that processes text sentence-by-sentence.

    Enables low-latency TTS by:
    - Detecting sentence boundaries in streaming text
    - Starting synthesis as soon as a sentence is complete
    - Overlapping synthesis with LLM generation
    """

    SENTENCE_ENDINGS = re.compile(r'[.!?]+[\s]*')
    MIN_SENTENCE_LENGTH = 10

    def __init__(
        self,
        tts: BaseTTS,
        buffer_sentences: int = 1,
        min_chunk_chars: int = 50,
    ):
        """
        Initialize streaming TTS.

        Args:
            tts: Underlying TTS engine
            buffer_sentences: Number of sentences to buffer before synthesis
            min_chunk_chars: Minimum characters before forcing synthesis
        """
        self.tts = tts
        self.buffer_sentences = buffer_sentences
        self.min_chunk_chars = min_chunk_chars
        self._buffer = ""
        self._sentences: list[str] = []

    async def process_token(self, token: str) -> AsyncIterator[np.ndarray]:
        """
        Process a single token from LLM output.

        Args:
            token: Token from LLM generation

        Yields:
            Audio chunks when sentences are complete
        """
        self._buffer += token

        sentences = self._extract_sentences()
        for sentence in sentences:
            if len(sentence.strip()) >= self.MIN_SENTENCE_LENGTH:
                async for chunk in self.tts.synthesize_stream(sentence):
                    yield chunk

    def _extract_sentences(self) -> list[str]:
        """Extract complete sentences from buffer."""
        sentences = []

        while True:
            match = self.SENTENCE_ENDINGS.search(self._buffer)
            if not match:
                break

            end_pos = match.end()
            sentence = self._buffer[:end_pos].strip()
            self._buffer = self._buffer[end_pos:]

            if sentence:
                sentences.append(sentence)

        return sentences

    async def flush(self) -> AsyncIterator[np.ndarray]:
        """
        Flush any remaining text in the buffer.

        Yields:
            Audio chunks for remaining text
        """
        if self._buffer.strip():
            async for chunk in self.tts.synthesize_stream(self._buffer.strip()):
                yield chunk
            self._buffer = ""

    def reset(self) -> None:
        """Reset the streaming state."""
        self._buffer = ""
        self._sentences.clear()

    async def synthesize_streaming_text(
        self,
        text_stream: AsyncIterator[str],
        voice_id: str | None = None,
    ) -> AsyncIterator[np.ndarray]:
        """
        Synthesize audio from a streaming text source.

        Args:
            text_stream: Async iterator of text tokens
            voice_id: Optional voice identifier

        Yields:
            Audio chunks
        """
        self.reset()

        async for token in text_stream:
            async for chunk in self.process_token(token):
                yield chunk

        async for chunk in self.flush():
            yield chunk


class AudioChunker(LoggerMixin):
    """
    Utility for chunking audio for real-time playback.

    Handles:
    - Fixed-size chunking for consistent playback
    - Overlap-add for smooth transitions
    - Sample rate conversion if needed
    """

    def __init__(
        self,
        chunk_duration_ms: int = 50,
        sample_rate: int = 22050,
        overlap_ms: int = 5,
    ):
        """
        Initialize audio chunker.

        Args:
            chunk_duration_ms: Duration of each chunk in milliseconds
            sample_rate: Audio sample rate
            overlap_ms: Overlap between chunks for smooth transitions
        """
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = sample_rate
        self.overlap_ms = overlap_ms

        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.overlap_samples = int(sample_rate * overlap_ms / 1000)

        self._buffer = np.array([], dtype=np.float32)

    def add_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        """
        Add audio and return complete chunks.

        Args:
            audio: Audio samples to add

        Returns:
            List of complete audio chunks
        """
        self._buffer = np.concatenate([self._buffer, audio])

        chunks = []
        while len(self._buffer) >= self.chunk_samples:
            chunk = self._buffer[:self.chunk_samples]
            chunks.append(chunk)
            self._buffer = self._buffer[self.chunk_samples - self.overlap_samples:]

        return chunks

    def flush(self) -> np.ndarray | None:
        """
        Flush remaining audio in buffer.

        Returns:
            Remaining audio or None if empty
        """
        if len(self._buffer) > 0:
            result = self._buffer.copy()
            self._buffer = np.array([], dtype=np.float32)
            return result
        return None

    def reset(self) -> None:
        """Reset the chunker state."""
        self._buffer = np.array([], dtype=np.float32)
