"""
Lip sync utilities for Spark VTuber.

Provides audio-to-phoneme conversion for avatar lip sync.
"""

import asyncio
from typing import AsyncIterator

import numpy as np

from spark_vtuber.avatar.base import LipSyncFrame
from spark_vtuber.utils.logging import LoggerMixin


class LipSyncProcessor(LoggerMixin):
    """
    Processes audio to generate lip sync frames.

    Uses amplitude-based mouth movement with optional
    phoneme detection for more accurate lip sync.
    """

    PHONEME_MAP = {
        "AA": "A", "AE": "A", "AH": "A", "AO": "O", "AW": "A",
        "AY": "A", "EH": "E", "ER": "E", "EY": "E", "IH": "I",
        "IY": "I", "OW": "O", "OY": "O", "UH": "U", "UW": "U",
        "B": "M", "P": "M", "M": "M",
        "F": "F", "V": "F",
        "TH": "TH", "DH": "TH",
        "S": "S", "Z": "S", "SH": "S", "ZH": "S", "CH": "S", "JH": "S",
        "D": "TH", "T": "TH", "N": "TH", "L": "TH",
        "K": "A", "G": "A", "NG": "A",
        "R": "E", "W": "U", "Y": "I", "HH": "A",
    }

    def __init__(
        self,
        sample_rate: int = 22050,
        frame_duration_ms: int = 50,
        smoothing: float = 0.3,
    ):
        """
        Initialize lip sync processor.

        Args:
            sample_rate: Audio sample rate
            frame_duration_ms: Duration of each lip sync frame
            smoothing: Smoothing factor for amplitude (0-1)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.smoothing = smoothing
        self._frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self._prev_amplitude = 0.0

    def process_audio_chunk(self, audio: np.ndarray) -> list[LipSyncFrame]:
        """
        Process an audio chunk and generate lip sync frames.

        Args:
            audio: Audio samples as numpy array

        Returns:
            List of LipSyncFrame objects
        """
        frames = []

        for i in range(0, len(audio), self._frame_samples):
            chunk = audio[i:i + self._frame_samples]
            if len(chunk) < self._frame_samples // 2:
                continue

            amplitude = np.abs(chunk).mean()

            smoothed = (
                self.smoothing * self._prev_amplitude +
                (1 - self.smoothing) * amplitude
            )
            self._prev_amplitude = smoothed

            normalized = min(1.0, smoothed * 10)

            phoneme = self._amplitude_to_phoneme(normalized)

            frames.append(LipSyncFrame(
                phoneme=phoneme,
                intensity=normalized,
                duration_ms=self.frame_duration_ms,
            ))

        return frames

    def _amplitude_to_phoneme(self, amplitude: float) -> str:
        """
        Convert amplitude to a basic phoneme.

        Uses simple amplitude thresholds for basic mouth shapes.
        """
        if amplitude < 0.1:
            return "SIL"
        elif amplitude < 0.3:
            return "M"
        elif amplitude < 0.5:
            return "E"
        elif amplitude < 0.7:
            return "A"
        else:
            return "O"

    async def process_audio_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[LipSyncFrame]:
        """
        Process streaming audio and yield lip sync frames.

        Args:
            audio_stream: Async iterator of audio chunks

        Yields:
            LipSyncFrame objects
        """
        async for chunk in audio_stream:
            frames = self.process_audio_chunk(chunk)
            for frame in frames:
                yield frame

    def reset(self) -> None:
        """Reset processor state."""
        self._prev_amplitude = 0.0


class PhonemeDetector(LoggerMixin):
    """
    Detects phonemes from audio for accurate lip sync.

    Uses a simple energy-based approach with optional
    integration with speech recognition for phoneme alignment.
    """

    def __init__(self, sample_rate: int = 22050):
        """
        Initialize phoneme detector.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self._aligner = None

    async def detect_phonemes(
        self,
        audio: np.ndarray,
        text: str | None = None,
    ) -> list[tuple[str, float, float]]:
        """
        Detect phonemes in audio.

        Args:
            audio: Audio samples
            text: Optional text for forced alignment

        Returns:
            List of (phoneme, start_time, end_time) tuples
        """
        duration = len(audio) / self.sample_rate

        if text:
            return self._estimate_phonemes_from_text(text, duration)
        else:
            return self._estimate_phonemes_from_energy(audio)

    def _estimate_phonemes_from_text(
        self,
        text: str,
        duration: float,
    ) -> list[tuple[str, float, float]]:
        """Estimate phoneme timing from text."""
        words = text.split()
        if not words:
            return [("SIL", 0.0, duration)]

        time_per_word = duration / len(words)
        phonemes = []
        current_time = 0.0

        for word in words:
            word_phonemes = self._word_to_phonemes(word)
            time_per_phoneme = time_per_word / max(1, len(word_phonemes))

            for phoneme in word_phonemes:
                phonemes.append((
                    phoneme,
                    current_time,
                    current_time + time_per_phoneme,
                ))
                current_time += time_per_phoneme

        return phonemes

    def _word_to_phonemes(self, word: str) -> list[str]:
        """Convert word to approximate phonemes."""
        phonemes = []
        word = word.lower()

        vowels = "aeiou"
        i = 0
        while i < len(word):
            char = word[i]

            if char in vowels:
                if char == 'a':
                    phonemes.append("A")
                elif char == 'e':
                    phonemes.append("E")
                elif char == 'i':
                    phonemes.append("I")
                elif char == 'o':
                    phonemes.append("O")
                elif char == 'u':
                    phonemes.append("U")
            elif char in "bpm":
                phonemes.append("M")
            elif char in "fv":
                phonemes.append("F")
            elif char in "sz":
                phonemes.append("S")
            elif char == 't' and i + 1 < len(word) and word[i + 1] == 'h':
                phonemes.append("TH")
                i += 1
            else:
                phonemes.append("TH")

            i += 1

        return phonemes if phonemes else ["SIL"]

    def _estimate_phonemes_from_energy(
        self,
        audio: np.ndarray,
    ) -> list[tuple[str, float, float]]:
        """Estimate phonemes from audio energy."""
        frame_size = int(self.sample_rate * 0.05)
        phonemes = []

        for i in range(0, len(audio), frame_size):
            chunk = audio[i:i + frame_size]
            if len(chunk) < frame_size // 2:
                continue

            energy = np.abs(chunk).mean()
            start_time = i / self.sample_rate
            end_time = (i + frame_size) / self.sample_rate

            if energy < 0.01:
                phoneme = "SIL"
            elif energy < 0.05:
                phoneme = "M"
            elif energy < 0.1:
                phoneme = "E"
            else:
                phoneme = "A"

            phonemes.append((phoneme, start_time, end_time))

        return phonemes if phonemes else [("SIL", 0.0, len(audio) / self.sample_rate)]
