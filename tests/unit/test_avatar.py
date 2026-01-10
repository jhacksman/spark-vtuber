"""Tests for avatar module."""

import pytest

from spark_vtuber.avatar.base import Emotion, LipSyncFrame, ExpressionState
from spark_vtuber.avatar.lip_sync import LipSyncProcessor, PhonemeDetector
import numpy as np


class TestEmotion:
    """Tests for Emotion enum."""

    def test_emotion_values(self):
        """Test emotion values."""
        assert Emotion.NEUTRAL.value == "neutral"
        assert Emotion.HAPPY.value == "happy"
        assert Emotion.SAD.value == "sad"
        assert Emotion.ANGRY.value == "angry"
        assert Emotion.SURPRISED.value == "surprised"


class TestLipSyncFrame:
    """Tests for LipSyncFrame class."""

    def test_create_frame(self):
        """Test frame creation."""
        frame = LipSyncFrame(
            phoneme="A",
            intensity=0.8,
            duration_ms=50.0,
        )
        assert frame.phoneme == "A"
        assert frame.intensity == 0.8
        assert frame.duration_ms == 50.0

    def test_default_values(self):
        """Test default values."""
        frame = LipSyncFrame(phoneme="M")
        assert frame.intensity == 1.0
        assert frame.duration_ms == 50.0


class TestExpressionState:
    """Tests for ExpressionState class."""

    def test_create_state(self):
        """Test state creation."""
        state = ExpressionState(
            emotion=Emotion.HAPPY,
            intensity=0.9,
        )
        assert state.emotion == Emotion.HAPPY
        assert state.intensity == 0.9

    def test_default_values(self):
        """Test default values."""
        state = ExpressionState()
        assert state.emotion == Emotion.NEUTRAL
        assert state.intensity == 1.0


class TestLipSyncProcessor:
    """Tests for LipSyncProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a lip sync processor."""
        return LipSyncProcessor(
            sample_rate=22050,
            frame_duration_ms=50,
        )

    def test_process_silent_audio(self, processor):
        """Test processing silent audio."""
        audio = np.zeros(22050, dtype=np.float32)
        frames = processor.process_audio_chunk(audio)
        assert len(frames) > 0
        for frame in frames:
            assert frame.phoneme == "SIL"

    def test_process_loud_audio(self, processor):
        """Test processing loud audio."""
        audio = np.ones(22050, dtype=np.float32) * 0.5
        frames = processor.process_audio_chunk(audio)
        assert len(frames) > 0
        for frame in frames:
            assert frame.phoneme != "SIL"

    def test_reset(self, processor):
        """Test processor reset."""
        audio = np.ones(22050, dtype=np.float32) * 0.5
        processor.process_audio_chunk(audio)
        processor.reset()
        assert processor._prev_amplitude == 0.0


class TestPhonemeDetector:
    """Tests for PhonemeDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a phoneme detector."""
        return PhonemeDetector(sample_rate=22050)

    def test_word_to_phonemes(self, detector):
        """Test word to phoneme conversion."""
        phonemes = detector._word_to_phonemes("hello")
        assert len(phonemes) > 0
        assert all(p in ["A", "E", "I", "O", "U", "M", "F", "S", "TH", "SIL"] for p in phonemes)

    def test_estimate_from_energy(self, detector):
        """Test phoneme estimation from energy."""
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        phonemes = detector._estimate_phonemes_from_energy(audio)
        assert len(phonemes) > 0
        for phoneme, start, end in phonemes:
            assert start < end
