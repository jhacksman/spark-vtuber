"""Tests for personality module."""

import pytest

from spark_vtuber.personality.base import (
    Personality,
    PersonalityConfig,
    PersonalityTrait,
    DEFAULT_PRIMARY_PERSONALITY,
    DEFAULT_SECONDARY_PERSONALITY,
)


class TestPersonalityConfig:
    """Tests for PersonalityConfig class."""

    def test_create_config(self):
        """Test config creation."""
        config = PersonalityConfig(
            name="test",
            display_name="Test",
            system_prompt="You are a test personality.",
        )
        assert config.name == "test"
        assert config.display_name == "Test"
        assert config.system_prompt == "You are a test personality."

    def test_config_with_traits(self):
        """Test config with traits."""
        config = PersonalityConfig(
            name="test",
            display_name="Test",
            system_prompt="Test prompt",
            traits=[PersonalityTrait.FRIENDLY, PersonalityTrait.PLAYFUL],
        )
        assert PersonalityTrait.FRIENDLY in config.traits
        assert PersonalityTrait.PLAYFUL in config.traits

    def test_config_validation_no_name(self):
        """Test config validation without name."""
        with pytest.raises(ValueError, match="name is required"):
            PersonalityConfig(
                name="",
                display_name="Test",
                system_prompt="Test prompt",
            )

    def test_config_validation_no_prompt(self):
        """Test config validation without system prompt."""
        with pytest.raises(ValueError, match="System prompt is required"):
            PersonalityConfig(
                name="test",
                display_name="Test",
                system_prompt="",
            )


class TestPersonality:
    """Tests for Personality class."""

    def test_create_personality(self):
        """Test personality creation."""
        config = PersonalityConfig(
            name="test",
            display_name="Test",
            system_prompt="Test prompt",
        )
        personality = Personality(config=config)
        assert personality.name == "test"
        assert personality.display_name == "Test"
        assert personality.system_prompt == "Test prompt"
        assert personality.is_active is False
        assert personality.message_count == 0

    def test_should_respond_to_trigger_phrase(self):
        """Test trigger phrase detection."""
        config = PersonalityConfig(
            name="spark",
            display_name="Spark",
            system_prompt="Test prompt",
            trigger_phrases=["hey spark", "yo spark"],
        )
        personality = Personality(config=config)

        assert personality.should_respond_to("hey spark, how are you?")
        assert personality.should_respond_to("YO SPARK!")
        assert not personality.should_respond_to("hello everyone")

    def test_should_respond_to_mention(self):
        """Test @mention detection."""
        config = PersonalityConfig(
            name="spark",
            display_name="Spark",
            system_prompt="Test prompt",
        )
        personality = Personality(config=config)

        assert personality.should_respond_to("@spark hello!")
        assert personality.should_respond_to("@Spark how are you?")

    def test_is_topic_allowed(self):
        """Test forbidden topic detection."""
        config = PersonalityConfig(
            name="test",
            display_name="Test",
            system_prompt="Test prompt",
            forbidden_topics=["politics", "religion"],
        )
        personality = Personality(config=config)

        assert personality.is_topic_allowed("games")
        assert not personality.is_topic_allowed("politics")
        assert not personality.is_topic_allowed("RELIGION")


class TestDefaultPersonalities:
    """Tests for default personalities."""

    def test_primary_personality(self):
        """Test default primary personality."""
        assert DEFAULT_PRIMARY_PERSONALITY.name == "spark"
        assert DEFAULT_PRIMARY_PERSONALITY.display_name == "Spark"
        assert PersonalityTrait.FRIENDLY in DEFAULT_PRIMARY_PERSONALITY.traits
        assert "spark" in DEFAULT_PRIMARY_PERSONALITY.trigger_phrases

    def test_secondary_personality(self):
        """Test default secondary personality."""
        assert DEFAULT_SECONDARY_PERSONALITY.name == "shadow"
        assert DEFAULT_SECONDARY_PERSONALITY.display_name == "Shadow"
        assert PersonalityTrait.SARCASTIC in DEFAULT_SECONDARY_PERSONALITY.traits
        assert "shadow" in DEFAULT_SECONDARY_PERSONALITY.trigger_phrases
