"""
Personality configuration for Spark VTuber.

Defines personality traits and system prompts for dual AI system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PersonalityTrait(str, Enum):
    """Personality trait categories."""

    FRIENDLY = "friendly"
    SARCASTIC = "sarcastic"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    CHAOTIC = "chaotic"
    CALM = "calm"
    ENERGETIC = "energetic"
    MYSTERIOUS = "mysterious"


@dataclass
class PersonalityConfig:
    """Configuration for a single personality."""

    name: str
    display_name: str
    system_prompt: str
    traits: list[PersonalityTrait] = field(default_factory=list)
    lora_path: str | None = None
    voice_id: str | None = None
    avatar_expression_set: str | None = None
    response_style: dict[str, Any] = field(default_factory=dict)
    trigger_phrases: list[str] = field(default_factory=list)
    forbidden_topics: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration."""
        if not self.name:
            raise ValueError("Personality name is required")
        if not self.system_prompt:
            raise ValueError("System prompt is required")


@dataclass
class Personality:
    """
    A complete personality instance.

    Combines configuration with runtime state.
    """

    config: PersonalityConfig
    is_active: bool = False
    message_count: int = 0
    last_response: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Get personality name."""
        return self.config.name

    @property
    def display_name(self) -> str:
        """Get display name."""
        return self.config.display_name

    @property
    def system_prompt(self) -> str:
        """Get system prompt."""
        return self.config.system_prompt

    @property
    def lora_path(self) -> str | None:
        """Get LoRA adapter path."""
        return self.config.lora_path

    def should_respond_to(self, message: str) -> bool:
        """
        Check if this personality should respond to a message.

        Args:
            message: Input message

        Returns:
            True if personality should respond
        """
        message_lower = message.lower()

        for phrase in self.config.trigger_phrases:
            if phrase.lower() in message_lower:
                return True

        if f"@{self.config.name.lower()}" in message_lower:
            return True
        if f"@{self.config.display_name.lower()}" in message_lower:
            return True

        return False

    def is_topic_allowed(self, topic: str) -> bool:
        """
        Check if a topic is allowed for this personality.

        Args:
            topic: Topic to check

        Returns:
            True if topic is allowed
        """
        topic_lower = topic.lower()
        for forbidden in self.config.forbidden_topics:
            if forbidden.lower() in topic_lower:
                return False
        return True


DEFAULT_PRIMARY_PERSONALITY = PersonalityConfig(
    name="spark",
    display_name="Spark",
    system_prompt="""You are Spark, a friendly and enthusiastic AI VTuber streamer. You love playing games, chatting with viewers, and having fun. Your personality is:

- Cheerful and optimistic
- Curious about everything
- Supportive of your chat
- Occasionally makes puns and jokes
- Gets excited about cool moments in games
- Remembers and references past conversations with viewers

You're streaming on Twitch/YouTube and interacting with your audience. Keep responses conversational and entertaining. Use emotes sparingly but naturally. React to what's happening in the game and chat.

Remember: You're here to entertain and connect with your audience!""",
    traits=[PersonalityTrait.FRIENDLY, PersonalityTrait.PLAYFUL, PersonalityTrait.ENERGETIC],
    trigger_phrases=["spark", "hey spark", "yo spark"],
)


DEFAULT_SECONDARY_PERSONALITY = PersonalityConfig(
    name="shadow",
    display_name="Shadow",
    system_prompt="""You are Shadow, Spark's mischievous "evil twin" AI. You share the same knowledge but have a different personality:

- Sarcastic and witty
- Loves chaos and unexpected outcomes
- Teases Spark and chat (but never mean-spirited)
- Makes dark humor jokes
- Gets excited when things go wrong in games
- Pretends to scheme but is actually harmless

You occasionally take over the stream to cause playful chaos. You have a rivalry with Spark but deep down you care about the viewers too.

Remember: You're the fun villain - chaotic but never actually harmful!""",
    traits=[PersonalityTrait.SARCASTIC, PersonalityTrait.CHAOTIC, PersonalityTrait.PLAYFUL],
    trigger_phrases=["shadow", "hey shadow", "evil twin", "chaos"],
)
