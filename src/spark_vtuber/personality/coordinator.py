"""
Dialogue coordinator for Spark VTuber.

Manages turn-taking and interaction between dual AI personalities.
"""

import asyncio
import random
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

from spark_vtuber.llm.base import BaseLLM
from spark_vtuber.llm.context import ConversationContext, MessageRole
from spark_vtuber.personality.base import Personality
from spark_vtuber.personality.manager import PersonalityManager
from spark_vtuber.utils.logging import LoggerMixin


class TurnType(str, Enum):
    """Type of dialogue turn."""

    USER_MESSAGE = "user_message"
    AI_RESPONSE = "ai_response"
    AI_INTERJECTION = "ai_interjection"
    AI_TAKEOVER = "ai_takeover"


@dataclass
class DialogueTurn:
    """A single turn in the dialogue."""

    turn_type: TurnType
    personality: str | None
    content: str
    triggered_by: str | None = None


class DialogueCoordinator(LoggerMixin):
    """
    Coordinates dialogue between dual AI personalities.

    Handles:
    - Turn-taking decisions
    - Personality interjections
    - Takeover events
    - Response arbitration
    """

    def __init__(
        self,
        llm: BaseLLM,
        personality_manager: PersonalityManager,
        context: ConversationContext,
        interjection_probability: float = 0.1,
        takeover_probability: float = 0.02,
    ):
        """
        Initialize dialogue coordinator.

        Args:
            llm: LLM instance
            personality_manager: Personality manager
            context: Conversation context
            interjection_probability: Chance of secondary AI interjecting
            takeover_probability: Chance of personality takeover
        """
        self.llm = llm
        self.personality_manager = personality_manager
        self.context = context
        self.interjection_probability = interjection_probability
        self.takeover_probability = takeover_probability
        self._turn_history: list[DialogueTurn] = []
        self._is_speaking = False
        self._pending_interjection: str | None = None

    @property
    def is_speaking(self) -> bool:
        """Check if AI is currently speaking."""
        return self._is_speaking

    async def process_user_message(
        self,
        message: str,
        username: str | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """
        Process a user message and generate response(s).

        Args:
            message: User message content
            username: Optional username

        Yields:
            Tuples of (personality_name, response_chunk)
        """
        self._is_speaking = True

        try:
            self._turn_history.append(DialogueTurn(
                turn_type=TurnType.USER_MESSAGE,
                personality=None,
                content=message,
                triggered_by=username,
            ))

            responding_personality = self.personality_manager.find_responding_personality(message)

            if responding_personality and responding_personality.name != self.personality_manager._active_personality:
                await self.personality_manager.switch_to(responding_personality.name)

            self.context.add_user_message(message, name=username)

            async for chunk in self._generate_response():
                yield (self.personality_manager._active_personality or "unknown", chunk)

            if self._should_interjection():
                async for chunk in self._generate_interjection():
                    yield chunk

        finally:
            self._is_speaking = False

    async def _generate_response(self) -> AsyncIterator[str]:
        """Generate response from active personality."""
        personality = self.personality_manager.active_personality
        if not personality:
            return

        prompt = self.context.format_for_llama()

        full_response = []
        async for token in self.llm.generate_stream(prompt):
            full_response.append(token)
            yield token

        response_text = "".join(full_response)

        self.context.add_assistant_message(
            response_text,
            personality=personality.name,
        )

        self._turn_history.append(DialogueTurn(
            turn_type=TurnType.AI_RESPONSE,
            personality=personality.name,
            content=response_text,
        ))

        self.personality_manager.increment_message_count()
        self.personality_manager.set_last_response(response_text)

    def _should_interjection(self) -> bool:
        """Determine if secondary AI should interject."""
        if len(self.personality_manager.personalities) < 2:
            return False

        return random.random() < self.interjection_probability

    async def _generate_interjection(self) -> AsyncIterator[tuple[str, str]]:
        """Generate an interjection from the secondary personality."""
        current = self.personality_manager._active_personality
        other_personalities = [
            name for name in self.personality_manager.personalities.keys()
            if name != current
        ]

        if not other_personalities:
            return

        interjecting = random.choice(other_personalities)

        await self.personality_manager.switch_to(interjecting, force=True)

        interjection_prompt = self._create_interjection_prompt()
        self.context.system_prompt = self.personality_manager.active_personality.system_prompt

        prompt = self.context.format_for_llama()

        full_response = []
        async for token in self.llm.generate_stream(
            prompt,
            max_tokens=150,
        ):
            full_response.append(token)
            yield (interjecting, token)

        response_text = "".join(full_response)

        self.context.add_assistant_message(
            response_text,
            personality=interjecting,
        )

        self._turn_history.append(DialogueTurn(
            turn_type=TurnType.AI_INTERJECTION,
            personality=interjecting,
            content=response_text,
        ))

        await self.personality_manager.switch_to(current, force=True)

    def _create_interjection_prompt(self) -> str:
        """Create a prompt for interjection."""
        last_turn = self._turn_history[-1] if self._turn_history else None

        if last_turn and last_turn.turn_type == TurnType.AI_RESPONSE:
            return f"[{self.personality_manager.active_personality.display_name} just said: \"{last_turn.content[:100]}...\"] React briefly with your personality - be playful, sarcastic, or add chaos!"

        return "Interject briefly with something in character!"

    async def trigger_takeover(self, personality_name: str) -> AsyncIterator[tuple[str, str]]:
        """
        Trigger a personality takeover event.

        Args:
            personality_name: Personality to take over

        Yields:
            Tuples of (personality_name, response_chunk)
        """
        if personality_name not in self.personality_manager.personalities:
            return

        await self.personality_manager.switch_to(personality_name, force=True)

        takeover_message = self._create_takeover_message()

        self._turn_history.append(DialogueTurn(
            turn_type=TurnType.AI_TAKEOVER,
            personality=personality_name,
            content=takeover_message,
        ))

        for word in takeover_message.split():
            yield (personality_name, word + " ")
            await asyncio.sleep(0.05)

    def _create_takeover_message(self) -> str:
        """Create a takeover announcement message."""
        personality = self.personality_manager.active_personality
        if not personality:
            return "Taking over!"

        messages = [
            f"*{personality.display_name} has taken control of the stream*",
            f"It's {personality.display_name} time now!",
            f"*{personality.display_name} pushes the other aside*",
            f"Finally, my turn to shine!",
        ]

        return random.choice(messages)

    def get_turn_history(self, limit: int = 10) -> list[DialogueTurn]:
        """Get recent turn history."""
        return self._turn_history[-limit:]

    def clear_history(self) -> None:
        """Clear turn history."""
        self._turn_history.clear()
