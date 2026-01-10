"""
Personality manager for Spark VTuber.

Handles personality switching and LoRA adapter management.
"""

import asyncio
import time
from typing import Callable

from spark_vtuber.llm.base import BaseLLM
from spark_vtuber.personality.base import (
    Personality,
    PersonalityConfig,
    DEFAULT_PRIMARY_PERSONALITY,
    DEFAULT_SECONDARY_PERSONALITY,
)
from spark_vtuber.utils.logging import LoggerMixin


class PersonalityManager(LoggerMixin):
    """
    Manages multiple AI personalities with LoRA switching.

    Handles:
    - Personality registration and configuration
    - LoRA adapter loading and switching
    - Active personality tracking
    - Switch cooldown management
    """

    def __init__(
        self,
        llm: BaseLLM,
        switch_cooldown: float = 5.0,
    ):
        """
        Initialize personality manager.

        Args:
            llm: LLM instance for LoRA management
            switch_cooldown: Minimum seconds between switches
        """
        self.llm = llm
        self.switch_cooldown = switch_cooldown
        self._personalities: dict[str, Personality] = {}
        self._active_personality: str | None = None
        self._last_switch_time: float = 0.0
        self._switch_callbacks: list[Callable] = []

    @property
    def active_personality(self) -> Personality | None:
        """Get the currently active personality."""
        if self._active_personality:
            return self._personalities.get(self._active_personality)
        return None

    @property
    def personalities(self) -> dict[str, Personality]:
        """Get all registered personalities."""
        return self._personalities.copy()

    def register_personality(self, config: PersonalityConfig) -> Personality:
        """
        Register a new personality.

        Args:
            config: Personality configuration

        Returns:
            Created Personality instance
        """
        personality = Personality(config=config)
        self._personalities[config.name] = personality
        self.logger.info(f"Registered personality: {config.name}")
        return personality

    def register_default_personalities(self) -> None:
        """Register the default primary and secondary personalities."""
        self.register_personality(DEFAULT_PRIMARY_PERSONALITY)
        self.register_personality(DEFAULT_SECONDARY_PERSONALITY)
        self.logger.info("Registered default personalities")

    async def initialize(self) -> None:
        """Initialize personalities and load LoRA adapters."""
        for name, personality in self._personalities.items():
            if personality.config.lora_path:
                await self.llm.load_lora_adapter(
                    personality.config.lora_path,
                    name,
                )
                self.logger.info(f"Loaded LoRA adapter for {name}")

        if self._personalities:
            first_name = next(iter(self._personalities.keys()))
            await self.switch_to(first_name)

    async def switch_to(self, personality_name: str, force: bool = False) -> bool:
        """
        Switch to a different personality.

        Args:
            personality_name: Name of personality to switch to
            force: Bypass cooldown check

        Returns:
            True if switch was successful
        """
        if personality_name not in self._personalities:
            self.logger.error(f"Unknown personality: {personality_name}")
            return False

        if not force:
            time_since_switch = time.time() - self._last_switch_time
            if time_since_switch < self.switch_cooldown:
                self.logger.warning(
                    f"Switch cooldown active ({self.switch_cooldown - time_since_switch:.1f}s remaining)"
                )
                return False

        if self._active_personality:
            self._personalities[self._active_personality].is_active = False

        personality = self._personalities[personality_name]

        if personality.config.lora_path:
            await self.llm.set_active_adapter(personality_name)
        else:
            await self.llm.set_active_adapter(None)

        personality.is_active = True
        self._active_personality = personality_name
        self._last_switch_time = time.time()

        self.logger.info(f"Switched to personality: {personality_name}")

        for callback in self._switch_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(personality)
                else:
                    callback(personality)
            except Exception as e:
                self.logger.error(f"Switch callback error: {e}")

        return True

    def on_switch(self, callback: Callable) -> None:
        """
        Register a callback for personality switches.

        Args:
            callback: Function to call on switch (receives Personality)
        """
        self._switch_callbacks.append(callback)

    def get_personality(self, name: str) -> Personality | None:
        """Get a personality by name."""
        return self._personalities.get(name)

    def find_responding_personality(self, message: str) -> Personality | None:
        """
        Find which personality should respond to a message.

        Args:
            message: Input message

        Returns:
            Personality that should respond, or None
        """
        for personality in self._personalities.values():
            if personality.should_respond_to(message):
                return personality

        return self.active_personality

    async def get_system_prompt(self) -> str:
        """Get the system prompt for the active personality."""
        if self.active_personality:
            return self.active_personality.system_prompt
        return ""

    def increment_message_count(self) -> None:
        """Increment message count for active personality."""
        if self.active_personality:
            self.active_personality.message_count += 1

    def set_last_response(self, response: str) -> None:
        """Set last response for active personality."""
        if self.active_personality:
            self.active_personality.last_response = response
