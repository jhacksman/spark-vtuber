"""
Main streaming pipeline for Spark VTuber.

Orchestrates all components for real-time AI VTuber streaming.
"""

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import numpy as np

from spark_vtuber.avatar.base import BaseAvatar, Emotion, LipSyncFrame
from spark_vtuber.avatar.lip_sync import LipSyncProcessor
from spark_vtuber.chat.base import BaseChat, ChatMessage
from spark_vtuber.chat.queue import MessageQueue
from spark_vtuber.config.settings import Settings
from spark_vtuber.game.base import BaseGame
from spark_vtuber.llm.base import BaseLLM
from spark_vtuber.llm.context import ConversationContext
from spark_vtuber.memory.base import BaseMemory
from spark_vtuber.personality.coordinator import DialogueCoordinator
from spark_vtuber.personality.manager import PersonalityManager
from spark_vtuber.tts.base import BaseTTS
from spark_vtuber.tts.streaming import StreamingTTS
from spark_vtuber.utils.logging import LoggerMixin


@dataclass
class PipelineStats:
    """Statistics for pipeline performance."""

    messages_processed: int = 0
    total_response_time_ms: float = 0.0
    average_response_time_ms: float = 0.0
    tts_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    errors: int = 0


class StreamingPipeline(LoggerMixin):
    """
    Main streaming pipeline that orchestrates all components.

    Flow:
    1. Receive chat message
    2. Retrieve relevant memories
    3. Generate LLM response (streaming)
    4. Synthesize speech (streaming)
    5. Update avatar lip sync
    6. Store new memories
    """

    def __init__(
        self,
        llm: BaseLLM,
        tts: BaseTTS,
        memory: BaseMemory,
        avatar: BaseAvatar | None = None,
        chat: BaseChat | None = None,
        game: BaseGame | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize the streaming pipeline.

        Args:
            llm: LLM instance
            tts: TTS instance
            memory: Memory instance
            avatar: Optional avatar controller
            chat: Optional chat client
            game: Optional game integration
            settings: Application settings
        """
        self.llm = llm
        self.tts = tts
        self.memory = memory
        self.avatar = avatar
        self.chat = chat
        self.game = game
        self.settings = settings or Settings()

        self.context = ConversationContext(
            max_tokens=self.settings.llm.context_length,
        )

        self.personality_manager = PersonalityManager(
            llm=llm,
            switch_cooldown=self.settings.personality.switch_cooldown,
        )

        self.coordinator = DialogueCoordinator(
            llm=llm,
            personality_manager=self.personality_manager,
            context=self.context,
        )

        self.message_queue = MessageQueue(
            max_size=self.settings.chat.message_queue_size,
            rate_limit_per_minute=self.settings.chat.rate_limit_per_minute,
        )

        self.streaming_tts = StreamingTTS(tts)
        self.lip_sync = LipSyncProcessor()

        self._running = False
        self._stats = PipelineStats()
        self._audio_callbacks: list[Callable[[np.ndarray], None]] = []

    @property
    def stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        self.logger.info("Initializing streaming pipeline")

        await self.llm.load()
        await self.tts.load()
        await self.memory.initialize()

        if self.avatar:
            await self.avatar.connect()

        if self.chat:
            await self.chat.connect()

        if self.game:
            await self.game.connect()

        self.personality_manager.register_default_personalities()
        await self.personality_manager.initialize()

        self.context.system_prompt = await self.personality_manager.get_system_prompt()

        self.logger.info("Pipeline initialized")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        self.logger.info("Shutting down pipeline")

        self._running = False

        if self.game:
            await self.game.disconnect()

        if self.chat:
            await self.chat.disconnect()

        if self.avatar:
            await self.avatar.disconnect()

        await self.memory.close()
        await self.tts.unload()
        await self.llm.unload()

        self.logger.info("Pipeline shutdown complete")

    async def start(self) -> None:
        """Start the streaming pipeline."""
        if self._running:
            self.logger.warning("Pipeline already running")
            return

        self._running = True
        self.logger.info("Starting streaming pipeline")

        tasks = [
            asyncio.create_task(self._message_processing_loop()),
        ]

        if self.chat:
            tasks.append(asyncio.create_task(self._chat_reading_loop()))

        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        """Stop the streaming pipeline."""
        self._running = False
        self.logger.info("Stopping streaming pipeline")

    async def _chat_reading_loop(self) -> None:
        """Read messages from chat and add to queue."""
        if not self.chat:
            return

        async for message in self.chat.get_messages():
            if not self._running:
                break

            await self.message_queue.add(message)

    async def _message_processing_loop(self) -> None:
        """Process messages from the queue."""
        while self._running:
            message = await self.message_queue.get(timeout=1.0)
            if message:
                try:
                    await self.process_message(message)
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    self._stats.errors += 1

    async def process_message(self, message: ChatMessage) -> None:
        """
        Process a single chat message through the full pipeline.

        Args:
            message: Chat message to process
        """
        import time
        start_time = time.time()

        self.logger.info(f"Processing message from {message.username}: {message.content[:50]}...")

        relevant_memories = await self.memory.search(
            message.content,
            top_k=self.settings.memory.retrieval_top_k,
        )

        if relevant_memories:
            memory_context = "\n".join([
                f"- {m.entry.content}" for m in relevant_memories[:3]
            ])
            self.context.add_message(
                "system",
                f"Relevant memories:\n{memory_context}",
            )

        async for personality_name, chunk in self.coordinator.process_user_message(
            message.content,
            username=message.username,
        ):
            async for audio_chunk in self.streaming_tts.process_token(chunk):
                await self._process_audio(audio_chunk)

        async for audio_chunk in self.streaming_tts.flush():
            await self._process_audio(audio_chunk)

        await self.memory.add(
            content=f"{message.username}: {message.content}",
            category="chat",
            personality=self.personality_manager._active_personality,
        )

        response_time = (time.time() - start_time) * 1000
        self._stats.messages_processed += 1
        self._stats.total_response_time_ms += response_time
        self._stats.average_response_time_ms = (
            self._stats.total_response_time_ms / self._stats.messages_processed
        )

        self.logger.info(f"Message processed in {response_time:.0f}ms")

    async def _process_audio(self, audio: np.ndarray) -> None:
        """Process audio chunk for playback and lip sync."""
        for callback in self._audio_callbacks:
            try:
                callback(audio)
            except Exception as e:
                self.logger.error(f"Audio callback error: {e}")

        if self.avatar:
            frames = self.lip_sync.process_audio_chunk(audio)
            for frame in frames:
                await self.avatar.update_lip_sync(frame)

    def on_audio(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Register callback for audio output.

        Args:
            callback: Function to call with audio chunks
        """
        self._audio_callbacks.append(callback)

    async def send_response(self, text: str) -> None:
        """
        Send a direct response (bypassing chat input).

        Args:
            text: Response text to speak
        """
        async for audio_chunk in self.tts.synthesize_stream(text):
            await self._process_audio(audio_chunk)

    async def switch_personality(self, name: str) -> bool:
        """
        Switch to a different personality.

        Args:
            name: Personality name

        Returns:
            True if switch was successful
        """
        success = await self.personality_manager.switch_to(name)
        if success:
            self.context.system_prompt = await self.personality_manager.get_system_prompt()
        return success

    async def set_emotion(self, emotion: Emotion) -> None:
        """
        Set avatar emotion.

        Args:
            emotion: Emotion to display
        """
        if self.avatar:
            await self.avatar.set_expression(emotion)

    def get_memory_usage(self) -> dict[str, float]:
        """Get memory usage of all components."""
        usage = {}

        llm_usage = self.llm.get_memory_usage()
        usage.update({f"llm_{k}": v for k, v in llm_usage.items()})

        tts_usage = self.tts.get_memory_usage()
        usage.update({f"tts_{k}": v for k, v in tts_usage.items()})

        return usage


class PipelineBuilder(LoggerMixin):
    """Builder for constructing streaming pipelines."""

    def __init__(self, settings: Settings | None = None):
        """
        Initialize pipeline builder.

        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self._llm = None
        self._tts = None
        self._memory = None
        self._avatar = None
        self._chat = None
        self._game = None

    def with_llm(self, llm: BaseLLM) -> "PipelineBuilder":
        """Set LLM instance."""
        self._llm = llm
        return self

    def with_tts(self, tts: BaseTTS) -> "PipelineBuilder":
        """Set TTS instance."""
        self._tts = tts
        return self

    def with_memory(self, memory: BaseMemory) -> "PipelineBuilder":
        """Set memory instance."""
        self._memory = memory
        return self

    def with_avatar(self, avatar: BaseAvatar) -> "PipelineBuilder":
        """Set avatar instance."""
        self._avatar = avatar
        return self

    def with_chat(self, chat: BaseChat) -> "PipelineBuilder":
        """Set chat instance."""
        self._chat = chat
        return self

    def with_game(self, game: BaseGame) -> "PipelineBuilder":
        """Set game instance."""
        self._game = game
        return self

    def build(self) -> StreamingPipeline:
        """Build the pipeline."""
        if not self._llm:
            raise ValueError("LLM is required")
        if not self._tts:
            raise ValueError("TTS is required")
        if not self._memory:
            raise ValueError("Memory is required")

        return StreamingPipeline(
            llm=self._llm,
            tts=self._tts,
            memory=self._memory,
            avatar=self._avatar,
            chat=self._chat,
            game=self._game,
            settings=self.settings,
        )
