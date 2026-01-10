"""
Watch mode for Spark VTuber.

Enables AI to watch and comment on gameplay without direct control.
"""

import asyncio
import io
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import numpy as np

from spark_vtuber.utils.logging import LoggerMixin


@dataclass
class WatchFrame:
    """A single frame from watch mode."""

    image: bytes
    timestamp: float
    audio: np.ndarray | None = None
    audio_sample_rate: int = 16000


@dataclass
class WatchEvent:
    """An event detected during watch mode."""

    event_type: str
    description: str
    timestamp: float
    confidence: float = 1.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WatchMode(LoggerMixin):
    """
    Watch mode for observing gameplay.

    Captures screen and audio to enable AI commentary
    without direct game control.
    """

    def __init__(
        self,
        capture_fps: int = 2,
        capture_audio: bool = True,
        screen_region: tuple[int, int, int, int] | None = None,
    ):
        """
        Initialize watch mode.

        Args:
            capture_fps: Frames per second to capture
            capture_audio: Whether to capture audio
            screen_region: Region to capture (x, y, width, height)
        """
        self.capture_fps = capture_fps
        self.capture_audio = capture_audio
        self.screen_region = screen_region
        self._running = False
        self._frame_queue: asyncio.Queue[WatchFrame] = asyncio.Queue(maxsize=10)
        self._event_callbacks: list[Callable[[WatchEvent], None]] = []

    async def start(self) -> None:
        """Start watch mode capture."""
        if self._running:
            self.logger.warning("Watch mode already running")
            return

        self.logger.info("Starting watch mode")
        self._running = True

        asyncio.create_task(self._capture_loop())

        if self.capture_audio:
            asyncio.create_task(self._audio_capture_loop())

    async def stop(self) -> None:
        """Stop watch mode capture."""
        if not self._running:
            return

        self.logger.info("Stopping watch mode")
        self._running = False

        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _capture_loop(self) -> None:
        """Main capture loop for screen frames."""
        frame_interval = 1.0 / self.capture_fps

        while self._running:
            start_time = time.time()

            try:
                frame = await self._capture_frame()
                if frame:
                    try:
                        self._frame_queue.put_nowait(frame)
                    except asyncio.QueueFull:
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put_nowait(frame)
                        except asyncio.QueueEmpty:
                            pass

            except Exception as e:
                self.logger.error(f"Capture error: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _capture_frame(self) -> WatchFrame | None:
        """Capture a single frame."""
        try:
            import pyautogui
            from PIL import Image

            if self.screen_region:
                screenshot = pyautogui.screenshot(region=self.screen_region)
            else:
                screenshot = pyautogui.screenshot()

            buffer = io.BytesIO()
            screenshot.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            return WatchFrame(
                image=image_bytes,
                timestamp=time.time(),
            )

        except ImportError:
            self.logger.warning("pyautogui not available for screen capture")
            return None
        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}")
            return None

    async def _audio_capture_loop(self) -> None:
        """Capture audio in background."""
        pass

    async def get_frames(self) -> AsyncIterator[WatchFrame]:
        """
        Get captured frames.

        Yields:
            WatchFrame objects as they become available
        """
        while self._running:
            try:
                frame = await asyncio.wait_for(
                    self._frame_queue.get(),
                    timeout=1.0,
                )
                yield frame
            except asyncio.TimeoutError:
                continue

    def on_event(self, callback: Callable[[WatchEvent], None]) -> None:
        """
        Register callback for watch events.

        Args:
            callback: Function to call with WatchEvent
        """
        self._event_callbacks.append(callback)

    async def _dispatch_event(self, event: WatchEvent) -> None:
        """Dispatch event to callbacks."""
        for callback in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")

    @property
    def is_running(self) -> bool:
        """Check if watch mode is running."""
        return self._running


class ScreenAnalyzer(LoggerMixin):
    """
    Analyzes screen captures for game events.

    Uses vision models to understand what's happening
    in the game for commentary generation.
    """

    def __init__(self, model_name: str = "yolo-world"):
        """
        Initialize screen analyzer.

        Args:
            model_name: Vision model to use
        """
        self.model_name = model_name
        self._model = None
        self._loaded = False

    async def load(self) -> None:
        """Load the vision model."""
        self.logger.info(f"Loading vision model: {self.model_name}")
        self._loaded = True

    async def unload(self) -> None:
        """Unload the vision model."""
        self._model = None
        self._loaded = False

    async def analyze(self, frame: WatchFrame) -> list[WatchEvent]:
        """
        Analyze a frame for events.

        Args:
            frame: Frame to analyze

        Returns:
            List of detected events
        """
        if not self._loaded:
            return []

        events = []

        events.append(WatchEvent(
            event_type="frame_captured",
            description="New frame captured",
            timestamp=frame.timestamp,
        ))

        return events

    async def describe_scene(self, frame: WatchFrame) -> str:
        """
        Generate a description of the current scene.

        Args:
            frame: Frame to describe

        Returns:
            Scene description string
        """
        if not self._loaded:
            return "Scene analysis not available"

        return "Game scene captured - ready for analysis"

    async def detect_changes(
        self,
        frame1: WatchFrame,
        frame2: WatchFrame,
    ) -> list[str]:
        """
        Detect changes between two frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            List of detected changes
        """
        return ["Frame updated"]
