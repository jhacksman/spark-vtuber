"""
Dual VTube Studio avatar controller for Spark VTuber.

Manages two simultaneous VTube Studio connections for VNet Multiplayer Collab.
Enables both AI personalities (Spark and Shadow) to be visible on screen simultaneously.
"""

import asyncio
from typing import Literal

from spark_vtuber.avatar.base import BaseAvatar, Emotion, ExpressionState, LipSyncFrame
from spark_vtuber.avatar.vtube_studio import VTubeStudioAvatar


class DualVTubeStudioAvatar(BaseAvatar):
    """
    Dual avatar controller for VNet Multiplayer Collab.

    Manages two VTube Studio instances:
    - Primary avatar (Spark) on configurable port (default 8001)
    - Secondary avatar (Shadow) on configurable port (default 8002)

    Both avatars are controlled independently via VTube Studio Plugin API.
    VNet handles spatial synchronization and rendering.

    Requirements:
    - VTube Studio Pro ($14.99)
    - VNet Multiplayer Collab ($20.00)
    - Two Live2D models (one per personality)
    """

    def __init__(
        self,
        primary_host: str = "localhost",
        primary_port: int = 8001,
        secondary_host: str = "localhost",
        secondary_port: int = 8002,
        primary_plugin_name: str = "SparkVTuber_Primary",
        secondary_plugin_name: str = "SparkVTuber_Secondary",
        **kwargs,
    ):
        """
        Initialize dual VTube Studio controller.

        Args:
            primary_host: VTube Studio host for primary avatar
            primary_port: VTube Studio port for primary avatar (Spark)
            secondary_host: VTube Studio host for secondary avatar
            secondary_port: VTube Studio port for secondary avatar (Shadow)
            primary_plugin_name: Plugin name for primary avatar
            secondary_plugin_name: Plugin name for secondary avatar
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)

        self.primary = VTubeStudioAvatar(
            host=primary_host,
            port=primary_port,
            plugin_name=primary_plugin_name,
        )

        self.secondary = VTubeStudioAvatar(
            host=secondary_host,
            port=secondary_port,
            plugin_name=secondary_plugin_name,
        )

        self._active_speaker: Literal["primary", "secondary"] = "primary"
        self._primary_expression = ExpressionState()
        self._secondary_expression = ExpressionState()

    @property
    def active_speaker(self) -> Literal["primary", "secondary"]:
        """Get the currently active speaker."""
        return self._active_speaker

    @property
    def primary_expression(self) -> ExpressionState:
        """Get primary avatar's current expression state."""
        return self._primary_expression

    @property
    def secondary_expression(self) -> ExpressionState:
        """Get secondary avatar's current expression state."""
        return self._secondary_expression

    async def connect(self) -> None:
        """Connect both avatars to their respective VTube Studio instances."""
        if self._connected:
            self.logger.warning("Already connected to VTube Studio instances")
            return

        self.logger.info("Connecting to dual VTube Studio instances (VNet mode)")

        errors = []

        try:
            await self.primary.connect()
            self.logger.info("Primary avatar connected")
        except Exception as e:
            self.logger.error(f"Failed to connect primary avatar: {e}")
            errors.append(("primary", e))

        try:
            await self.secondary.connect()
            self.logger.info("Secondary avatar connected")
        except Exception as e:
            self.logger.error(f"Failed to connect secondary avatar: {e}")
            errors.append(("secondary", e))

        if len(errors) == 2:
            raise ConnectionError(
                f"Failed to connect both avatars: "
                f"primary={errors[0][1]}, secondary={errors[1][1]}"
            )

        if errors:
            self.logger.warning(
                f"Partial connection: {errors[0][0]} avatar failed to connect. "
                "Continuing with single avatar mode."
            )

        self._connected = True
        self.logger.info("Dual avatar connection complete")

    async def disconnect(self) -> None:
        """Disconnect both avatars from VTube Studio."""
        if not self._connected:
            return

        self.logger.info("Disconnecting from VTube Studio instances")

        try:
            await self.primary.disconnect()
        except Exception as e:
            self.logger.error(f"Error disconnecting primary avatar: {e}")

        try:
            await self.secondary.disconnect()
        except Exception as e:
            self.logger.error(f"Error disconnecting secondary avatar: {e}")

        self._connected = False
        self.logger.info("Dual avatar disconnection complete")

    async def set_active_speaker(
        self,
        speaker: Literal["primary", "secondary"],
    ) -> None:
        """
        Set which personality is actively speaking.

        The active speaker receives lip sync updates. The inactive speaker
        remains visible but with a neutral/idle mouth.

        Args:
            speaker: "primary" for Spark, "secondary" for Shadow
        """
        if speaker not in ("primary", "secondary"):
            raise ValueError(f"Invalid speaker: {speaker}. Must be 'primary' or 'secondary'")

        self._active_speaker = speaker
        self.logger.debug(f"Active speaker set to: {speaker}")

    async def set_parameter(self, name: str, value: float) -> None:
        """Set a parameter on the active speaker's avatar."""
        if self._active_speaker == "primary":
            await self.primary.set_parameter(name, value)
        else:
            await self.secondary.set_parameter(name, value)

    async def set_parameters(self, parameters: dict[str, float]) -> None:
        """Set multiple parameters on the active speaker's avatar."""
        if self._active_speaker == "primary":
            await self.primary.set_parameters(parameters)
        else:
            await self.secondary.set_parameters(parameters)

    async def get_parameter(self, name: str) -> float | None:
        """Get a parameter from the active speaker's avatar."""
        if self._active_speaker == "primary":
            return await self.primary.get_parameter(name)
        else:
            return await self.secondary.get_parameter(name)

    async def get_available_parameters(self) -> list[str]:
        """Get available parameters from the active speaker's avatar."""
        if self._active_speaker == "primary":
            return await self.primary.get_available_parameters()
        else:
            return await self.secondary.get_available_parameters()

    async def set_expression(self, emotion: Emotion, intensity: float = 1.0) -> None:
        """Set expression for the active speaker's avatar."""
        if self._active_speaker == "primary":
            await self.set_primary_expression(emotion, intensity)
        else:
            await self.set_secondary_expression(emotion, intensity)

    async def update_lip_sync(self, frame: LipSyncFrame) -> None:
        """
        Update lip sync for the active speaker only.

        The inactive speaker's mouth remains idle/neutral.
        """
        if not self._connected:
            return

        if self._active_speaker == "primary":
            if self.primary.is_connected:
                await self.primary.update_lip_sync(frame)
        else:
            if self.secondary.is_connected:
                await self.secondary.update_lip_sync(frame)

    async def trigger_animation(self, animation_name: str) -> None:
        """Trigger an animation on the active speaker's avatar."""
        if self._active_speaker == "primary":
            await self.primary.trigger_animation(animation_name)
        else:
            await self.secondary.trigger_animation(animation_name)

    async def get_available_animations(self) -> list[str]:
        """Get available animations from the active speaker's avatar."""
        if self._active_speaker == "primary":
            return await self.primary.get_available_animations()
        else:
            return await self.secondary.get_available_animations()

    async def set_primary_expression(
        self,
        emotion: Emotion,
        intensity: float = 1.0,
    ) -> None:
        """
        Set expression for the primary avatar (Spark).

        Args:
            emotion: Target emotion
            intensity: Expression intensity (0-1)
        """
        if self.primary.is_connected:
            await self.primary.set_expression(emotion, intensity)
            self._primary_expression = ExpressionState(
                emotion=emotion,
                intensity=intensity,
            )

    async def set_secondary_expression(
        self,
        emotion: Emotion,
        intensity: float = 1.0,
    ) -> None:
        """
        Set expression for the secondary avatar (Shadow).

        Args:
            emotion: Target emotion
            intensity: Expression intensity (0-1)
        """
        if self.secondary.is_connected:
            await self.secondary.set_expression(emotion, intensity)
            self._secondary_expression = ExpressionState(
                emotion=emotion,
                intensity=intensity,
            )

    async def set_expression_both(
        self,
        primary_emotion: Emotion,
        secondary_emotion: Emotion,
        intensity: float = 1.0,
    ) -> None:
        """
        Set expressions for both avatars simultaneously.

        Useful for coordinated reactions or synchronized emotions.

        Args:
            primary_emotion: Emotion for primary avatar (Spark)
            secondary_emotion: Emotion for secondary avatar (Shadow)
            intensity: Expression intensity for both (0-1)
        """
        await asyncio.gather(
            self.set_primary_expression(primary_emotion, intensity),
            self.set_secondary_expression(secondary_emotion, intensity),
        )

    async def set_primary_parameter(self, name: str, value: float) -> None:
        """Set a parameter on the primary avatar."""
        if self.primary.is_connected:
            await self.primary.set_parameter(name, value)

    async def set_secondary_parameter(self, name: str, value: float) -> None:
        """Set a parameter on the secondary avatar."""
        if self.secondary.is_connected:
            await self.secondary.set_parameter(name, value)

    async def update_primary_lip_sync(self, frame: LipSyncFrame) -> None:
        """Update lip sync for primary avatar only."""
        if self.primary.is_connected:
            await self.primary.update_lip_sync(frame)

    async def update_secondary_lip_sync(self, frame: LipSyncFrame) -> None:
        """Update lip sync for secondary avatar only."""
        if self.secondary.is_connected:
            await self.secondary.update_lip_sync(frame)

    async def trigger_primary_animation(self, animation_name: str) -> None:
        """Trigger an animation on the primary avatar."""
        if self.primary.is_connected:
            await self.primary.trigger_animation(animation_name)

    async def trigger_secondary_animation(self, animation_name: str) -> None:
        """Trigger an animation on the secondary avatar."""
        if self.secondary.is_connected:
            await self.secondary.trigger_animation(animation_name)

    def get_connection_status(self) -> dict[str, bool]:
        """
        Get connection status for both avatars.

        Returns:
            Dictionary with 'primary' and 'secondary' connection status
        """
        return {
            "primary": self.primary.is_connected,
            "secondary": self.secondary.is_connected,
        }
