"""
VTube Studio avatar controller for Spark VTuber.

Implements the VTube Studio API for Live2D avatar control.
"""

import asyncio
import json
import uuid
from typing import Any

import websockets

from spark_vtuber.avatar.base import BaseAvatar, Emotion, ExpressionState, LipSyncFrame


class VTubeStudioAvatar(BaseAvatar):
    """
    VTube Studio avatar controller.

    Implements the VTube Studio Plugin API for:
    - Parameter control
    - Expression/hotkey triggering
    - Lip sync via mouth parameters
    - Custom parameter injection
    """

    EMOTION_HOTKEYS = {
        Emotion.NEUTRAL: "neutral",
        Emotion.HAPPY: "happy",
        Emotion.SAD: "sad",
        Emotion.ANGRY: "angry",
        Emotion.SURPRISED: "surprised",
        Emotion.THINKING: "thinking",
        Emotion.EXCITED: "excited",
        Emotion.CONFUSED: "confused",
    }

    PHONEME_PARAMS = {
        "A": {"MouthOpen": 1.0, "MouthForm": 0.0},
        "E": {"MouthOpen": 0.7, "MouthForm": 0.3},
        "I": {"MouthOpen": 0.5, "MouthForm": 0.5},
        "O": {"MouthOpen": 0.9, "MouthForm": -0.3},
        "U": {"MouthOpen": 0.6, "MouthForm": -0.5},
        "M": {"MouthOpen": 0.1, "MouthForm": 0.0},
        "S": {"MouthOpen": 0.3, "MouthForm": 0.4},
        "F": {"MouthOpen": 0.2, "MouthForm": 0.3},
        "TH": {"MouthOpen": 0.4, "MouthForm": 0.2},
        "SIL": {"MouthOpen": 0.0, "MouthForm": 0.0},
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8001,
        plugin_name: str = "SparkVTuber",
        plugin_developer: str = "SparkVTuber",
        **kwargs,
    ):
        """
        Initialize VTube Studio controller.

        Args:
            host: VTube Studio WebSocket host
            port: VTube Studio WebSocket port
            plugin_name: Plugin name for authentication
            plugin_developer: Plugin developer name
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self._ws = None
        self._auth_token: str | None = None
        self._available_parameters: list[str] = []
        self._available_hotkeys: list[str] = []
        self._request_id = 0

    def _get_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_id += 1
        return f"req_{self._request_id}"

    async def connect(self) -> None:
        """Connect to VTube Studio and authenticate."""
        if self._connected:
            self.logger.warning("Already connected to VTube Studio")
            return

        self.logger.info(f"Connecting to VTube Studio at {self.host}:{self.port}")

        try:
            uri = f"ws://{self.host}:{self.port}"
            self._ws = await websockets.connect(uri)

            await self._authenticate()

            await self._fetch_available_parameters()
            await self._fetch_available_hotkeys()

            self._connected = True
            self.logger.info("Connected to VTube Studio")

        except Exception as e:
            self.logger.error(f"Failed to connect to VTube Studio: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from VTube Studio."""
        if not self._connected:
            return

        self.logger.info("Disconnecting from VTube Studio")

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._connected = False

    async def _send_request(self, request_type: str, data: dict | None = None) -> dict:
        """Send a request to VTube Studio and wait for response."""
        if not self._ws:
            raise RuntimeError("Not connected to VTube Studio")

        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": self._get_request_id(),
            "messageType": request_type,
            "data": data or {},
        }

        await self._ws.send(json.dumps(request))
        response = await self._ws.recv()
        return json.loads(response)

    async def _authenticate(self) -> None:
        """Authenticate with VTube Studio."""
        response = await self._send_request(
            "AuthenticationTokenRequest",
            {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
            },
        )

        if "data" in response and "authenticationToken" in response["data"]:
            self._auth_token = response["data"]["authenticationToken"]

            auth_response = await self._send_request(
                "AuthenticationRequest",
                {
                    "pluginName": self.plugin_name,
                    "pluginDeveloper": self.plugin_developer,
                    "authenticationToken": self._auth_token,
                },
            )

            if auth_response.get("data", {}).get("authenticated"):
                self.logger.info("Authenticated with VTube Studio")
            else:
                self.logger.warning("Authentication pending - please accept in VTube Studio")

    async def _fetch_available_parameters(self) -> None:
        """Fetch list of available parameters."""
        response = await self._send_request("InputParameterListRequest")

        if "data" in response:
            default_params = response["data"].get("defaultParameters", [])
            custom_params = response["data"].get("customParameters", [])

            self._available_parameters = [
                p["name"] for p in default_params + custom_params
            ]

    async def _fetch_available_hotkeys(self) -> None:
        """Fetch list of available hotkeys."""
        response = await self._send_request("HotkeysInCurrentModelRequest")

        if "data" in response:
            hotkeys = response["data"].get("availableHotkeys", [])
            self._available_hotkeys = [h["name"] for h in hotkeys]

    async def set_parameter(self, name: str, value: float) -> None:
        """Set a single avatar parameter."""
        await self.set_parameters({name: value})

    async def set_parameters(self, parameters: dict[str, float]) -> None:
        """Set multiple avatar parameters at once."""
        if not self._connected:
            raise RuntimeError("Not connected to VTube Studio")

        param_values = [
            {"id": name, "value": value}
            for name, value in parameters.items()
        ]

        await self._send_request(
            "InjectParameterDataRequest",
            {
                "parameterValues": param_values,
                "mode": "set",
            },
        )

    async def get_parameter(self, name: str) -> float | None:
        """Get current value of a parameter."""
        if not self._connected:
            raise RuntimeError("Not connected to VTube Studio")

        response = await self._send_request(
            "ParameterValueRequest",
            {"name": name},
        )

        if "data" in response:
            return response["data"].get("value")
        return None

    async def get_available_parameters(self) -> list[str]:
        """Get list of available parameter names."""
        return self._available_parameters.copy()

    async def set_expression(self, emotion: Emotion, intensity: float = 1.0) -> None:
        """Set the avatar's expression via hotkey."""
        if not self._connected:
            raise RuntimeError("Not connected to VTube Studio")

        hotkey_name = self.EMOTION_HOTKEYS.get(emotion)
        if hotkey_name and hotkey_name in self._available_hotkeys:
            await self._send_request(
                "HotkeyTriggerRequest",
                {"hotkeyID": hotkey_name},
            )

        self._current_expression = ExpressionState(
            emotion=emotion,
            intensity=intensity,
        )

    async def update_lip_sync(self, frame: LipSyncFrame) -> None:
        """Update lip sync with a single frame."""
        if not self._connected:
            return

        phoneme = frame.phoneme.upper()
        if phoneme not in self.PHONEME_PARAMS:
            phoneme = "SIL"

        params = self.PHONEME_PARAMS[phoneme]
        scaled_params = {
            name: value * frame.intensity
            for name, value in params.items()
        }

        await self.set_parameters(scaled_params)

    async def trigger_animation(self, animation_name: str) -> None:
        """Trigger a named animation/hotkey."""
        if not self._connected:
            raise RuntimeError("Not connected to VTube Studio")

        if animation_name in self._available_hotkeys:
            await self._send_request(
                "HotkeyTriggerRequest",
                {"hotkeyID": animation_name},
            )
        else:
            self.logger.warning(f"Animation not found: {animation_name}")

    async def get_available_animations(self) -> list[str]:
        """Get list of available animation/hotkey names."""
        return self._available_hotkeys.copy()
