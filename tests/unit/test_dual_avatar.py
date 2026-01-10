"""Tests for dual avatar module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spark_vtuber.avatar.base import Emotion, LipSyncFrame, ExpressionState
from spark_vtuber.avatar.dual_vtube_studio import DualVTubeStudioAvatar


class TestDualVTubeStudioAvatar:
    """Tests for DualVTubeStudioAvatar class."""

    @pytest.fixture
    def dual_avatar(self):
        """Create a dual avatar instance."""
        return DualVTubeStudioAvatar(
            primary_host="localhost",
            primary_port=8001,
            secondary_host="localhost",
            secondary_port=8002,
        )

    def test_init(self, dual_avatar):
        """Test dual avatar initialization."""
        assert dual_avatar.primary is not None
        assert dual_avatar.secondary is not None
        assert dual_avatar._active_speaker == "primary"
        assert not dual_avatar._connected

    def test_init_custom_ports(self):
        """Test initialization with custom ports."""
        avatar = DualVTubeStudioAvatar(
            primary_port=9001,
            secondary_port=9002,
        )
        assert avatar.primary.port == 9001
        assert avatar.secondary.port == 9002

    def test_active_speaker_property(self, dual_avatar):
        """Test active speaker property."""
        assert dual_avatar.active_speaker == "primary"

    def test_expression_properties(self, dual_avatar):
        """Test expression state properties."""
        assert dual_avatar.primary_expression.emotion == Emotion.NEUTRAL
        assert dual_avatar.secondary_expression.emotion == Emotion.NEUTRAL

    @pytest.mark.asyncio
    async def test_set_active_speaker_primary(self, dual_avatar):
        """Test setting active speaker to primary."""
        await dual_avatar.set_active_speaker("primary")
        assert dual_avatar._active_speaker == "primary"

    @pytest.mark.asyncio
    async def test_set_active_speaker_secondary(self, dual_avatar):
        """Test setting active speaker to secondary."""
        await dual_avatar.set_active_speaker("secondary")
        assert dual_avatar._active_speaker == "secondary"

    @pytest.mark.asyncio
    async def test_set_active_speaker_invalid(self, dual_avatar):
        """Test setting invalid active speaker raises error."""
        with pytest.raises(ValueError, match="Invalid speaker"):
            await dual_avatar.set_active_speaker("invalid")

    @pytest.mark.asyncio
    async def test_connect_both_avatars(self, dual_avatar):
        """Test connecting both avatars."""
        dual_avatar.primary.connect = AsyncMock()
        dual_avatar.secondary.connect = AsyncMock()

        await dual_avatar.connect()

        dual_avatar.primary.connect.assert_called_once()
        dual_avatar.secondary.connect.assert_called_once()
        assert dual_avatar._connected

    @pytest.mark.asyncio
    async def test_connect_partial_failure(self, dual_avatar):
        """Test partial connection failure (one avatar fails)."""
        dual_avatar.primary.connect = AsyncMock()
        dual_avatar.secondary.connect = AsyncMock(side_effect=Exception("Connection failed"))

        await dual_avatar.connect()

        assert dual_avatar._connected

    @pytest.mark.asyncio
    async def test_connect_total_failure(self, dual_avatar):
        """Test total connection failure (both avatars fail)."""
        dual_avatar.primary.connect = AsyncMock(side_effect=Exception("Primary failed"))
        dual_avatar.secondary.connect = AsyncMock(side_effect=Exception("Secondary failed"))

        with pytest.raises(ConnectionError):
            await dual_avatar.connect()

    @pytest.mark.asyncio
    async def test_disconnect_both_avatars(self, dual_avatar):
        """Test disconnecting both avatars."""
        dual_avatar._connected = True
        dual_avatar.primary.disconnect = AsyncMock()
        dual_avatar.secondary.disconnect = AsyncMock()

        await dual_avatar.disconnect()

        dual_avatar.primary.disconnect.assert_called_once()
        dual_avatar.secondary.disconnect.assert_called_once()
        assert not dual_avatar._connected

    @pytest.mark.asyncio
    async def test_update_lip_sync_primary(self, dual_avatar):
        """Test lip sync updates go to primary when active."""
        dual_avatar._connected = True
        dual_avatar._active_speaker = "primary"
        dual_avatar.primary._connected = True
        dual_avatar.primary.update_lip_sync = AsyncMock()
        dual_avatar.secondary.update_lip_sync = AsyncMock()

        frame = LipSyncFrame(phoneme="A", intensity=0.8)
        await dual_avatar.update_lip_sync(frame)

        dual_avatar.primary.update_lip_sync.assert_called_once_with(frame)
        dual_avatar.secondary.update_lip_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_lip_sync_secondary(self, dual_avatar):
        """Test lip sync updates go to secondary when active."""
        dual_avatar._connected = True
        dual_avatar._active_speaker = "secondary"
        dual_avatar.secondary._connected = True
        dual_avatar.primary.update_lip_sync = AsyncMock()
        dual_avatar.secondary.update_lip_sync = AsyncMock()

        frame = LipSyncFrame(phoneme="O", intensity=0.9)
        await dual_avatar.update_lip_sync(frame)

        dual_avatar.secondary.update_lip_sync.assert_called_once_with(frame)
        dual_avatar.primary.update_lip_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_expression_primary(self, dual_avatar):
        """Test expression updates go to primary when active."""
        dual_avatar._active_speaker = "primary"
        dual_avatar.primary._connected = True
        dual_avatar.primary.set_expression = AsyncMock()

        await dual_avatar.set_expression(Emotion.HAPPY, 0.8)

        dual_avatar.primary.set_expression.assert_called_once_with(Emotion.HAPPY, 0.8)
        assert dual_avatar._primary_expression.emotion == Emotion.HAPPY

    @pytest.mark.asyncio
    async def test_set_expression_secondary(self, dual_avatar):
        """Test expression updates go to secondary when active."""
        dual_avatar._active_speaker = "secondary"
        dual_avatar.secondary._connected = True
        dual_avatar.secondary.set_expression = AsyncMock()

        await dual_avatar.set_expression(Emotion.SAD, 0.7)

        dual_avatar.secondary.set_expression.assert_called_once_with(Emotion.SAD, 0.7)
        assert dual_avatar._secondary_expression.emotion == Emotion.SAD

    @pytest.mark.asyncio
    async def test_set_expression_both(self, dual_avatar):
        """Test setting expressions for both avatars simultaneously."""
        dual_avatar.primary._connected = True
        dual_avatar.secondary._connected = True
        dual_avatar.primary.set_expression = AsyncMock()
        dual_avatar.secondary.set_expression = AsyncMock()

        await dual_avatar.set_expression_both(Emotion.HAPPY, Emotion.ANGRY, 0.9)

        dual_avatar.primary.set_expression.assert_called_once_with(Emotion.HAPPY, 0.9)
        dual_avatar.secondary.set_expression.assert_called_once_with(Emotion.ANGRY, 0.9)

    @pytest.mark.asyncio
    async def test_set_primary_expression_directly(self, dual_avatar):
        """Test setting primary expression directly."""
        dual_avatar.primary._connected = True
        dual_avatar.primary.set_expression = AsyncMock()

        await dual_avatar.set_primary_expression(Emotion.EXCITED, 1.0)

        dual_avatar.primary.set_expression.assert_called_once_with(Emotion.EXCITED, 1.0)

    @pytest.mark.asyncio
    async def test_set_secondary_expression_directly(self, dual_avatar):
        """Test setting secondary expression directly."""
        dual_avatar.secondary._connected = True
        dual_avatar.secondary.set_expression = AsyncMock()

        await dual_avatar.set_secondary_expression(Emotion.CONFUSED, 0.5)

        dual_avatar.secondary.set_expression.assert_called_once_with(Emotion.CONFUSED, 0.5)

    def test_get_connection_status(self, dual_avatar):
        """Test getting connection status for both avatars."""
        dual_avatar.primary._connected = True
        dual_avatar.secondary._connected = False

        status = dual_avatar.get_connection_status()

        assert status["primary"] is True
        assert status["secondary"] is False

    @pytest.mark.asyncio
    async def test_set_parameter_routes_to_active(self, dual_avatar):
        """Test parameter setting routes to active speaker."""
        dual_avatar._active_speaker = "primary"
        dual_avatar.primary.set_parameter = AsyncMock()
        dual_avatar.secondary.set_parameter = AsyncMock()

        await dual_avatar.set_parameter("MouthOpen", 0.5)

        dual_avatar.primary.set_parameter.assert_called_once_with("MouthOpen", 0.5)
        dual_avatar.secondary.set_parameter.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_animation_routes_to_active(self, dual_avatar):
        """Test animation triggering routes to active speaker."""
        dual_avatar._active_speaker = "secondary"
        dual_avatar.primary.trigger_animation = AsyncMock()
        dual_avatar.secondary.trigger_animation = AsyncMock()

        await dual_avatar.trigger_animation("wave")

        dual_avatar.secondary.trigger_animation.assert_called_once_with("wave")
        dual_avatar.primary.trigger_animation.assert_not_called()


class TestDualAvatarBackwardCompatibility:
    """Tests for backward compatibility with single avatar mode."""

    def test_single_avatar_still_works(self):
        """Test that single avatar mode still works."""
        from spark_vtuber.avatar.vtube_studio import VTubeStudioAvatar

        avatar = VTubeStudioAvatar(
            host="localhost",
            port=8001,
        )
        assert avatar.host == "localhost"
        assert avatar.port == 8001

    def test_dual_avatar_inherits_base(self):
        """Test that DualVTubeStudioAvatar inherits from BaseAvatar."""
        from spark_vtuber.avatar.base import BaseAvatar

        avatar = DualVTubeStudioAvatar()
        assert isinstance(avatar, BaseAvatar)
