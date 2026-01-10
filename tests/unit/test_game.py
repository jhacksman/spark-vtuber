"""Tests for game module."""

import pytest

from spark_vtuber.game.base import GameState, GameAction, GameStatus


class TestGameStatus:
    """Tests for GameStatus enum."""

    def test_status_values(self):
        """Test status values."""
        assert GameStatus.DISCONNECTED.value == "disconnected"
        assert GameStatus.CONNECTING.value == "connecting"
        assert GameStatus.CONNECTED.value == "connected"
        assert GameStatus.PLAYING.value == "playing"


class TestGameState:
    """Tests for GameState class."""

    def test_create_state(self):
        """Test state creation."""
        state = GameState(
            status=GameStatus.CONNECTED,
            game_name="Minecraft",
            current_scene="Overworld",
        )
        assert state.status == GameStatus.CONNECTED
        assert state.game_name == "Minecraft"
        assert state.current_scene == "Overworld"

    def test_default_values(self):
        """Test default values."""
        state = GameState()
        assert state.status == GameStatus.DISCONNECTED
        assert state.player_health == 1.0
        assert state.score == 0

    def test_to_prompt_context(self):
        """Test prompt context generation."""
        state = GameState(
            game_name="Minecraft",
            current_scene="Overworld",
            player_position=(100.0, 64.0, -50.0),
            player_health=0.8,
            score=1000,
        )
        context = state.to_prompt_context()
        assert "Minecraft" in context
        assert "Overworld" in context
        assert "100.0" in context
        assert "80%" in context
        assert "1000" in context

    def test_to_prompt_context_minimal(self):
        """Test minimal prompt context."""
        state = GameState(game_name="TestGame")
        context = state.to_prompt_context()
        assert "TestGame" in context


class TestGameAction:
    """Tests for GameAction class."""

    def test_create_action(self):
        """Test action creation."""
        action = GameAction(
            action_type="move",
            parameters={"direction": "forward", "duration": 1.0},
            description="Move forward",
        )
        assert action.action_type == "move"
        assert action.parameters["direction"] == "forward"
        assert action.description == "Move forward"

    def test_default_values(self):
        """Test default values."""
        action = GameAction(action_type="jump")
        assert action.parameters == {}
        assert action.description == ""
