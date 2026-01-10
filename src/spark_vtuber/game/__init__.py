"""Game integration framework for Spark VTuber."""

from spark_vtuber.game.base import BaseGame, GameState, GameAction
from spark_vtuber.game.watch_mode import WatchMode

__all__ = ["BaseGame", "GameState", "GameAction", "WatchMode"]
