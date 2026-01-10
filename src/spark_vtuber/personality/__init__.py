"""Dual AI personality system for Spark VTuber."""

from spark_vtuber.personality.base import Personality, PersonalityConfig
from spark_vtuber.personality.manager import PersonalityManager
from spark_vtuber.personality.coordinator import DialogueCoordinator

__all__ = ["Personality", "PersonalityConfig", "PersonalityManager", "DialogueCoordinator"]
