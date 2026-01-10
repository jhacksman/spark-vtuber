"""
Spark VTuber - AI-powered VTuber streaming system for NVIDIA DGX Spark.

This package provides a complete AI VTuber streaming system with:
- LLM-based conversational AI with dual personalities
- Real-time text-to-speech synthesis
- Speech-to-text for collaborator audio
- Long-term memory with RAG
- Avatar control with lip sync
- Chat integration (Twitch/YouTube)
- Game integration framework
"""

__version__ = "0.1.0"
__author__ = "Jack Hacksman"

from spark_vtuber.config.settings import Settings

__all__ = ["Settings", "__version__"]
