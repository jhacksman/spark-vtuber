"""Avatar control system for Spark VTuber."""

from spark_vtuber.avatar.base import BaseAvatar
from spark_vtuber.avatar.vtube_studio import VTubeStudioAvatar
from spark_vtuber.avatar.dual_vtube_studio import DualVTubeStudioAvatar

__all__ = ["BaseAvatar", "VTubeStudioAvatar", "DualVTubeStudioAvatar"]
