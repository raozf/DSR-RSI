"""Closed-loop geometry adapters for isolated random-task candidates.

The package is deliberately separate from the active policy registry.  These
functions consume RGB-derived estimates and return bounded updates; they never
read simulator state or checker results.
"""

from .bowl_profile import BowlProfileInterval, estimate_profile_interval
from .push_response import PushResponseModel
from .scene_packet import RandomScenePacket, build_scene_packet

__all__ = [
    "BowlProfileInterval", "estimate_profile_interval", "PushResponseModel",
    "RandomScenePacket", "build_scene_packet",
]
