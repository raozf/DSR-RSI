"""Versioned RGB observation packet used by isolated random controllers."""

from dataclasses import dataclass, field
from typing import Any, Mapping
import math


SCHEMA = "dsr.random-feature-packet/v1"


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return all(_finite(x) for x in value)
    return False


@dataclass(frozen=True)
class RandomScenePacket:
    episode_id: str
    task_config_identity: str
    frame_id: int
    native_step: int
    sensor_time: float
    calibration_revision: str
    tracks: tuple[Mapping[str, Any], ...] = ()
    clutter_geometry_or_unknown: Any = None
    robot_state: Mapping[str, Any] = field(default_factory=dict)
    holding_hypothesis: Mapping[str, Any] | None = None
    released_state_ledger: tuple[Mapping[str, Any], ...] = ()
    consumed_steps: int = 0
    remaining_steps: int = 0

    def __post_init__(self) -> None:
        if not self.episode_id or not self.task_config_identity or not self.calibration_revision:
            raise ValueError("scene identity is required")
        if self.frame_id < 0 or self.native_step < 0 or self.consumed_steps < 0 or self.remaining_steps < 0:
            raise ValueError("scene counters must be non-negative")
        if not _finite(self.sensor_time):
            raise ValueError("sensor_time must be finite")
        for track in self.tracks:
            if not track.get("visual_id") or "observed_at" not in track:
                raise ValueError("track identity and observation time are required")
            if not _finite(track["observed_at"]):
                raise ValueError("track observation time must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "episode_id": self.episode_id,
            "task_config_identity": self.task_config_identity,
            "frame_id": self.frame_id, "native_step": self.native_step,
            "sensor_time": self.sensor_time, "calibration_revision": self.calibration_revision,
            "tracks": [dict(x) for x in self.tracks],
            "clutter_geometry_or_unknown": self.clutter_geometry_or_unknown,
            "robot_state": dict(self.robot_state),
            "holding_hypothesis": self.holding_hypothesis,
            "released_state_ledger": [dict(x) for x in self.released_state_ledger],
            "consumed_steps": self.consumed_steps, "remaining_steps": self.remaining_steps,
        }


def build_scene_packet(*, episode_id: str, task_config_identity: str,
                       frame_id: int, native_step: int, sensor_time: float,
                       calibration_revision: str, tracks=(),
                       clutter_geometry_or_unknown=None, robot_state=None,
                       holding_hypothesis=None, released_state_ledger=(),
                       consumed_steps=0, remaining_steps=0) -> RandomScenePacket:
    """Build and validate a packet at the RGB/controller boundary."""
    return RandomScenePacket(
        episode_id=episode_id, task_config_identity=task_config_identity,
        frame_id=frame_id, native_step=native_step, sensor_time=sensor_time,
        calibration_revision=calibration_revision, tracks=tuple(tracks),
        clutter_geometry_or_unknown=clutter_geometry_or_unknown,
        robot_state={} if robot_state is None else dict(robot_state),
        holding_hypothesis=holding_hypothesis,
        released_state_ledger=tuple(released_state_ledger),
        consumed_steps=consumed_steps, remaining_steps=remaining_steps,
    )
