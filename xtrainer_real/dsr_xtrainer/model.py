"""Explicit units and a small task -> skills -> executor contract."""
from dataclasses import asdict, dataclass, field
from typing import Any


class PreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Step:
    kind: str
    stage: str
    arm: str | None = None
    pose_world_mm_deg: list[float] | None = None
    opening: float | None = None  # 0 = calibrated closed; 1 = calibrated open
    expected: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""


@dataclass
class Plan:
    task: str
    steps: list[Step]
    memory: dict[str, Any]
    schema: str = "dsr.xtrainer-plan/v1"
    controller: str = "xtrainer-calibrated-code-policy/v1"

    def to_dict(self):
        return asdict(self)
