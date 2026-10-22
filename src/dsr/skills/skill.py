"""Reusable embodied skill schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Skill:
    name: str
    description: str
    precondition: str
    controller: list[str] = field(default_factory=list)
    fallback: list[str] = field(default_factory=list)
    postcondition: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reuse_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
