"""Input contract for failure diagnosis.

The context deliberately accepts mappings so simulator-specific objects do not
leak into the core package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DiagnosisContext:
    instruction: str
    observation: Mapping[str, Any]
    action: Any = None
    next_observation: Mapping[str, Any] | None = None
    reward: float | None = None
    info: Mapping[str, Any] = field(default_factory=dict)
    history: tuple[Mapping[str, Any], ...] = ()

