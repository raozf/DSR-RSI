"""Serializable contracts shared by DSR-RSI components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


SINGLE_RETRY_PROTOCOL = "single_retry_same_layout/v1"


class FailureType(str, Enum):
    INFRASTRUCTURE = "INFRASTRUCTURE"
    PERCEPTION = "PERCEPTION"
    TASK_PLANNING = "TASK_PLANNING"
    OBJECT_GROUNDING = "OBJECT_GROUNDING"
    SPATIAL_REASONING = "SPATIAL_REASONING"
    MEMORY = "MEMORY"
    SKILL_SELECTION = "SKILL_SELECTION"
    EXECUTION = "EXECUTION"
    PRECISION = "PRECISION"
    RECOVERY = "RECOVERY"
    FALSE_TERMINATION = "FALSE_TERMINATION"
    UNKNOWN = "UNKNOWN"


class UpdateTarget(str, Enum):
    HARNESS_UPDATE = "HARNESS_UPDATE"
    MEMORY_UPDATE = "MEMORY_UPDATE"
    SKILL_UPDATE = "SKILL_UPDATE"
    VLA_UPDATE = "VLA_UPDATE"
    NO_UPDATE = "NO_UPDATE"


@dataclass(frozen=True)
class FailureDiagnosis:
    failure_type: FailureType
    evidence: tuple[str, ...] = ()
    confidence: float = 0.0
    failed_stage: str = "unknown"
    recommended_update: UpdateTarget = UpdateTarget.NO_UPDATE
    source: str = "heuristic"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "evidence": list(self.evidence),
            "confidence": float(self.confidence),
            "failed_stage": self.failed_stage,
            "recommended_update": self.recommended_update.value,
            "source": self.source,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ModificationProposal:
    generation: int
    target: UpdateTarget
    hypothesis: str
    changes: Mapping[str, Any] = field(default_factory=dict)
    parent_version: str = "v0"
    evidence_ids: tuple[str, ...] = ()
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target"] = self.target.value
        data["evidence_ids"] = list(self.evidence_ids)
        return data


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    proposal: ModificationProposal
    before_metrics: Mapping[str, float] = field(default_factory=dict)
    after_metrics: Mapping[str, float] = field(default_factory=dict)
    regression_metrics: Mapping[str, float] = field(default_factory=dict)
    accepted: bool = False
    rejected_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["proposal"] = self.proposal.to_dict()
        return data


@dataclass
class EpisodeTrace:
    episode_id: str
    task: str
    generation: int
    success: bool
    reward: float = 0.0
    steps: int = 0
    observations: list[Mapping[str, Any]] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    diagnosis: FailureDiagnosis | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.diagnosis:
            data["diagnosis"] = self.diagnosis.to_dict()
        return data


def clamp_confidence(value: float) -> float:
    """Keep model-provided confidence within the documented [0, 1] range."""

    return max(0.0, min(1.0, float(value)))
