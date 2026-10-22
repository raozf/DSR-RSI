"""Distill repeated verified experiences into a reusable skill."""

from __future__ import annotations

from collections import Counter

from dsr.memory.experience_ledger import ExperienceRecord
from .skill import Skill


class SkillDistiller:
    def __init__(self, min_support: int = 2):
        self.min_support = max(1, int(min_support))

    def distill(self, records: list[ExperienceRecord]) -> Skill | None:
        verified = [r for r in records if r.postcondition and r.confidence >= 0.5 and r.recovery]
        if len(verified) < self.min_support:
            return None
        recovery, count = Counter(r.recovery for r in verified).most_common(1)[0]
        if count < self.min_support:
            return None
        representative = max(verified, key=lambda r: r.confidence)
        return Skill(
            name=recovery,
            description=f"validated recovery distilled from {count} episodes",
            precondition=representative.precondition,
            controller=[recovery],
            fallback=[representative.failed_strategy] if representative.failed_strategy else [],
            postcondition=representative.postcondition,
            evidence_ids=[r.record_id for r in verified if r.recovery == recovery],
            confidence=sum(r.confidence for r in verified if r.recovery == recovery) / count,
        )
