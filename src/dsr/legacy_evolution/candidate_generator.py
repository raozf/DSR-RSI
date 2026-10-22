"""Generate bounded candidate modifications from diagnosis and experience."""

from __future__ import annotations

from dsr.schema import FailureDiagnosis, ModificationProposal, UpdateTarget


class CandidateGenerator:
    def propose(
        self,
        diagnosis: FailureDiagnosis,
        *,
        generation: int,
        parent_version: str,
        evidence_ids: tuple[str, ...] = (),
    ) -> ModificationProposal | None:
        if diagnosis.recommended_update is UpdateTarget.NO_UPDATE:
            return None
        hypothesis = {
            UpdateTarget.HARNESS_UPDATE: "adjust execution guard or task decomposition",
            UpdateTarget.MEMORY_UPDATE: "retain state needed to avoid repeated failure",
            UpdateTarget.SKILL_UPDATE: "register a bounded recovery skill for this failure",
            UpdateTarget.VLA_UPDATE: "adapt policy on repeated verified precision failures",
        }[diagnosis.recommended_update]
        changes = {
            "failure_type": diagnosis.failure_type.value,
            "failed_stage": diagnosis.failed_stage,
            "evidence": list(diagnosis.evidence),
        }
        return ModificationProposal(
            generation=generation,
            target=diagnosis.recommended_update,
            hypothesis=hypothesis,
            changes=changes,
            parent_version=parent_version,
            evidence_ids=evidence_ids,
            confidence=diagnosis.confidence,
        )
