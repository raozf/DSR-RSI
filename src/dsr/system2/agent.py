"""Episode-level recursive improvement agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dsr.diagnosis.failure_schema import DiagnosisContext
from dsr.diagnosis.failure_diagnoser import FailureDiagnoser
from dsr.legacy_evolution.candidate_generator import CandidateGenerator
from dsr.memory.experience_ledger import ExperienceLedger
from dsr.schema import FailureDiagnosis, ModificationProposal


@dataclass
class System2Agent:
    diagnoser: FailureDiagnoser
    generator: CandidateGenerator
    ledger: ExperienceLedger
    backend: Any = None

    def reflect(self, context: DiagnosisContext, *, generation: int, parent_version: str) -> tuple[FailureDiagnosis, ModificationProposal | None]:
        infrastructure = FailureDiagnoser.infrastructure(context)
        if infrastructure is not None:
            return infrastructure, None
        if self.backend is not None:
            return self.backend.reflect(context, generation=generation, parent_version=parent_version)
        diagnosis = self.diagnoser.diagnose(context)
        evidence_ids: tuple[str, ...] = ()
        if context.info.get("episode_id"):
            evidence_ids = (str(context.info["episode_id"]),)
        proposal = self.generator.propose(
            diagnosis,
            generation=generation,
            parent_version=parent_version,
            evidence_ids=evidence_ids,
        )
        return diagnosis, proposal
