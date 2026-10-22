"""Failure-to-update diagnosis used by System 2."""

from __future__ import annotations

from typing import Any, Mapping

from dsr.schema import FailureDiagnosis, FailureType, UpdateTarget, clamp_confidence
from .failure_schema import DiagnosisContext


class FailureDiagnoser:
    """Deterministic first-pass diagnosis with a typed output schema.

    A deployment may replace this with a VLM/LLM implementation as long as it
    returns :class:`FailureDiagnosis` and preserves the fail-closed defaults.
    """

    _target_by_failure = {
        FailureType.INFRASTRUCTURE: UpdateTarget.NO_UPDATE,
        FailureType.PERCEPTION: UpdateTarget.HARNESS_UPDATE,
        FailureType.TASK_PLANNING: UpdateTarget.HARNESS_UPDATE,
        FailureType.OBJECT_GROUNDING: UpdateTarget.HARNESS_UPDATE,
        FailureType.SPATIAL_REASONING: UpdateTarget.HARNESS_UPDATE,
        FailureType.MEMORY: UpdateTarget.MEMORY_UPDATE,
        FailureType.SKILL_SELECTION: UpdateTarget.SKILL_UPDATE,
        FailureType.EXECUTION: UpdateTarget.VLA_UPDATE,
        FailureType.PRECISION: UpdateTarget.VLA_UPDATE,
        FailureType.RECOVERY: UpdateTarget.SKILL_UPDATE,
        FailureType.FALSE_TERMINATION: UpdateTarget.HARNESS_UPDATE,
        FailureType.UNKNOWN: UpdateTarget.NO_UPDATE,
    }

    @staticmethod
    def infrastructure(context: DiagnosisContext) -> FailureDiagnosis | None:
        """Route explicit execution faults before interpreting task performance.

        A retry infrastructure fault does not relabel the original completed
        rollout. Only the status/domain of the attempt being diagnosed applies.
        Error text and robot images alone cannot establish an infrastructure fault.
        """
        info = context.info
        if not (info.get("infrastructure_failure") is True
                or info.get("status") == "INFRASTRUCTURE_FAILED"
                or str(info.get("failure_type", "")).upper() == "INFRASTRUCTURE"
                or info.get("failure_domain") == "infrastructure"):
            return None
        refs = info.get("evidence", ())
        if isinstance(refs, str):
            refs = (refs,)
        return FailureDiagnosis(
            FailureType.INFRASTRUCTURE, tuple(str(ref) for ref in refs), 1.0,
            str(info.get("failed_stage", "runtime")), UpdateTarget.NO_UPDATE,
            "explicit_runtime_status",
            details={"capability_update_eligible": False,
                     "reason": str(info.get("error", "Execution could not be evaluated")),
                     "next_action": "repair_infrastructure_and_rerun_same_case"},
        )

    def diagnose(self, context: DiagnosisContext) -> FailureDiagnosis:
        infrastructure = self.infrastructure(context)
        if infrastructure is not None:
            return infrastructure
        info = {str(k): v for k, v in context.info.items()}
        package = info.get("failure_evidence_package")
        if isinstance(package, Mapping):
            # The recorder distinguishes the earliest observed deviation from
            # the terminal symptom. Preserve that distinction: an observed
            # discrepancy supports only a low-confidence execution diagnosis;
            # it does not establish a hidden physical cause.
            deviation = package.get("earliest_observed_deviation")
            if isinstance(deviation, Mapping):
                refs = tuple(str(ref) for ref in deviation.get("evidence_refs", ())
                             if isinstance(ref, str))
                evidence = refs or (str(deviation.get("ref", "stage observation")),)
                return FailureDiagnosis(
                    FailureType.EXECUTION, evidence, 0.35,
                    str(deviation.get("stage") or "observed deviation"),
                    UpdateTarget.VLA_UPDATE, "evidence_observation",
                    details={"observation_only": True,
                             "earliest_observed_deviation": dict(deviation),
                             "final_visible_failure": package.get("final_visible_failure"),
                             "uncertainty": package.get("uncertainty", ""),
                             "cause_candidates": package.get("cause_candidates", [])},
                )
        explicit = str(info.get("failure_type", "")).upper()
        if explicit in FailureType.__members__:
            kind = FailureType[explicit]
            evidence = tuple(str(item) for item in info.get("evidence", ()))
            confidence = clamp_confidence(float(info.get("confidence", 0.8)))
            stage = str(info.get("failed_stage", kind.value.lower()))
            return FailureDiagnosis(kind, evidence, confidence, stage, self._target_by_failure[kind], "explicit")

        before = context.observation
        after = context.next_observation or {}
        if info.get("false_termination") or before.get("terminated_early"):
            return FailureDiagnosis(
                FailureType.FALSE_TERMINATION,
                ("termination predicate contradicted by task state",),
                0.9,
                "termination",
                UpdateTarget.HARNESS_UPDATE,
            )
        if info.get("precision_error") or info.get("tracking_error"):
            return FailureDiagnosis(
                FailureType.PRECISION,
                ("controller tracking error exceeded tolerance",),
                0.85,
                "execution",
                UpdateTarget.VLA_UPDATE,
            )
        if info.get("recovery_failed"):
            return FailureDiagnosis(
                FailureType.RECOVERY,
                ("previous recovery primitive did not restore progress",),
                0.8,
                "recovery",
                UpdateTarget.SKILL_UPDATE,
            )
        if info.get("memory_miss"):
            return FailureDiagnosis(
                FailureType.MEMORY,
                ("retrieved experience did not include required state",),
                0.75,
                "retrieval",
                UpdateTarget.MEMORY_UPDATE,
            )
        if info.get("wrong_object") or info.get("grounding_error"):
            return FailureDiagnosis(
                FailureType.OBJECT_GROUNDING,
                ("selected object did not match instruction",),
                0.8,
                "grounding",
                UpdateTarget.HARNESS_UPDATE,
            )

        progress_before = self._progress(before)
        progress_after = self._progress(after)
        if progress_before is not None and progress_after is not None and progress_after < progress_before:
            return FailureDiagnosis(
                FailureType.EXECUTION,
                (f"progress decreased from {progress_before:.3f} to {progress_after:.3f}",),
                0.65,
                "execution",
                UpdateTarget.VLA_UPDATE,
            )
        if context.reward is not None and context.reward < 0:
            return FailureDiagnosis(
                FailureType.TASK_PLANNING,
                (f"negative reward {context.reward:.3f}",),
                0.55,
                "planning",
                UpdateTarget.HARNESS_UPDATE,
            )
        return FailureDiagnosis(
            FailureType.UNKNOWN,
            ("no typed failure evidence available",),
            0.1,
            "unknown",
            UpdateTarget.NO_UPDATE,
        )

    @staticmethod
    def _progress(observation: Mapping[str, Any]) -> float | None:
        value = observation.get("progress")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
