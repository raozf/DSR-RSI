"""Promotion gate with regression checks and fail-closed defaults."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from dsr.schema import CandidateEvaluation


class EvidenceGate:
    def __init__(self, *, min_improvement: float = 0.0, max_regression: float = 0.02):
        self.min_improvement = float(min_improvement)
        self.max_regression = float(max_regression)

    def decide(self, evaluation: CandidateEvaluation) -> CandidateEvaluation:
        before = float(evaluation.before_metrics.get("success_rate", 0.0))
        after = float(evaluation.after_metrics.get("success_rate", 0.0))
        regression = float(evaluation.regression_metrics.get("success_rate", before))
        gain = after - before
        if gain < self.min_improvement:
            return replace(evaluation, accepted=False, rejected_reason=f"insufficient improvement ({gain:.4f})")
        if before - regression > self.max_regression:
            return replace(evaluation, accepted=False, rejected_reason="regression threshold exceeded")
        return replace(evaluation, accepted=True, rejected_reason=None)

