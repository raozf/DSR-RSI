"""Evaluation protocol for temporary candidate versions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import Any, Mapping

from dsr.schema import CandidateEvaluation, ModificationProposal


class CandidateEvaluator:
    def __init__(self, rollout: Callable[[ModificationProposal, int], Mapping[str, float]] | None = None):
        self.rollout = rollout or self._default_rollout

    def evaluate(
        self,
        proposal: ModificationProposal,
        *,
        before: Mapping[str, float],
        seeds: Iterable[int] = (0,),
        regression: Mapping[str, float] | None = None,
    ) -> CandidateEvaluation:
        after_samples = [self.rollout(proposal, int(seed)) for seed in seeds]
        after = self._mean(after_samples)
        candidate_id = hashlib.sha256(repr(proposal.to_dict()).encode()).hexdigest()[:12]
        return CandidateEvaluation(
            candidate_id,
            proposal,
            dict(before),
            after,
            dict(regression or {}),
            metadata={"evaluation_backend": getattr(self.rollout, "__name__", type(self.rollout).__name__)},
        )

    @staticmethod
    def _mean(rows: list[Mapping[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        keys = set().union(*(row.keys() for row in rows))
        return {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}

    @staticmethod
    def _default_rollout(_proposal: ModificationProposal, _seed: int) -> Mapping[str, float]:
        # No claim of performance: this is only the deterministic dry-run hook.
        return {"success_rate": 0.0, "score": 0.0}
