"""Small deterministic retriever for structured experience."""

from __future__ import annotations

from collections.abc import Iterable

from .experience_ledger import ExperienceLedger, ExperienceRecord


class ExperienceRetriever:
    def __init__(self, ledger: ExperienceLedger):
        self.ledger = ledger

    def retrieve(self, *, task: str, failure_type: str | None = None, top_k: int = 4) -> list[ExperienceRecord]:
        task_tokens = set(task.lower().split())
        scored: list[tuple[float, ExperienceRecord]] = []
        for record in self.ledger.records:
            score = len(task_tokens.intersection(record.task.lower().split()))
            if failure_type and record.failure_type == failure_type:
                score += 2
            score += min(record.success_after_reuse, 2) * 0.25
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda pair: (-pair[0], pair[1].record_id))
        return [record for _, record in scored[: max(0, top_k)]]

