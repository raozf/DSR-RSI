"""Append-only structured experience storage."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass
class ExperienceRecord:
    task: str
    scene: str = "unknown"
    goal: str = ""
    failure_type: str = "UNKNOWN"
    successful_strategy: str = ""
    failed_strategy: str = ""
    recovery: str = ""
    precondition: str = ""
    postcondition: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source_episode: str = ""
    reuse_count: int = 0
    success_after_reuse: int = 0
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperienceLedger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records: list[ExperienceRecord] = []
        if self.path and self.path.exists():
            self._load()

    def append(self, record: ExperienceRecord | Mapping[str, Any]) -> ExperienceRecord:
        item = record if isinstance(record, ExperienceRecord) else ExperienceRecord(**dict(record))
        self.records.append(item)
        self.flush()
        return item

    def extend(self, records: Iterable[ExperienceRecord | Mapping[str, Any]]) -> None:
        for record in records:
            self.append(record)

    def retrieve(self, *, task: str, failure_type: str | None = None, top_k: int = 4) -> list[ExperienceRecord]:
        """Convenience retrieval API used by System 1 without leaking a retriever object."""

        from .retriever import ExperienceRetriever

        return ExperienceRetriever(self).retrieve(task=task, failure_type=failure_type, top_k=top_k)

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([r.to_dict() for r in self.records], indent=2) + "\n", encoding="utf-8")

    def _load(self) -> None:
        rows = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = [ExperienceRecord(**row) for row in rows]
