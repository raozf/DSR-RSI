"""Persistent skill registry with explicit validation status."""

from __future__ import annotations

import json
from pathlib import Path

from .skill import Skill


class SkillLibrary:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.skills: dict[str, Skill] = {}
        if self.path and self.path.exists():
            self.skills = {row["name"]: Skill(**row) for row in json.loads(self.path.read_text(encoding="utf-8"))}

    def register(self, skill: Skill) -> None:
        if not skill.name.strip():
            raise ValueError("skill name must be non-empty")
        if not 0.0 <= skill.confidence <= 1.0:
            raise ValueError("skill confidence must be within [0, 1]")
        self.skills[skill.name] = skill
        self.flush()

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def search(self, query: str, top_k: int = 4) -> list[Skill]:
        terms = set(query.lower().split())
        scored = []
        for skill in self.skills.values():
            text = f"{skill.name} {skill.description} {skill.precondition} {skill.postcondition}".lower()
            scored.append((len(terms.intersection(text.split())), skill))
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [skill for score, skill in scored[:top_k] if score > 0]

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([s.to_dict() for s in self.skills.values()], indent=2) + "\n", encoding="utf-8")
