"""Versioned bundle manager.  A rejected candidate never mutates current state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dsr.schema import CandidateEvaluation


class VersionManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_version = "v0"
        self.history: list[dict[str, Any]] = []
        history_path = self.root / "evolution_history.json"
        if history_path.exists():
            self.history = json.loads(history_path.read_text())
            promoted = [row for row in self.history if row.get("accepted") and row.get("version")]
            if promoted:
                self.current_version = promoted[-1]["version"]

    def promote(self, evaluation: CandidateEvaluation) -> str:
        if not evaluation.accepted:
            self.history.append(evaluation.to_dict())
            return self.current_version
        next_version = f"v{len(self.history) + 1}"
        payload = evaluation.to_dict() | {"version": next_version, "parent": self.current_version}
        (self.root / f"{next_version}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.history.append(payload)
        self.current_version = next_version
        return next_version

    def save_history(self) -> Path:
        path = self.root / "evolution_history.json"
        path.write_text(json.dumps(self.history, indent=2) + "\n", encoding="utf-8")
        return path
