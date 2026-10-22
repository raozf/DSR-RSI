"""Optional slow-loop DM0.5 LoRA adaptation interface.

The trainer records a manifest and delegates actual optimization to an injected
backend.  It never fabricates loss or evaluation metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass
class DM05AdapterTrainer:
    backend: Callable[..., Mapping[str, Any]] | None = None

    def train(self, verified_episodes: list[Mapping[str, Any]], *, output_dir: str | Path, steps: int = 0) -> dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"status": "NOT_RUN", "verified_episodes": len(verified_episodes), "steps": int(steps)}
        if self.backend is not None and verified_episodes and steps > 0:
            result = dict(self.backend(verified_episodes, output_dir=out, steps=steps))
            manifest.update(result)
            manifest.setdefault("status", "COMPLETED")
        (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
