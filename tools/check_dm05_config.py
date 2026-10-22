#!/usr/bin/env python3
"""Check consistency of DM0.5 dataset/training/checkpoint/deploy metadata.

The checker is deliberately schema based and never guesses dimensions.  Missing
fields are reported as errors so a real checkpoint can only be marked verified
when its metadata is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = ("action_dim", "action_horizon", "history_length", "camera_order", "state_dim", "normalization")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def check(paths: list[Path]) -> dict[str, Any]:
    docs = {str(path): load(path) for path in paths}
    missing = {name: [path for path, doc in docs.items() if name not in doc] for name in REQUIRED}
    missing = {name: values for name, values in missing.items() if values}
    values: dict[str, list[Any]] = {}
    for name in REQUIRED:
        values[name] = [doc[name] for doc in docs.values() if name in doc]
    mismatches = {name: rows for name, rows in values.items() if rows and any(row != rows[0] for row in rows[1:])}
    blocked = {
        path: str(doc.get("status"))
        for path, doc in docs.items()
        if str(doc.get("status", "")).startswith(("BLOCKED", "NOT_RUN", "FAILED"))
    }
    status = "FAILED" if missing or mismatches else ("BLOCKED" if blocked else "VERIFIED")
    return {
        "status": status,
        "files": list(docs),
        "missing": missing,
        "mismatches": mismatches,
        "blocked_or_unverified_files": blocked,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.paths)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
