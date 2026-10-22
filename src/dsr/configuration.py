"""Versioned, explicit runtime configuration; configuration never executes code."""
from __future__ import annotations

import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator


PATH_KEYS = {
    "DSR_POLICY_PYTHON", "DSR_SIM_PYTHON", "DSR_CODE_PYTHON",
    "DSR_TRAIN_PYTHON", "DSR_CHECKPOINT_ROOT", "DSR_ROBODOJO_ASSETS",
    "DSR_CODE_ASSET_ROOT", "DSR_CODEX_HOME", "DSR_ROOT", "DSR_LOCAL_STATE",
}
VALUE_KEYS = {"DSR_KIT_ARGS", "DSR_OMP_THREADS"}
SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema", "environment", "budgets"],
    "additionalProperties": False,
    "properties": {
        "schema": {"const": "dsr.runtime/v1"},
        "environment": {
            "type": "object", "additionalProperties": False,
            "properties": {key: {"type": "string", "minLength": 1}
                           for key in sorted(PATH_KEYS | VALUE_KEYS)},
        },
        "budgets": {
            "type": "object", "additionalProperties": False,
            "required": ["run_timeout_seconds"],
            "properties": {"run_timeout_seconds": {
                "type": "integer", "minimum": 1, "maximum": 86400,
            }},
        },
    },
}


def apply_config(path):
    """Validate completely before mutation; explicit file values override the shell.

    Relative paths resolve against the configuration file, not an implicit cwd.
    No authentication contents, Python modules, or shell commands are loaded.
    """
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    Draft202012Validator(SCHEMA).validate(data)
    environment = {}
    for key, value in data["environment"].items():
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(f"{key} must be a single non-NUL configuration value")
        if key in PATH_KEYS:
            location = Path(value).expanduser()
            if not location.is_absolute():
                location = path.parent / location
            # Preserve interpreter symlinks so virtual environments remain active.
            value = os.path.abspath(location)
        environment[key] = value
    environment["DSR_RUN_TIMEOUT"] = str(data["budgets"]["run_timeout_seconds"])
    os.environ.update(environment)
    return data
