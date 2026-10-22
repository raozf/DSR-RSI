"""Auditable, bounded Codex proposals for robotic execution, never fake scores."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from dsr.harness.runtime import normalize_native_harness
from dsr.diagnosis.failure_diagnoser import FailureDiagnoser
from dsr.schema import FailureDiagnosis, FailureType, ModificationProposal, UpdateTarget
from dsr.system2.retry_guidance import (
    FAILURE_ANALYSIS_SCHEMA,
    RETRY_PLAN_SCHEMA,
    REPAIR_INSTRUCTIONS,
    validate_evidence_refs,
    validate_repair_details,
)


class InvalidSupervisorResponse(ValueError):
    """The supervisor answered, but its proposal cannot be executed safely."""


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _validate_patch(data):
    try:
        if not isinstance(data, dict):
            raise ValueError("harness_patch must be an object")
        # Validate against the complete schema, but preserve patch sparsity so
        # an omitted field inherits the parent harness instead of silently
        # clearing it. This path supports historical harness transports; the
        # current CLI produces a prompt-only structured retry plan.
        normalized = normalize_native_harness(data, check_references=False)
        return {key: normalized[key] for key in data}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid harness_patch: {exc}") from exc


def validate_response(data):
    legacy_required = {"failure_type", "confidence", "evidence", "instruction_suffix", "action_steps", "reason"}
    harness_required = {"failure_type", "confidence", "evidence", "harness_patch", "reason"}
    retry_required = {"failure_type", "confidence", "evidence", "reason", "failure_analysis", "retry_plan"}
    if not isinstance(data, dict) or set(data) not in (
        legacy_required, harness_required, retry_required, retry_required | {"harness_patch"},
    ):
        raise ValueError("Agent response does not match the bounded repair schema")
    data = dict(data)
    FailureType(data["failure_type"])
    confidence = data["confidence"]
    if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid diagnosis confidence")
    if "failure_analysis" in data:
        compiled = validate_repair_details(data)
        patch = {"instruction_suffix": compiled}
        if "harness_patch" in data and data["harness_patch"] != patch:
            raise ValueError("Prompt-only retry cannot change other harness controls")
        data["harness_patch"] = patch
    elif "harness_patch" in data:
        data["harness_patch"] = _validate_patch(data["harness_patch"])
    else:
        if type(data["action_steps"]) is not int or not 1 <= data["action_steps"] <= 25:
            raise ValueError("Action horizon must be in [1, 25]")
        data["harness_patch"] = _validate_patch({
            "instruction_suffix": data.pop("instruction_suffix"),
            "action_steps": data.pop("action_steps"),
        })
    for name, maximum in (("reason", 1500),):
        if not isinstance(data[name], str) or len(data[name]) > maximum:
            raise ValueError(f"Invalid {name}")
    if not isinstance(data["evidence"], list) or not all(isinstance(x, str) for x in data["evidence"]):
        raise ValueError("Evidence must be a list of strings")
    return data


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "failure_type": {"type": "string", "enum": [x.value for x in FailureType]},
        "confidence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
        "failure_analysis": FAILURE_ANALYSIS_SCHEMA,
        "retry_plan": RETRY_PLAN_SCHEMA,
    },
    "required": ["failure_type", "confidence", "evidence", "reason",
                 "failure_analysis", "retry_plan"],
}



class CodexBackend:
    def __init__(self, output_dir, *, config_home, timeout=600, proxy=None):
        self.output_dir = Path(output_dir).resolve()
        self.config_home = Path(config_home).resolve()
        if not (self.config_home / "config.toml").is_file():
            raise FileNotFoundError(self.config_home / "config.toml")
        self.executable = shutil.which("codex")
        if not self.executable:
            raise FileNotFoundError("codex CLI is not on PATH")
        self.timeout = timeout
        self.proxy = proxy or os.environ.get("DSR_CODEX_PROXY")

    def decide(self, context, *, phase, directory, images=()):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        schema = directory / "schema.json"
        answer = directory / "answer.json"
        if answer.exists():
            raise FileExistsError(f"Refusing to reuse an old agent answer: {answer}")
        write_json(schema, SCHEMA)
        prompt = (
            REPAIR_INSTRUCTIONS + "\n"
            + json.dumps({"phase": phase, "context": context}, allow_nan=False)
        )
        (directory / "prompt.txt").write_text(prompt + "\n")
        command = [
            self.executable, "-a", "never", "exec", "--skip-git-repo-check", "--ephemeral",
            "-c", 'model_reasoning_effort="high"',
            "-s", "read-only", "-C", str(directory), "--json",
            "--output-schema", str(schema), "-o", str(answer),
        ]
        for image in images:
            command.extend(["--image", str(Path(image).resolve())])
        command.append("-")
        env = os.environ.copy()
        env["CODEX_HOME"] = str(self.config_home)
        if self.proxy:
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                env[key] = self.proxy
        for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"):
            env.pop(key, None)
        started = time.monotonic()
        write_json(directory / "invocation.json", {
            "command": command, "config_home": str(self.config_home),
            "role": "robot_supervisor", "phase": phase, "status": "RUNNING",
            "proxy_configured": bool(self.proxy),
        })
        with (directory / "events.jsonl").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout,
                                       stderr=stderr, env=env, text=True)
            try:
                process.communicate(prompt, timeout=self.timeout)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                write_json(directory / "status.json", {"status": "FAILED", "elapsed": time.monotonic() - started})
                raise
        if process.returncode or not answer.is_file():
            write_json(directory / "status.json", {"status": "FAILED", "exit_code": process.returncode})
            raise RuntimeError(f"Codex supervisor failed; inspect {directory}")
        for line in (directory / "events.jsonl").read_text().splitlines():
            event = json.loads(line)
            item = event.get("item", {})
            if item.get("type") in ("command_execution", "file_change", "mcp_tool_call", "web_search"):
                write_json(directory / "status.json", {"status": "FAILED", "reason": "unexpected tool use"})
                raise RuntimeError("Supervisor used tools instead of only the supplied episode evidence")
        try:
            response = validate_response(json.loads(answer.read_text()))
            validate_evidence_refs(response, context.get("evidence_catalog", []))
        except (TypeError, ValueError) as exc:
            write_json(directory / "status.json", {
                "status": "REJECTED_INVALID_RESPONSE", "exit_code": 0,
                "elapsed": time.monotonic() - started, "error": str(exc),
            })
            raise InvalidSupervisorResponse(str(exc)) from exc
        write_json(directory / "status.json", {
            "status": "COMPLETED", "exit_code": 0, "elapsed": time.monotonic() - started,
        })
        return response

    def reflect(self, context, *, generation, parent_version):
        infrastructure = FailureDiagnoser.infrastructure(context)
        if infrastructure is not None:
            return infrastructure, None
        trace = list(context.history)
        images = list(context.info.get("images", ()))
        catalog = []
        for index, path in enumerate(images, 1):
            catalog.append({"ref": f"image_{index}", "kind": "screenshot", "content": str(path)})
        for index, row in enumerate(trace, 1):
            catalog.append({"ref": f"trace_{index}", "kind": "action_observation", "content": row})
        catalog.append({"ref": "terminal", "kind": "terminal_outcome",
                        "content": context.info.get("terminal")})
        payload = {
                "instruction": context.instruction, "trace": list(context.history),
                "initial_observation": context.observation,
                "terminal": context.info.get("terminal"),
                "current_harness": context.info.get("harness"),
                "experience": context.info.get("experience", []),
                "evidence_catalog": catalog,
        }
        for attempt in range(2):
            directory = Path(context.info["agent_directory"])
            if attempt:
                directory = directory / f"retry_{attempt}"
            try:
                data = self.decide(payload, phase="repair", directory=directory,
                                   images=context.info.get("images", ()))
                # ``decide`` validates real CLI output. Re-validate here as
                # well so injected transports and test doubles cannot bypass
                # the bounded harness contract.
                try:
                    data = validate_response(data)
                    validate_evidence_refs(data, catalog)
                except (TypeError, ValueError) as exc:
                    raise InvalidSupervisorResponse(str(exc)) from exc
                break
            except InvalidSupervisorResponse as exc:
                # A malformed proposal is not a simulator failure. Give the
                # supervisor the validator feedback; never weaken admission.
                if attempt == 1:
                    raise
                payload = {**payload, "previous_validation_error": str(exc),
                           "retry_instruction": "Correct the invalid analysis or plan. Cite available evidence, keep guidance within 1200 characters, and return all schema fields."}
            except (subprocess.TimeoutExpired, RuntimeError):
                if attempt == 1:
                    raise
        target = UpdateTarget.HARNESS_UPDATE
        diagnosis = FailureDiagnosis(
            FailureType(data["failure_type"]), tuple(data["evidence"]), float(data["confidence"]),
            data.get("failure_analysis", {}).get("failed_stage",
                "unresolved" if data["failure_type"] == "UNKNOWN" else data["failure_type"].lower()),
            UpdateTarget.NO_UPDATE if data["failure_type"] in {"UNKNOWN", "INFRASTRUCTURE"} else target,
            "codex_execution_trace",
            details={"failure_analysis": data.get("failure_analysis", {}),
                     "retry_plan": data.get("retry_plan", {})},
        )
        if diagnosis.recommended_update is UpdateTarget.NO_UPDATE or diagnosis.confidence < 0.2:
            return diagnosis, None
        proposal = ModificationProposal(
            generation, target, data["reason"],
            data["harness_patch"],
            parent_version, (str(context.info["episode_id"]),), diagnosis.confidence,
        )
        return diagnosis, proposal
