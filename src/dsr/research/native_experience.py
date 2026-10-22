"""Release verified native development receipts for later RSI generations.

Only request-bound development trials enter this store.  Holdout measurements
stay outside it, including after a round is completed.  Executable inheritance
continues to belong to SelfHarness/HarnessState, never to this evidence store.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import shutil

from dsr.research.meta_improvement import case_identity
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json

SCHEMA = "dsr.native-development-memory/v1"
RECEIPT_SCHEMA = "dsr.meta-native-evaluation/v1"


def _checked(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or file_digest(path) != expected:
        raise ValueError("Native development evidence missing or changed")
    return path


def _inside(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Native development evidence escaped memory")
    return path


def _record(receipt, request, *, generation, receipt_sha, source_path):
    if (receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("state") != "completed"
            or type(generation) is not int or generation < 0):
        raise ValueError("Completed native receipts and an explicit generation required")
    case = request.get("case", {})
    if case.get("split") != "development" or receipt.get("split") != "development":
        raise ValueError("Only request-bound development receipts may be released; holdout is excluded")
    if (not case.get("task") or case.get("task") != receipt.get("task")
            or type(case.get("seed")) is not int or type(case.get("layout_id")) is not int
            or not case.get("layout_sha256") or digest(case) != receipt.get("case_sha256")
            or request.get("descendant_sha256") != receipt.get("descendant_sha256")
            or digest(request.get("verifier")) != receipt.get("verifier_sha256")):
        raise ValueError("Native development receipt has an incorrect case/capability/verifier binding")
    checks = receipt.get("checks")
    if (type(receipt.get("native_success")) is not bool or not isinstance(checks, dict) or not checks
            or any(type(value) is not bool for value in checks.values())):
        raise ValueError("Native receipt lacks complete boolean outcomes")
    # The attempt ID is independent of generation and where a receipt is copied.
    # Distinct executions on the same layout remain separate physical attempts.
    attempt = digest({"case": case_identity(case), "descendant": receipt["descendant_sha256"],
                      "evidence": sorted(receipt["evidence"].values())})
    return {"id": "native-" + attempt[:24], "task": receipt["task"], "split": "development",
            "success": receipt["native_success"] and all(checks.values()),
            "failure_domain": "unknown", "stage": "native_development_trial",
            "observation": "native_success=" + str(receipt["native_success"])
                           + "; checks=" + json.dumps(checks, sort_keys=True),
            "hypothesis": "Inspect the recorded trajectory before attributing a controller failure.",
            "provenance": {"source_generation": generation, "source_split": "development",
                "physical_case_sha256": case_identity(case), "physical_attempt_sha256": attempt,
                "case": deepcopy(case), "receipt_sha256": receipt_sha, "source_path": str(source_path),
                "descendant_sha256": receipt["descendant_sha256"],
                "verifier_sha256": receipt["verifier_sha256"]}}


def receipt_record(path, *, generation, holdout_case_digests=()):
    """Read a receipt through its hash-bound request and full evidence closure."""
    path = Path(path).resolve()
    receipt = read_json(path)
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("Native receipt lacks trajectory evidence")
    checked, requests = {}, []
    for name, expected in evidence.items():
        source = Path(name)
        source = _checked(source if source.is_absolute() else path.parent / source, expected)
        if not source.is_relative_to(path.parent):
            raise ValueError("Native development evidence is outside its receipt directory")
        checked[str(source)] = expected
        if source.name == "request.json":
            value = read_json(source)
            if isinstance(value, dict) and isinstance(value.get("case"), dict):
                requests.append((source, value))
    if len(requests) != 1:
        raise ValueError("Native receipt needs exactly one hash-bound case request")
    request_path, request = requests[0]
    row = _record(receipt, request, generation=generation, receipt_sha=file_digest(path), source_path=path)
    if row["provenance"]["physical_case_sha256"] in set(holdout_case_digests):
        raise ValueError("A holdout physical case was exposed as development experience")
    row["provenance"]["source_evidence"] = checked
    row["provenance"]["source_request"] = str(request_path)
    return row


def read_memory(path, *, holdout_case_digests=()):
    path = Path(path).resolve()
    data, root = read_json(path), path.parent
    if (data.get("schema") != SCHEMA or type(data.get("generation")) is not int
            or data["generation"] < 0 or not isinstance(data.get("records"), list)):
        raise ValueError("Invalid native development memory")
    seen = set()
    for row in data["records"]:
        provenance = row["provenance"]
        if (type(provenance.get("source_generation")) is not int
                or not 0 <= provenance["source_generation"] <= data["generation"]):
            raise ValueError("Native memory contains a future development generation")
        receipt_path = _checked(_inside(root, provenance["receipt_relpath"]), provenance["receipt_sha256"])
        receipt = read_json(receipt_path)
        copied = provenance["copied_evidence"]
        if set(copied) != set(provenance["source_evidence"]):
            raise ValueError("Native memory changed its receipt evidence closure")
        for source, relative in copied.items():
            _checked(_inside(root, relative), provenance["source_evidence"][source])
        # Original receipt evidence keys may be relative; recover exactly the
        # original normalized mapping without opening the original files.
        expected_sources = {str((Path(provenance["source_path"]).parent / name).resolve()): sha
                            for name, sha in receipt["evidence"].items()}
        if expected_sources != provenance["source_evidence"]:
            raise ValueError("Native memory evidence differs from original receipt")
        source_root = Path(provenance["source_path"]).parent.resolve()
        if (any(not Path(name).is_relative_to(source_root) for name in expected_sources)
                or Path(provenance["source_request"]).name != "request.json"):
            raise ValueError("Native memory original evidence escaped its receipt directory")
        request = read_json(_inside(root, copied[provenance["source_request"]]))
        rebuilt = _record(receipt, request, generation=provenance["source_generation"],
                          receipt_sha=provenance["receipt_sha256"], source_path=provenance["source_path"])
        rebuilt["provenance"].update({name: provenance[name] for name in
            ("source_evidence", "source_request", "copied_evidence", "receipt_relpath")})
        if rebuilt != row:
            raise ValueError("Native memory record differs from its measured receipt")
        if row["id"] in seen:
            raise ValueError("Duplicate native physical attempt")
        if provenance["physical_case_sha256"] in set(holdout_case_digests):
            raise ValueError("Native memory overlaps a holdout physical case")
        seen.add(row["id"])
    if data.get("records_sha256") != digest(data["records"]):
        raise ValueError("Native memory record digest changed")
    return data


def release_receipts(receipts, output, *, generation, holdout_case_digests=(), parent=None):
    """Copy full evidence and merge attempts once, even across generations."""
    output = Path(output).resolve()
    previous = read_memory(parent, holdout_case_digests=holdout_case_digests) if parent else None
    if type(generation) is not int or generation < (previous["generation"] if previous else 0):
        raise ValueError("Development generation cannot move backwards")
    rows = [receipt_record(item["path"] if isinstance(item, dict) else item,
                           generation=item["generation"] if isinstance(item, dict) else generation,
                           holdout_case_digests=holdout_case_digests)
            for item in receipts]
    if any(row["provenance"]["source_generation"] > generation for row in rows):
        raise ValueError("Cannot release evidence from a future generation")
    output.mkdir(parents=True, exist_ok=False)
    records = deepcopy(previous["records"]) if previous else []
    if parent:
        parent_root = Path(parent).resolve().parent
        for row in records:
            provenance = row["provenance"]
            for relative in [provenance["receipt_relpath"], *provenance["copied_evidence"].values()]:
                target = _inside(output, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_inside(parent_root, relative), target)
    seen, added = {row["id"] for row in records}, []
    for row in rows:
        if row["id"] in seen:
            continue
        provenance = row["provenance"]
        receipt_rel = "receipts/" + provenance["receipt_sha256"] + ".json"
        target = _inside(output, receipt_rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(provenance["source_path"], target)
        provenance["receipt_relpath"] = receipt_rel
        provenance["copied_evidence"] = {}
        for source, sha in provenance["source_evidence"].items():
            relative = "evidence/" + sha + Path(source).suffix
            target = _inside(output, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_checked(source, sha), target)
            provenance["copied_evidence"][source] = relative
        records.append(row)
        added.append(row["id"])
        seen.add(row["id"])
    data = {"schema": SCHEMA, "generation": generation, "records": records,
            "records_sha256": digest(records), "added_record_ids": added,
            "parent_memory_sha256": file_digest(Path(parent)) if parent else None,
            "qualification": None, "scope": "released development evidence only"}
    path = output / "development-memory.json"
    atomic_json(path, data)
    read_memory(path, holdout_case_digests=holdout_case_digests)
    return {"path": str(path), "records": deepcopy(records), "added": added, "memory": data}


def records_for_task(memory, task):
    data = read_memory(memory)
    return [deepcopy(row) for row in data["records"] if row["task"] == task]
