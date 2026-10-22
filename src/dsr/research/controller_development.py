"""Task-grounded adapters for the existing proposal and Self-Harness interfaces.

Generated source remains a candidate until the existing isolated execution,
paired evaluation, selection and lineage gates accept it. This module never
imports generated Python, writes the policy registry, or invents rollout scores.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import difflib
import json
from pathlib import Path

from dsr.paths import bootstrap, repository_root

bootstrap()

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json, relative_path
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_baselines.robodojo.primitive_api import for_task
from PhysicalRSI_baselines.robodojo.proposal import CodeProposer
from PhysicalRSI_baselines.robodojo.skill_proposal import validate_policy_source
from PhysicalRSI_baselines.robodojo.workflow import create as create_native_task


class TaskCatalog:
    """Freeze the installed contracts, then project only public design fields."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.paths = {name: self.root / "metadata" / (name + ".json") for name in
                      ("current_task_contracts", "current_policy_inventory", "current_skill_inventory")}
        self.hashes = {name: file_digest(path) for name, path in self.paths.items()}
        self.contracts = read_json(self.paths["current_task_contracts"])
        self.policies = read_json(self.paths["current_policy_inventory"])
        self.skills = read_json(self.paths["current_skill_inventory"])
        self.tasks = {row["native_configuration"]: row for row in self.contracts["tasks"]}
        if len(self.tasks) != len(self.contracts["tasks"]):
            raise ValueError("Duplicate native task contract")

    def identity(self):
        if {name: file_digest(path) for name, path in self.paths.items()} != self.hashes:
            raise ValueError("Installed controller-development inventory changed")
        return {"kind": "dsr.task-grounding/v1", "inventories": deepcopy(self.hashes),
                "implementation": file_digest(Path(__file__))}

    def task(self, task):
        self.identity()
        if task not in self.tasks:
            raise ValueError("Task is absent from the installed RoboDojo contract: " + task)
        row = self.tasks[task]
        # Changes to task/checker/reset source invalidate the development contract.
        for name in ("task_source", "configuration_source"):
            record = row[name]
            path = self.root / relative_path(record["path"])
            if not path.resolve().is_relative_to(self.root) or file_digest(path) != record["sha256"]:
                raise ValueError("Installed task or reset contract changed")
        return row

    def public_context(self, task):
        row = self.task(task)
        analysis = row["engineering_analysis"]
        related = [policy for policy in self.policies["policies"] if policy["task_family"] == row["task_family"]]
        return {
            "task": task, "task_family": row["task_family"], "category": row["category"],
            "robot": self.contracts["default_robot"],
            "permitted_observations": deepcopy(self.contracts["permitted_default_observations"]),
            "action_contract": deepcopy(self.contracts["default_action_contract"]),
            "requirements": {key: analysis[key] for key in (
                "perception_requirements", "geometric_and_manipulation_requirements",
                "temporal_requirements", "observable_failure_hypotheses", "missing_capability", "reuse_limit")},
            "existing_programs": [{key: policy[key] for key in (
                "policy_id", "description", "declared_configurations", "action_representation", "execution_validation")}
                for policy in related],
            "reusable_skills": [{key: skill[key] for key in ("skill_id", "kind", "signature", "description", "validation_status")}
                                for skill in self.skills["skills"] if skill["kind"] in {"manipulation_primitive", "skill_composition"}],
            "restriction": "Skills are retrieval candidates, not automatically bound primitives. Use only methods in the actual supplied primitive API. No evaluator state, object ground truth, reset layouts, or held-out observations are provided.",
        }


def admit_source(source, primitive_api, *, max_bytes=128 * 1024):
    """Reject contract-invalid proposals; this is not a security sandbox."""
    if not isinstance(source, str) or len(source.encode()) > max_bytes:
        raise ValueError("Candidate source exceeds its explicit size budget")
    tree = validate_policy_source(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    policies = [node for node in functions if node.name == "policy"]
    if len(policies) != 1:
        raise ValueError("Exactly one policy(robot, memory) function is required")
    args = policies[0].args
    if ([arg.arg for arg in args.args] != ["robot", "memory"] or args.defaults or args.posonlyargs
            or args.kwonlyargs or args.vararg or args.kwarg):
        raise ValueError("Candidate entrypoint must be policy(robot, memory)")
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) and not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            raise ValueError("Candidate may contain only function definitions and a module docstring")
    for function in functions:
        if function.decorator_list or function.args.defaults or function.args.kw_defaults or function.returns:
            raise ValueError("Candidate functions cannot execute decorators, defaults, or annotations")
        if any(argument.annotation is not None for argument in
               [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]):
            raise ValueError("Candidate argument annotations are not admitted")
    methods = primitive_api.get("methods", {})
    if not isinstance(methods, dict) or not methods:
        raise ValueError("Explicit primitive methods are required")
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda)):
            raise ValueError("Dynamic or class-based capability binding is not admitted")
        if isinstance(node, ast.Name) and node.id in {"getattr", "setattr", "delattr", "type", "object", "breakpoint"}:
            raise ValueError("Reflective capability access is not admitted")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private attributes are not admitted")
        if isinstance(node, ast.Name) and node.id == "robot":
            parent = parents.get(node)
            call = parents.get(parent)
            if not (isinstance(parent, ast.Attribute) and parent.value is node and
                    isinstance(call, ast.Call) and call.func is parent and parent.attr in methods):
                raise ValueError("Robot capability may only call a declared primitive directly")
            called.add(parent.attr)
    if not called:
        raise ValueError("Candidate never uses a declared robot primitive")
    return {"accepted": True, "source_sha256": digest(source), "primitive_api_sha256": digest(primitive_api),
            "called_primitives": sorted(called), "qualification": None,
            "scope": "Static contract admission only; execution requires the existing isolated worker"}


class TaskGroundedCodeProposer:
    """Drop-in CodeProposer wrapper for the existing RoboDojo workflow."""

    def __init__(self, root, proposer=None, *, model=None, max_context_bytes=256 * 1024, max_source_bytes=128 * 1024):
        if (proposer is None) == (model is None):
            raise ValueError("Provide one existing proposer or one configured model")
        if not 1024 <= max_context_bytes <= 2 * 1024 * 1024 or not 256 <= max_source_bytes <= 512 * 1024:
            raise ValueError("Explicit bounded proposal budgets required")
        self.catalog = TaskCatalog(root)
        self.proposer = proposer if proposer is not None else CodeProposer(model=model)
        self.max_context_bytes, self.max_source_bytes = max_context_bytes, max_source_bytes

    def identity(self):
        return {"kind": "dsr.task-grounded-code-proposer/v1", "catalog": self.catalog.identity(),
                "proposer": self.proposer.identity(), "max_context_bytes": self.max_context_bytes,
                "max_source_bytes": self.max_source_bytes}

    def propose(self, *, task, source, memory, feedback, skill_api, output):
        self.catalog.task(task)
        if not feedback or any(row.get("split") != "development" or row.get("task") != task for row in feedback):
            raise ValueError("Only this task's development feedback may reach the proposer")
        # Primitive configuration stays authoritative. The inventory never grants a new call.
        api = deepcopy(skill_api)
        api["dsr_task_grounding"] = self.catalog.public_context(task)
        local_api = for_task(api, task)
        if source.strip():
            admit_source(source, local_api, max_bytes=self.max_source_bytes)
        payload = {"source": source, "memory": memory, "feedback": feedback, "api": local_api}
        if len(json.dumps(payload, allow_nan=False).encode()) > self.max_context_bytes:
            raise ValueError("Development context exceeds its explicit budget")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if any((output / name).exists() for name in ("dsr-grounding.json", "request.json", "controller.py", "response.json")):
            raise FileExistsError("Use a new proposal directory; historical evidence is immutable")
        identity = self.identity()
        atomic_json(output / "dsr-grounding.json", {"identity": identity, "task_contract": api["dsr_task_grounding"],
                    "development_feedback_sha256": digest(feedback), "parent_source_sha256": digest(source)})
        candidate = self.proposer.propose(task=task, source=source, memory=deepcopy(memory),
                                          feedback=deepcopy(feedback), skill_api=api, output=output)
        if self.identity() != identity:
            raise ValueError("Proposer or installed contract changed during generation")
        if set(candidate) != {"source", "memory", "rationale"} or not isinstance(candidate["memory"], dict):
            raise ValueError("Existing code proposer contract requires source, memory and rationale")
        json.dumps(candidate, allow_nan=False)
        admission = admit_source(candidate["source"], local_api, max_bytes=self.max_source_bytes)
        if candidate["source"] == source and candidate["memory"] == memory:
            raise ValueError("Candidate changes neither source nor reusable memory")
        (output / "controller.py").write_text(candidate["source"], encoding="utf-8")
        patch = "".join(difflib.unified_diff(source.splitlines(True), candidate["source"].splitlines(True),
                                             fromfile="parent/controller.py", tofile="candidate/controller.py"))
        (output / "controller.patch").write_text(patch, encoding="utf-8")
        atomic_json(output / "source-admission.json", {**admission, "task": task, "identity": identity,
                    "development_feedback_sha256": digest(feedback), "memory_sha256": digest(candidate["memory"]),
                    "source_file_sha256": file_digest(output / "controller.py"), "diff_sha256": file_digest(output / "controller.patch"),
                    "status": "awaiting_isolated_development_and_paired_validation"})
        return candidate


class ContractBoundSuite:
    """Add source/installed-task admission to an existing ExperimentSuite."""

    def __init__(self, suite, catalog):
        self.suite, self.catalog = suite, catalog

    def identity(self):
        return {"kind": "dsr.contract-bound-suite/v1", "suite": self.suite.identity(), "catalog": self.catalog.identity()}

    def admit(self, candidate):
        frozen = verify_harness(candidate)
        root = Path(candidate["root"])
        skills = read_json(root / "skills.json")
        api = read_json(root / "tools.json")
        choices = read_json(root / "skill_selection.json")["skill_choices"]
        reports = {}
        for task, selected in choices.items():
            self.catalog.task(task)
            name = selected["name"]
            if name not in skills:
                continue  # The existing suite owns learned-policy admission.
            entry = skills[name]
            path = root / relative_path(entry["source"])
            if (not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()
                    or file_digest(path) != entry["revision"] or selected["revision"] != entry["revision"]):
                raise ValueError("Selected generated policy differs from frozen source")
            reports[task] = admit_source(path.read_text(), for_task(api, task))
        result = deepcopy(self.suite.admit(candidate))
        if verify_harness(candidate) != frozen:
            raise ValueError("Candidate changed during suite admission")
        result.setdefault("evidence", {})["dsr_contract_admission"] = reports
        return result

    def development_cases(self, parent):
        return self.suite.development_cases(parent)

    def validation_cases(self, comparison):
        return self.suite.validation_cases(comparison)

    def environment(self, task):
        self.catalog.task(task)
        return self.suite.environment(task)

    def policy(self, candidate, task):
        self.catalog.task(task)
        return self.suite.policy(candidate, task)

    def verifier(self, task):
        self.catalog.task(task)
        return self.suite.verifier(task)


def make_evaluator(*, root, suite, protocol, budgets, scope, system2, quota, device_registry=None, score_measurement=None):
    """Preserve the existing paired evaluator and immutable case protocol.

Callers supply real environment/policy/verifier implementations; no fallback or
synthetic results are provided. Selection and inheritance remain SelfHarness's.
"""
    if not isinstance(protocol, CaseProtocol):
        raise ValueError("Preregistered development, validation and report cases required")
    catalog = TaskCatalog(root)
    for task, cases in protocol.cases("validation").items():
        catalog.task(task)
        if len(cases) < 3 or len({case.get("seed") for case in cases}) < 2:
            raise ValueError("Validation requires at least three cases across two seeds per task")
    return ExperimentEvaluator(suite=PreregisteredSuite(ContractBoundSuite(suite, catalog), protocol),
        budgets=budgets, scope=scope, system2=system2, quota=quota,
        device_registry=device_registry, score_measurement=score_measurement)


def create(*, configuration, workspace, source_directory, language_model, rsi_plan=None, harness=None):
    """Task-package entrypoint using the existing native RSI loop and DSR grounding."""
    task = create_native_task(configuration=configuration, workspace=workspace,
                              source_directory=source_directory, language_model=language_model,
                              rsi_plan=rsi_plan, harness=harness)
    task.code_proposer = TaskGroundedCodeProposer(repository_root(), proposer=task.code_proposer)
    return task


create.configuration_schema = create_native_task.configuration_schema
