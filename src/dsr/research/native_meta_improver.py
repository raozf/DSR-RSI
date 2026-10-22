"""Measured development campaigns for frozen native RoboDojo library bundles.

The proposal port must materialize executable child libraries. The native
verifier owns every outcome; no numeric score may be supplied by the proposer.
This port never promotes a global capability. Its descendant is a shadow
candidate for MetaEvaluator and the existing SelfHarness selection/lineage.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import time

from dsr.research.meta_improvement import plan_improvement
from dsr.robodojo.native_campaign import library_from_bundle
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness


class NativeLibraryImprover:
    """Consume inherited allocation/curriculum/checks in real native trials.

    proposer.propose(parent, task, target, feedback, output) -> frozen manifest.
    Feedback contains only this campaign's recorded development native outcomes.
    The descriptor must point inside output and include actual changed content.
    Task priority is executed by the outer campaign scheduler; this port owns
    one task per invocation and therefore does not claim task-order improvement.
    """
    def __init__(self, *, verifier, proposer, required_checks=("native_task_success", "physical_actions")):
        self.verifier, self.proposer = verifier, proposer
        self.required_checks = tuple(required_checks)
        self._identity = deepcopy(self.identity())

    def identity(self):
        return {"kind": "dsr.native-library-improver/v1", "source": file_digest(Path(__file__)),
                "verifier": self.verifier.identity(), "proposer": self.proposer.identity(),
                "required_checks": list(self.required_checks),
                "scope": "native_development_only; strict paired case retention; no global promotion"}

    def _stable(self):
        if self.identity() != self._identity:
            raise ValueError("Native development implementation changed")

    def _trial(self, candidate, case, checks, output):
        self._stable()
        frozen = verify_harness(candidate)
        receipt_path = Path(self.verifier.evaluate(descendant=deepcopy(candidate), task=case["task"],
            case=deepcopy(case), required_checks=list(checks), output=output)).resolve()
        self._stable()
        if verify_harness(candidate) != frozen or not receipt_path.is_relative_to(output.resolve()):
            raise ValueError("Native development candidate or receipt location changed")
        receipt = read_json(receipt_path)
        expected = {"schema": "dsr.meta-native-evaluation/v1", "state": "completed", "task": case["task"],
                    "case_sha256": digest(case), "descendant_sha256": digest(candidate),
                    "verifier_sha256": digest(self.verifier.identity())}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("Development native receipt is incomplete or incorrectly bound")
        actual = receipt.get("checks")
        if (type(receipt.get("native_success")) is not bool or not isinstance(actual, dict)
                or set(actual) != set(checks) or any(type(value) is not bool for value in actual.values())):
            raise ValueError("Native development requires every independent boolean check")
        if not isinstance(receipt.get("evidence"), dict) or not receipt["evidence"]:
            raise ValueError("Native development trajectory evidence missing")
        evidence = {str(receipt_path): file_digest(receipt_path)}
        for name, expected_sha in receipt["evidence"].items():
            path = Path(name)
            if not path.is_absolute():
                path = output / path
            path = path.resolve()
            if not path.is_relative_to(output.resolve()) or not path.is_file() or file_digest(path) != expected_sha:
                raise ValueError("Native development evidence changed")
            evidence[str(path)] = expected_sha
        return {"task": case["task"], "split": "development", "case": deepcopy(case),
                "success": receipt["native_success"] and all(actual.values()),
                "native_success": receipt["native_success"], "checks": deepcopy(actual),
                "receipt": str(receipt_path), "evidence": evidence}

    def improve(self, *, meta_policy, problem, budget, output):
        self._stable()
        if set(budget) != {"episodes", "proposals", "wall_seconds"}:
            raise ValueError("Explicit native campaign budgets required")
        if (type(budget["episodes"]) is not int or type(budget["proposals"]) is not int
                or min(budget.values()) <= 0):
            raise ValueError("Positive episode/proposal budgets required")
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        initial = deepcopy(problem["initial_bundle"])
        if problem["start_revision"] != verify_harness(initial):
            raise ValueError("Native shadow campaign start revision differs from initial bundle")
        library_from_bundle(initial)
        catalogue = self.verifier.identity().get("checks", [])
        plan = plan_improvement(meta_policy, problem["development_records"],
            proposal_budget=budget["proposals"], cases=problem["development_cases"], check_catalogue=catalogue)
        ids = plan["development_case_ids"].get(problem["task"])
        if not ids:
            raise ValueError("No scoped native development cases selected")
        chosen = [case for name in ids for case in problem["development_cases"] if case["id"] == name]
        if len(chosen) != len(ids) or any(case["task"] != problem["task"] for case in chosen):
            raise ValueError("Native selected cases are duplicated or outside the task")
        checks = sorted(set(self.required_checks) | set(plan["required_checks"].get(problem["task"], [])))
        if not set(checks) <= set(catalogue):
            raise ValueError("Native stability checks must be independently implemented before use")
        required_episodes = len(chosen) * (1 + budget["proposals"])
        if required_episodes > budget["episodes"]:
            raise ValueError("Reserve all parent/candidate development episodes before execution")
        # Clone the unchanged parent so every possible descendant lives inside
        # this shadow workspace, including a fully rejected candidate pool.
        shutil.copytree(initial["root"], output / "parent")
        parent = {**initial, "root": str(output / "parent")}
        verify_harness(parent)
        started = time.monotonic()
        episodes, proposals, evidence = 0, 0, {}
        def run(candidate, label):
            nonlocal episodes
            rows = []
            for index, case in enumerate(chosen):
                if time.monotonic() - started >= budget["wall_seconds"]:
                    raise TimeoutError("Native improvement wall-clock budget exhausted")
                directory = output / "trials" / label / f"case-{index:04d}"
                directory.mkdir(parents=True)
                row = self._trial(candidate, case, checks, directory)
                episodes += 1
                evidence.update(row["evidence"])
                rows.append(row)
            return rows
        baseline = run(parent, "parent")
        best, best_rows = parent, baseline
        decisions = []
        for target, count in plan["proposal_allocation"].items():
            for _ in range(count):
                if time.monotonic() - started >= budget["wall_seconds"]:
                    raise TimeoutError("Native improvement proposal budget exhausted")
                directory = output / "proposals" / f"candidate-{proposals:04d}"
                directory.parent.mkdir(parents=True, exist_ok=True)
                child = self.proposer.propose(parent=deepcopy(parent), task=problem["task"], target=target,
                                             feedback=deepcopy(baseline), output=directory)
                proposals += 1
                self._stable()
                if verify_harness(initial) != problem["start_revision"] or verify_harness(parent) != problem["start_revision"]:
                    raise ValueError("Native proposal changed the initial capability")
                if not Path(child["root"]).resolve().is_relative_to(directory):
                    raise ValueError("Native child library must be materialized in its isolated proposal directory")
                if verify_harness(child) == verify_harness(parent):
                    raise ValueError("Native proposal changed no executable capability")
                if child["components"]["foundation"] != parent["components"]["foundation"]:
                    raise ValueError("Native proposal cannot change the fixed foundation")
                library_from_bundle(child)
                rows = run(child, f"candidate-{proposals:04d}")
                regressed = [index for index, (before, after) in enumerate(zip(baseline, rows))
                             if (before["success"] and not after["success"])
                             or (before["native_success"] and not after["native_success"])]
                best_regressed = [index for index, (before, after) in enumerate(zip(best_rows, rows))
                                  if (before["success"] and not after["success"])
                                  or (before["native_success"] and not after["native_success"])]
                improved = sum(row["success"] for row in rows) > sum(row["success"] for row in best_rows)
                accepted = not regressed and not best_regressed and improved
                decisions.append({"candidate": child, "target": target, "accepted": accepted,
                                  "lost_parent_cases": regressed, "lost_selected_cases": best_regressed,
                                  "native_successes": sum(row["success"] for row in rows)})
                if accepted:
                    best, best_rows = child, rows
        elapsed = time.monotonic() - started
        if elapsed > budget["wall_seconds"]:
            raise TimeoutError("Completed native campaign exceeded its declared wall-clock budget")
        report = output / "selection.json"
        atomic_json(report, {"plan": plan, "baseline": baseline, "decisions": decisions,
                            "selected": best, "qualification": None,
                            "scope": "development selection only; outer holdout and lineage are required"})
        evidence[str(report)] = file_digest(report)
        for name, expected in evidence.items():
            if file_digest(Path(name)) != expected:
                raise ValueError("Development evidence changed during candidate comparison")
        self._stable()
        return {"start_revision": problem["start_revision"], "descendant": best,
                "selected": verify_harness(best) != verify_harness(parent),
                "cost": {"episodes": episodes, "proposals": proposals, "wall_seconds": elapsed},
                "evidence": evidence}
