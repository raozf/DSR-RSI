"""Real source-patch evolution on a bounded numerical control experiment.

The proposer is an explicit engineering rule, not an LLM. It edits an executable
PD controller, measures fresh trajectories in a chroot/seccomp worker, and uses
the inherited PhysicalRSI paired selection/lineage/campaign implementation.
This experiment does not qualify DM0.5, RoboDojo, a learned skill or native RSI.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import difflib
import json
import math
from pathlib import Path
import shutil
import statistics
import time

from dsr.paths import physicalrsi_root
import sys

if str(physicalrsi_root()) not in sys.path:
    sys.path.insert(0, str(physicalrsi_root()))

from PhysicalRSI_core.infra import isolation
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness, now, selection
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign

from . import plant

SCOPE = "numerical_1d_control_source_evolution; no robot, DM0.5 or RoboDojo qualification"
PROTOCOL = "dsr.damped-point-mass/v1"


def source(kp: float, kd: float) -> str:
    return (
        "def action(observation):\n"
        f"    kp = {kp!r}\n"
        f"    kd = {kd!r}\n"
        "    error = observation['goal'] - observation['position']\n"
        "    return max(-4.0, min(4.0, kp * error - kd * observation['velocity']))\n"
    )


def validate_source(text: str) -> None:
    """Admit only a small numeric observation-to-force program language."""
    tree = ast.parse(text)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != "action" or tree.body[0].decorator_list
            or [a.arg for a in tree.body[0].args.args] != ["observation"]
            or tree.body[0].args.defaults or tree.body[0].args.kwonlyargs
            or tree.body[0].args.vararg or tree.body[0].args.kwarg):
        raise ValueError("Only action(observation) is admitted")
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign,
               ast.Return, ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UnaryOp, ast.USub,
               ast.Call, ast.Subscript)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("Unsupported controller syntax: " + type(node).__name__)
        if isinstance(node, ast.Name) and node.id not in {"observation", "kp", "kd", "error", "min", "max"}:
            raise ValueError("Undeclared controller name")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                or node.func.id not in {"min", "max"} or node.keywords):
            raise ValueError("Only numeric saturation calls are admitted")
        if isinstance(node, ast.Subscript) and (not isinstance(node.value, ast.Name)
                or node.value.id != "observation" or not isinstance(node.slice, ast.Constant)
                or node.slice.value not in {"position", "velocity", "goal"}):
            raise ValueError("Only declared observations are readable")


def implementation() -> dict:
    paths = [Path(__file__), Path(plant.__file__), Path(isolation.__file__)]
    return {p.name: file_digest(p) for p in paths}


def manifest(root: Path, name: str, **metadata) -> dict:
    components = {k: {k + ".json": file_digest(root / (k + ".json"))} for k in CLOSURE}
    components["control"]["controller.py"] = file_digest(root / "controller.py")
    return dict(id=name, root=str(root), components=components, **metadata)


def case_suite(round_index: int, split: str, count: int) -> list[dict]:
    # No development, validation or report seed is reused across rounds/splits.
    starts = {"development": 1000, "validation": 20000, "report": 90000}
    return [plant.make_case(starts[split] + round_index * 100 + task_i * 20 + i, task)
            for task_i, task in enumerate(("reach", "brake")) for i in range(count)]


class Evaluation:
    def __init__(self, runtime: Path):
        self.runtime = runtime

    def identity(self):
        return {"kind": PROTOCOL, "source": implementation(),
                "runtime": file_digest(self.runtime / "manifest.json")}

    def admit(self, candidate, output):
        try:
            root = Path(candidate["root"])
            validate_source((root / "controller.py").read_text())
            if read_json(root / "dependencies.json") != implementation():
                raise ValueError("Implementation changed")
            accepted, reason = True, "Numeric observation/action source and frozen dependency contract"
        except Exception as exc:
            accepted, reason = False, f"{type(exc).__name__}: {exc}"
        return {"accepted": accepted, "reason": reason, "freeze_sha256": verify_harness(candidate),
                "evidence": {"source": implementation()}}

    def execute(self, candidate, cases, output):
        output = Path(output)
        started = time.monotonic()
        text = (Path(candidate["root"]) / "controller.py").read_text()
        validate_source(text)
        # One isolated worker per finite batch. No network/process capability is
        # available inside the upstream jail, and no fallback is allowed.
        response = isolation.execute(text + plant.RUNNER_SOURCE, {"cases": cases},
            runtime=self.runtime, output=output / "worker", timeout_s=12,
            memory_bytes=256 * 1024 * 1024, output_bytes=8 * 1024 * 1024)
        rollouts = response.get("rollouts")
        if not isinstance(rollouts, list) or len(rollouts) != len(cases):
            raise ValueError("Missing or extra candidate rollout")
        episodes = []
        for i, (case, row) in enumerate(zip(cases, rollouts)):
            if (row["seed"], row["task"]) != (case["seed"], case["task"]):
                raise ValueError("Candidate case identity mismatch")
            verdict = plant.verify_trajectory(case, row["trace"])
            path = output / f"episode-{i:03d}.json"
            atomic_json(path, {"schema": PROTOCOL, "scope": SCOPE,
                "candidate_id": candidate["id"], "freeze_sha256": verify_harness(candidate),
                "case": case, "trace": row["trace"], "verdict": verdict,
                "controller_sha256": digest(text), "implementation": implementation()})
            episodes.append(dict(task=case["task"], layout_sha256=digest(case),
                state="completed", score=verdict["score"], success=verdict["success"],
                evidence_path=str(path), terminal_error=verdict["terminal_error"]))
        atomic_json(output / "cost.json", {"wall_seconds": time.monotonic() - started,
                    "episodes": len(cases), "simulation_steps": len(cases) * plant.STEPS,
                    "api_calls": 0, "gpu_hours": 0, "cpu_worker_limit": 1})
        return episodes

    def validation(self, comparison, output):
        round_index = int(output.name.split('-')[-1]) - 1
        cases = case_suite(round_index, "validation", 3)
        atomic_json(output / "cases.json", cases)
        return {"split": "validation", "comparison_sha256": digest(comparison),
                "generated_at": now(), "layouts": {task: [digest(c) for c in cases if c["task"] == task]
                    for task in ("reach", "brake")},
                "admission_evidence": {"cases.json": file_digest(output / "cases.json")}}

    def evaluate(self, candidate, comparison, cohort, output):
        episodes = self.execute(candidate, read_json(output / "cases.json"), output / "evaluation" / candidate["id"])
        for row in episodes:
            path = Path(row.pop("evidence_path"))
            row["evidence_sha256"] = {str(path.relative_to(output)): file_digest(path)}
        return {"candidate_id": candidate["id"], "kind": "numerical_control_evaluation",
                "native_exit_code": 0, "comparison_sha256": digest(comparison),
                "cohort_sha256": digest(cohort), "freeze_sha256": verify_harness(candidate),
                "evaluator_revision": PROTOCOL, "episodes": episodes}


class Proposal:
    """Declared engineering heuristic, with no access to validation outcomes."""
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def identity(self):
        return {"kind": "measured_tracking_error_pd_source_patch", "source": implementation()}

    def develop(self, parent, output):
        round_index = int(output.name.split('-')[-1]) - 1
        rows = self.evaluator.execute(parent, case_suite(round_index, "development", 2), output / "development")
        return {"split": "evolve", "evidence": {r["evidence_path"]: file_digest(Path(r["evidence_path"])) for r in rows},
                "mean_terminal_error": statistics.mean(r["terminal_error"] for r in rows),
                "mean_score": statistics.mean(r["score"] for r in rows),
                "costs": {"episodes": len(rows), "api_calls": 0}}

    def propose(self, parent, feedback, output):
        previous = read_json(Path(parent["root"]) / "control.json")
        if feedback["mean_terminal_error"] < 0.005:
            return []
        kp = previous["kp"] * 2.0
        kd = 1.5 * math.sqrt(kp)
        root = output / "candidate"
        shutil.copytree(parent["root"], root)
        old = (root / "controller.py").read_text()
        new = source(kp, kd)
        validate_source(new)
        (root / "controller.py").write_text(new)
        atomic_json(root / "control.json", {"kp": kp, "kd": kd})
        patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='parent/controller.py', tofile='candidate/controller.py'))
        (output / "controller.patch").write_text(patch)
        return [manifest(root, "controller_" + output.name.replace('-', '_'),
                parent_sha256=verify_harness(parent), method="error_attributed_pd_source_patch",
                changes={"control": {"from": previous, "to": {"kp": kp, "kd": kd},
                "patch_sha256": digest(patch), "hypothesis": "Higher stiffness with damping reduces residual tracking error"}},
                environment={"kind": "numerical_damped_point_mass", "scope": SCOPE},
                evidence=feedback["evidence"], costs={"proposals": 1, "api_calls": 0})]


class Selector:
    def identity(self):
        return {"source": file_digest(Path(selection.__file__))}

    def select(self, *args, **kwargs):
        return selection.select_survivor(*args, **kwargs)


def run(workspace: str | Path, *, rounds: int = 3) -> dict:
    if type(rounds) is not int or not 1 <= rounds <= 3:
        raise ValueError("Use one to three bounded rounds")
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime = root / "isolation-runtime"
    if not runtime.exists():
        isolation.build_runtime(runtime)
    state = HarnessState(root / "state")
    evaluator = Evaluation(runtime)
    profile = {"tasks": {task: {"weight": 1, "episodes": 3, "score_range": [0, 1],
                "maximum_regression": 0} for task in ("reach", "brake")},
                "minimum_gain": 0.0001, "tie_tolerance": 1e-10}
    def build_loop(directory):
        return SelfHarness(directory, state=state, proposer=Proposal(evaluator), evaluator=evaluator,
                selector=Selector(), scope=SCOPE, profile=profile, max_candidates=1,
                protocol={"identity": PROTOCOL, "evaluation_kind": "numerical_control_evaluation"})
    seed = root / "seed"
    if not seed.exists():
        seed.mkdir()
        for kind in CLOSURE:
            value = {"kind": kind, "scope": SCOPE}
            if kind == "control": value = {"kp": 0.5, "kd": 0.0}
            if kind == "dependencies": value = implementation()
            atomic_json(seed / (kind + ".json"), value)
        (seed / "controller.py").write_text(source(0.5, 0.0))
    initial = manifest(seed, "initial")
    campaign = ImprovementCampaign(root / "campaign", state=state, build_loop=build_loop, max_rounds=rounds)
    result = campaign.run(initial)
    # Report-only unseen seeds never feed candidate proposal or selection.
    report_path = root / "report.json"
    if report_path.exists():
        report = read_json(report_path)
        if report["current_revision"] != result["current"]["revision"]:
            raise ValueError("Report belongs to a different selected version")
        for name, sha in report["report_evidence"].items():
            if file_digest(Path(name)) != sha:
                raise ValueError("Report-only evidence changed")
        return report
    cases = case_suite(0, "report", 8)
    baseline = evaluator.execute(initial, cases, root / "report_trials" / "initial")
    selected = evaluator.execute(result["current"]["harness"], cases, root / "report_trials" / "selected")
    differences = [b["score"] - a["score"] for a, b in zip(baseline, selected)]
    report = {"schema": "dsr.control-evolution-report/v1", "scope": SCOPE, "qualification": None,
              "status": result["status"], "rounds": [{k: r[k] for k in ("index", "parent_revision", "result_revision", "outcome")} for r in result["rounds"]],
              "current_revision": result["current"]["revision"], "current_harness": result["current"]["harness"]["id"],
              "initial_mean_score": statistics.mean(r["score"] for r in baseline),
              "selected_mean_score": statistics.mean(r["score"] for r in selected),
              "initial_successes": sum(r["success"] for r in baseline),
              "selected_successes": sum(r["success"] for r in selected),
              "report_cases": len(cases), "paired_mean_gain": statistics.mean(differences),
              "paired_gain_standard_error": statistics.stdev(differences) / math.sqrt(len(differences)),
              "api_calls": 0, "gpu_hours": 0, "implementation": implementation(),
              "report_evidence": {r["evidence_path"]: file_digest(Path(r["evidence_path"])) for r in baseline + selected},
              "limitations": ["Numerical point-mass simulation only", "Authored PD tuning rule, not an LLM proposer", "No DM0.5 or native RoboDojo result", "Not evidence of research novelty or full PhysicalRSI parity"]}
    atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--rollback', action='store_true')
    args = parser.parse_args()
    if args.rollback:
        state = HarnessState(Path(args.workspace).resolve() / 'state')
        current = state.resolve()
        print(json.dumps(state.rollback(current['revision']), indent=2))
    else:
        print(json.dumps(run(args.workspace, rounds=args.rounds), indent=2))


if __name__ == '__main__':
    main()
