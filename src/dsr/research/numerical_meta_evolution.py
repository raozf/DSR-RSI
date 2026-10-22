"""Measured meta-improvement on a 1D plant; never a RoboDojo result.

Both improvers receive the same authored controller candidates. The incumbent
checks terminal position alone; an optional stability check changes which code
it accepts. Fresh outer trajectories independently judge resulting controllers.
Default meta proposal is authored. Pass a MetaPolicyProposer to use an actual
configured model, and retain its request/response evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from dsr.research import plant
from dsr.research.control_evolution import manifest, source, validate_source
from dsr.research.meta_improvement import MetaEvaluator, default_meta_policy, plan_improvement
from dsr.research.meta_lineage import (
    BundleMetaEvaluator, BundleMetaProposer, MetaLineageSelector,
    policy_bundle, policy_from_bundle,
)
from PhysicalRSI_core.infra import isolation
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
from PhysicalRSI_core.lineage import HarnessState


SCOPE = "numerical_1d_meta_improvement; no robot, DM0.5 or RoboDojo qualification"
CHECK = "terminal_stability"
POOL = {"reach": [(48.0, 4.0), (12.0, 6.0)], "brake": [(5.0, 1.5), (12.0, 6.0)]}


class AuthoredMechanismProposal:
    """Add missing stability verification, then test its removal as a negative control."""
    def identity(self):
        return {"kind": "authored_numerical_meta_proposal", "source": file_digest(Path(__file__))}

    def propose(self, *, parent_policy, development_records, check_catalogue, output):
        child = deepcopy(parent_policy)
        child["supplemental_checks"] = {} if parent_policy["supplemental_checks"] else {
            task: [CHECK] for task in ("reach", "brake")}
        output.mkdir(parents=True, exist_ok=False)
        result = {"policy": child, "hypothesis": "Test whether terminal stability verification changes descendant reliability.",
                  "evidence_ids": [row["id"] for row in development_records], "mode": "authored_positive_then_negative_control"}
        atomic_json(output / "proposal.json", result)
        return result


class NumericalMetaCases:
    def identity(self):
        return {"kind": "fixed_numerical_meta_cases", "source": file_digest(Path(__file__))}

    def cases(self, comparison):
        index = int(comparison["round_id"].split("-")[-1])
        return {task: [{**plant.make_case(seed, task), "id": task + "-" + str(seed), "split": "meta_holdout"}
                       for seed in range(6000 + 100 * index, 6003 + 100 * index)]
                for task in ("reach", "brake")}


def _identity(runtime):
    return {"kind": "dsr.numerical-meta-evolution/v1", "scope": SCOPE,
            "implementation": {path.name: file_digest(path) for path in
                               (Path(__file__), Path(plant.__file__), Path(isolation.__file__))},
            "runtime_sha256": file_digest(Path(runtime) / "manifest.json")}


def _run(source_text, cases, runtime, output):
    validate_source(source_text)
    response = isolation.execute(source_text + plant.RUNNER_SOURCE, {"cases": cases},
        runtime=runtime, output=output / "worker", timeout_s=12,
        memory_bytes=256 * 1024 * 1024, output_bytes=8 * 1024 * 1024)
    rows = response.get("rollouts")
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError("Isolated numerical worker returned incomplete trajectories")
    reports, evidence = [], {}
    for index, (case, row) in enumerate(zip(cases, rows)):
        if (row.get("task"), row.get("seed")) != (case["task"], case["seed"]):
            raise ValueError("Numerical task/seed binding changed")
        verdict = plant.verify_trajectory(case, row["trace"])
        path = output / f"trajectory-{index:04d}.json"
        atomic_json(path, {"scope": SCOPE, "case": case, "source_sha256": digest(source_text),
                          "trace": row["trace"], "verdict": verdict})
        evidence[str(path)] = file_digest(path)
        reports.append(verdict)
    return reports, evidence


def _bundle(root, name, kp, kd):
    root.mkdir(parents=True)
    for kind in CLOSURE:
        value = {"scope": SCOPE, "kind": kind}
        if kind == "control":
            value = {"kp": kp, "kd": kd}
        atomic_json(root / (kind + ".json"), value)
    (root / "controller.py").write_text(source(kp, kd))
    return manifest(root, name)


class NumericalImprover:
    """Actual candidate rollouts plus inherited development check selection."""
    def __init__(self, runtime):
        self.runtime = Path(runtime)

    def identity(self):
        return {**_identity(self.runtime), "role": "development-improver",
                "controller_pool": {task: [list(values) for values in pool] for task, pool in POOL.items()}}

    def improve(self, *, meta_policy, problem, budget, output):
        started = time.monotonic()
        output = Path(output)
        plan = plan_improvement(meta_policy, problem["development_records"],
            proposal_budget=2, cases=problem["development_cases"], check_catalogue=[CHECK])
        ids = plan["development_case_ids"][problem["task"]]
        cases = [case for name in ids for case in problem["development_cases"] if case["id"] == name]
        if len(cases) * len(POOL[problem["task"]]) > budget["episodes"] or len(POOL[problem["task"]]) > budget["proposals"]:
            raise ValueError("Numerical development pool exceeds fixed equal budget")
        use_stability = CHECK in plan["required_checks"].get(problem["task"], [])
        candidates, evidence = [], {}
        for index, (kp, kd) in enumerate(POOL[problem["task"]]):
            reports, native_evidence = _run(source(kp, kd), cases, self.runtime, output / f"candidate-{index}")
            evidence.update(native_evidence)
            accepted = [row["terminal_error"] < 0.12 and
                        (not use_stability or abs(row["terminal_velocity"]) < 0.2) for row in reports]
            candidates.append({"index": index, "kp": kp, "kd": kd,
                               "development_acceptance": sum(accepted) / len(accepted),
                               "mean_terminal_error": sum(row["terminal_error"] for row in reports) / len(reports),
                               "mean_terminal_speed": sum(abs(row["terminal_velocity"]) for row in reports) / len(reports),
                               "mean_effort": sum(row["effort_integral"] for row in reports) / len(reports)})
        # Deliberately declared incumbent flaw: position-only acceptance and a
        # stable first-in-pool tie break. Meta evaluation can discover this.
        best = max(candidates, key=lambda row: (row["development_acceptance"], -row["index"]))
        descendant = _bundle(output / "descendant", "descendant-" + str(best["index"]), best["kp"], best["kd"])
        report = output / "selection.json"
        atomic_json(report, {"scope": SCOPE, "plan": plan, "candidates": candidates,
                            "selected": best, "check_terminal_velocity": use_stability,
                            "proposal_mode": "fixed_authored_controller_pool", "qualification": None})
        evidence[str(report)] = file_digest(report)
        return {"start_revision": problem["start_revision"], "descendant": descendant,
                "selected": best["development_acceptance"] > 0,
                "cost": {"episodes": len(cases) * len(candidates), "proposals": len(candidates),
                         "wall_seconds": time.monotonic() - started}, "evidence": evidence}


class NumericalVerifier:
    """Pinned task dynamics plus goal and terminal-stability outcomes."""
    def __init__(self, runtime):
        self.runtime = Path(runtime)

    def identity(self):
        return {**_identity(self.runtime), "role": "outer-native-verifier", "checks": [CHECK]}

    def evaluate(self, *, descendant, task, case, required_checks, output):
        if set(required_checks) != {CHECK}:
            raise ValueError("Numerical outer goal and stability contract is immutable")
        verify_harness(descendant)
        source_text = (Path(descendant["root"]) / "controller.py").read_text()
        reports, evidence = _run(source_text, [case], self.runtime, output)
        verdict = reports[0]
        receipt = {"schema": "dsr.meta-native-evaluation/v1", "state": "completed", "task": task,
                   "case_sha256": digest(case), "descendant_sha256": digest(descendant),
                   "verifier_sha256": digest(self.identity()), "native_success": verdict["success"],
                   "checks": {CHECK: abs(verdict["terminal_velocity"]) < 0.2},
                   "evidence": evidence, "qualification": None, "scope": SCOPE}
        atomic_json(output / "receipt.json", receipt)
        return output / "receipt.json"


def run(workspace, *, runtime=None, meta_proposer=None):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    runtime = Path(runtime).resolve() if runtime else workspace / "isolation-runtime"
    if runtime.exists() and runtime.stat().st_dev != workspace.stat().st_dev:
        raise ValueError("Isolation runtime and workspace must share a filesystem for immutable hard links")
    if not runtime.exists():
        isolation.build_runtime(runtime)
    problems, holdout, records, evidence = [], {}, [], {}
    for task in ("reach", "brake"):
        development = [{**plant.make_case(seed, task), "id": task + "-" + str(seed), "split": "development",
                        "failure_domain": "verification"} for seed in (1000, 1001, 1002)]
        # Real trace evidence exposes the missing check before the meta proposer.
        reports, native = _run(source(*POOL[task][0]), development, runtime, workspace / "discovery" / task)
        evidence.update(native)
        local_records = [{"id": task + "-observed-" + str(index), "task": task, "split": "development",
                          "success": report["success"], "failure_domain": "verification", "stage": "terminal_verification",
                          "observation": "Terminal position error=" + str(report["terminal_error"]) +
                              ", terminal speed=" + str(abs(report["terminal_velocity"])) +
                              "; position-only acceptance=" + str(report["terminal_error"] < 0.12),
                          "hypothesis": "Checking terminal speed may prevent accepting controllers still moving through the goal."}
                         for index, report in enumerate(reports)]
        records.extend(local_records)
        initial = _bundle(workspace / "initial" / task, "initial-" + task, 0.5, 0)
        problems.append({"id": task, "task": task, "start_revision": verify_harness(initial),
                         "initial_bundle": initial, "development_records": local_records,
                         "development_cases": development})
        holdout[task] = [{**plant.make_case(seed, task), "id": task + "-" + str(seed), "split": "meta_holdout"}
                         for seed in (5000, 5001, 5002)]
    parent = default_meta_policy()
    parent["proposal_weights"] = {"code_policy": 1, "skill": 0, "memory": 0}
    if meta_proposer is None:
        candidate = deepcopy(parent)
        candidate["supplemental_checks"] = {task: [CHECK] for task in ("reach", "brake")}
        proposal_mode = "authored_mechanism_change_not_autonomous_model_proposal"
    else:
        candidate = meta_proposer.propose(parent_policy=parent, development_records=records,
            check_catalogue=[CHECK], output=workspace / "model-proposal")["policy"]
        proposal_mode = "model_proposed_from_measured_development_evidence"
    atomic_json(workspace / "meta-policies.json", {"parent": parent, "candidate": candidate,
                "proposal_mode": proposal_mode, "discovery_evidence": evidence})
    evaluator = MetaEvaluator(improver=NumericalImprover(runtime), verifier=NumericalVerifier(runtime),
        profile={"minimum_gain": 0.1, "maximum_task_regression": 0,
                 "maximum_false_positive_rate": 0, "maximum_cost_ratio":
                    {"episodes": 1.0, "proposals": 1.0, "wall_seconds": 4.0},
                 "required_checks": [CHECK]},
        budget={"episodes": 6, "proposals": 2, "wall_seconds": 120}, check_catalogue=[CHECK])
    decision = evaluator.compare(parent, candidate, problems=problems, holdout_cases=holdout,
                                  output=workspace / "comparison")
    # The same fixed ports now participate in the real inherited SelfHarness
    # lineage. The second round starts from the committed first-round policy;
    # the authored negative control removes its added check and must be rejected.
    state = HarnessState(workspace / "meta-state")
    bundle_evaluator = BundleMetaEvaluator(evaluator=evaluator, problems=problems, case_factory=NumericalMetaCases())
    selector = MetaLineageSelector(profile=evaluator.profile)
    proposer = BundleMetaProposer(problems=problems, proposal=meta_proposer or AuthoredMechanismProposal(),
                                  check_catalogue=[CHECK], proposal_budget=evaluator.budget["proposals"])
    profile = {"tasks": {task: {"weight": 1, "episodes": 3, "score_range": [0, 1], "maximum_regression": 0}
                         for task in ("reach", "brake")}, "minimum_gain": .1, "tie_tolerance": 1e-10}
    initial_meta = policy_bundle(workspace / "meta-initial", "initial-meta", parent)
    def build_loop(directory):
        return SelfHarness(directory, state=state, proposer=proposer, evaluator=bundle_evaluator,
                           selector=selector, profile=profile,
                           protocol={"identity": bundle_evaluator.revision, "evaluation_kind": "meta_native_evaluation"},
                           scope=SCOPE, max_candidates=1)
    campaign = ImprovementCampaign(workspace / "meta-campaign", state=state, build_loop=build_loop, max_rounds=2)
    lineage = campaign.run(initial_meta)
    # Reinstantiate the store and read the actual inherited bundle as a new
    # process would. No report-selected policy is injected into this generation.
    inherited = HarnessState(workspace / "meta-state").resolve()
    inherited_policy = policy_from_bundle(inherited["harness"], [CHECK])
    next_output = workspace / "next-generation"
    next_output.mkdir()
    next_generation = NumericalImprover(runtime).improve(meta_policy=inherited_policy, problem=problems[0],
                                                        budget=evaluator.budget, output=next_output)
    next_selection = read_json(next_output / "selection.json")
    report = {"schema": "dsr.numerical-meta-report/v1", "scope": SCOPE, "qualification": None,
              "proposal_mode": proposal_mode, "accepted": decision["accepted"], "reasons": decision["reasons"],
              "mean_downstream_gain": decision["mean_downstream_gain"],
              "parent": {key: decision["metrics"]["parent"][key] for key in ("per_task", "false_positive_rate", "cost")},
              "candidate": {key: decision["metrics"]["candidate"][key] for key in ("per_task", "false_positive_rate", "cost")},
              "selected_meta_policy": decision["survivor_policy"], "decision_path": str(workspace / "comparison/decision.json"),
              "lineage": {"current_revision": inherited["revision"], "current_bundle": inherited["harness"]["id"],
                          "rounds": [{key: row[key] for key in ("index", "parent_revision", "result_revision", "outcome")}
                                     for row in lineage["rounds"]],
                          "state_root": str(state.root), "inherited_policy": inherited_policy,
                          "next_generation_selected_controller": next_generation["descendant"],
                          "next_generation_used_stability_check": next_selection["check_terminal_velocity"]},
              "limitations": ["Numerical damped point mass only; not a robot or RoboDojo result",
                              "Same authored controller pool in both development campaigns",
                              "Meta mechanism selection is not an autonomous research breakthrough",
                              "Native RoboDojo shadow driver and statistical generalization remain unvalidated"]}
    atomic_json(workspace / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--runtime")
    args = parser.parse_args()
    print(json.dumps(run(args.workspace, runtime=args.runtime), indent=2))


if __name__ == "__main__":
    main()
