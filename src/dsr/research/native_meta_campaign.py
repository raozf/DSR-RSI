"""Executable native RSI research campaign; never a production deployment.

The spec pins real development reports and disjoint native layouts before any
model call. Model proposals, executable child libraries, simulator receipts,
paired selection and the existing lineage store are connected here. A bounded
research lineage has no authority to qualify a global robot controller.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from dsr.paths import bootstrap
bootstrap()
from dsr.research.library_source_proposer import LibrarySourceProposer
from dsr.research.meta_improvement import (
    MetaPolicyProposer, plan_improvement, case_identity,
)
from dsr.research.meta_lineage import (
    BundleMetaEvaluator, BundleMetaProposer,
    policy_bundle, policy_from_bundle,
)
from dsr.research.native_meta_improver import NativeLibraryImprover
from dsr.research.native_experience import (
    read_memory, receipt_record, release_receipts,
)
from dsr.research.native_generations import NativeGenerationContext
from dsr.research.native_initial_protection import (
    NativeProtectedMetaEvaluator, NativeProtectedMetaSelector, verify_initial_protection,
)
from dsr.robodojo.native_campaign import CHECKS, NativeLibraryVerifier, freeze_library, native_case
from PhysicalRSI_core.infra.language import LanguageModel, ModelConfig
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign


SCOPE = "native_robot_research_lineage_only; production_promotion_prohibited"


def measured_record(path, task):
    """Keep completed first attempts even if a subsequent restart failed."""
    path = Path(path).resolve()
    report = read_json(path)
    first = report.get("first_attempt", {})
    if (report.get("task") != task or first.get("status") != "COMPLETED"
            or type(first.get("success")) is not bool
            or type(first.get("steps")) is not int or first["steps"] < 1
            or type(report.get("seed")) is not int
            or type(report.get("layout_id")) is not int):
        raise ValueError("Development discovery requires a completed physical first attempt")
    case = native_case(task, report["seed"], report["layout_id"], split="development")
    if report.get("layout_sha256", case["layout_sha256"]) != case["layout_sha256"]:
        raise ValueError("Discovery layout fingerprint differs from the installed native layout")
    for key in ("task", "seed", "layout_id"):
        if key in first and first[key] != report[key]:
            raise ValueError("First-attempt physical identity differs from the episode report")
    # Do not infer a perception/control diagnosis from terminal outcome alone.
    return {"id": "episode-" + file_digest(path)[:20], "task": task,
            "split": "development", "success": first["success"],
            "failure_domain": "unknown", "stage": "completed_first_attempt",
            "observation": "Native first-attempt success=" + str(first["success"])
                           + "; physical steps=" + str(first["steps"])
                           + ("; historical layout fingerprint unavailable" if "layout_sha256" not in report else ""),
            "hypothesis": "Inspect measured execution feedback before proposing a controller change."}, (
                task, report["seed"], report["layout_id"])


class PreregisteredCases:
    def __init__(self, rounds):
        self.rounds = deepcopy(rounds)

    def identity(self):
        return {"source": file_digest(Path(__file__)), "rounds_sha256": digest(self.rounds)}

    def cases(self, comparison):
        index = int(comparison["round_id"].rsplit("-", 1)[1]) - 1
        if not 0 <= index < len(self.rounds):
            raise ValueError("Fresh preregistered holdout round unavailable")
        return deepcopy(self.rounds[index])


def _experience_inputs(spec):
    """Load optional prior-generation experience without trusting its labels."""
    optional = {"experience_memory", "generation_receipts", "generation"}
    unknown = set(spec) - ({"schema", "tasks", "model", "verifier", "budget", "profile",
                             "rounds", "meta_policy"} | optional)
    if unknown:
        raise ValueError("Unknown native campaign fields: " + ", ".join(sorted(unknown)))
    generation = spec.get("generation", 0)
    if type(generation) is not int or generation < 0:
        raise ValueError("Generation must be a nonnegative integer")
    records, sources = [], {}
    memory_path = spec.get("experience_memory")
    if memory_path is not None:
        memory = read_memory(memory_path)
        if generation < memory["generation"]:
            raise ValueError("Imported development generation cannot move backwards")
        records.extend(memory["records"])
        root = Path(memory_path).resolve().parent
        sources[str(Path(memory_path).resolve())] = file_digest(Path(memory_path))
        for row in memory["records"]:
            provenance = row["provenance"]
            sources[str(root / provenance["receipt_relpath"])] = provenance["receipt_sha256"]
            for original, relative in provenance["copied_evidence"].items():
                sources[str(root / relative)] = provenance["source_evidence"][original]
    receipts = spec.get("generation_receipts", [])
    if isinstance(receipts, dict):
        if any(not isinstance(names, list) for names in receipts.values()):
            raise ValueError("Generation receipt mapping values must be lists of paths")
        receipts = [name for names in receipts.values() for name in names]
    if not isinstance(receipts, list) or any(not isinstance(name, (str, Path)) for name in receipts):
        raise ValueError("Generation receipts must be a list of paths")
    for name in receipts:
        row = receipt_record(name, generation=generation)
        records.append(row)
        sources[row["provenance"]["source_path"]] = row["provenance"]["receipt_sha256"]
        sources.update(row["provenance"]["source_evidence"])
    return records, generation, sources


def admit_spec(spec):
    """Validate split boundaries and provenance without launching a model/GPU."""
    required = {"schema", "tasks", "model", "verifier", "budget", "profile", "rounds", "meta_policy"}
    if (not isinstance(spec, dict) or not required <= set(spec)
            or spec["schema"] != "dsr.native-meta-campaign/v1"
            or type(spec["rounds"]) is not int or not 1 <= spec["rounds"] <= 100
            or not isinstance(spec["tasks"], list) or len(spec["tasks"]) < 2):
        raise ValueError("Explicit native meta-campaign spec with at least two tasks required")
    if any(not isinstance(spec[field], dict) for field in ("model", "verifier", "budget", "profile", "meta_policy")):
        raise ValueError("Model, verifier, budget, profile and meta policy must be objects")
    budget = spec["budget"]
    if (set(budget) != {"episodes", "proposals", "wall_seconds"}
            or any(type(budget[key]) is not int or budget[key] < 1 for key in ("episodes", "proposals"))
            or type(budget["wall_seconds"]) not in (int, float)
            or not math.isfinite(budget["wall_seconds"]) or budget["wall_seconds"] <= 0):
        raise ValueError("Explicit positive finite native campaign budgets required")
    model_config = ModelConfig(**spec["model"])
    if spec["profile"].get("maximum_task_regression") != 0:
        raise ValueError("Any observed parent-success regression must reject the candidate")
    if set(spec["profile"].get("required_checks", [])) != set(CHECKS):
        raise ValueError("Native success and physical actions are mandatory independent checks")
    accumulated, generation, experience_sources = _experience_inputs(spec)
    tasks, seen, holdout_seen, sources = [], set(), set(), dict(experience_sources)
    rounds = [{} for _ in range(spec["rounds"])]
    for entry in spec["tasks"]:
        if set(entry) != {"task", "library", "operation", "reports", "development", "holdout_rounds"}:
            raise ValueError("Explicit source library, operation, measured reports and layout splits required")
        task = entry["task"]
        if task in seen:
            raise ValueError("Duplicate native task")
        seen.add(task)
        library_path = Path(entry["library"]).resolve()
        library = read_json(library_path)
        if entry["operation"] not in library.get("program_registry", {}):
            raise ValueError("Chosen native operation is unavailable")
        sources[str(library_path)] = file_digest(library_path)
        records, development_seen, record_ids = [], set(), set()
        # Prior-generation records are accepted only after their own memory
        # closure has been verified.  They remain development evidence and are
        # deduplicated by immutable receipt-derived IDs.
        for row in accumulated:
            if row.get("task") == task:
                if row["id"] in record_ids:
                    continue
                records.append(deepcopy(row))
                record_ids.add(row["id"])
        for name in entry["reports"]:
            path = Path(name).resolve()
            row, key = measured_record(path, task)
            if key in development_seen:
                raise ValueError("Duplicate discovery physical case")
            records.append(row)
            development_seen.add(key)
            sources[str(path)] = file_digest(path)
        if not records:
            raise ValueError("Measured discovery evidence is required before proposing changes")
        development = [native_case(task, row["seed"], row["layout_id"], split="development")
                       for row in entry["development"]]
        development_seen.update((task, row["seed"], row["layout_id"]) for row in development)
        plan_improvement(spec["meta_policy"], records, proposal_budget=spec["budget"]["proposals"],
                         cases=development, check_catalogue=CHECKS)
        if spec["meta_policy"]["development_cases_per_task"] * (1 + budget["proposals"]) > budget["episodes"]:
            raise ValueError("Budget cannot cover the parent and every proposed development candidate")
        if len(entry["holdout_rounds"]) != spec["rounds"]:
            raise ValueError("Every research generation needs its own fresh holdout cohort")
        for index, cases in enumerate(entry["holdout_rounds"]):
            if len(cases) < 3 or len({row["seed"] for row in cases}) < 2:
                raise ValueError("Research holdout requires at least three layouts over two seeds")
            rows = []
            for row in cases:
                case = native_case(task, row["seed"], row["layout_id"], split="meta_holdout")
                key = (task, row["seed"], row["layout_id"])
                if key in development_seen or key in holdout_seen:
                    raise ValueError("A holdout layout was exposed in discovery/development or another generation")
                if case_identity(case) in {item["provenance"]["physical_case_sha256"] for item in accumulated}:
                    raise ValueError("A holdout physical case was exposed in released development memory")
                holdout_seen.add(key)
                rows.append(case)
            rounds[index][task] = rows
        tasks.append({"task": task, "library": str(library_path), "operation": entry["operation"],
                      "development_records": records, "development_cases": development})
    if any(row["task"] not in seen for row in accumulated):
        raise ValueError("Released development experience contains a task outside this campaign")
    return {"tasks": tasks, "holdout_rounds": rounds, "sources": sources,
            "model": model_config.public(), "spec_sha256": digest(spec),
            "qualification": None, "promotion_allowed": False, "scope": SCOPE,
            "coverage": {"task_count": len(tasks), "full_benchmark_qualification": False,
                         "split_unit": "native task/seed/layout; seed domains may overlap",
            "discovery": ("historical reports plus verified released development memory"
                          if accumulated else "fixed historical reports"),
            "experience_generation": generation,
            "development_record_count": sum(len(entry["development_records"]) for entry in tasks)}}


def native_round_decisions(result, campaign_root, profile):
    """Expose completed outer gates without turning retention into improvement.

    A no-op proposal produces no comparison and may still complete a retained
    SelfHarness round. Its unmeasured acceptance remains null. Actual native
    decisions must be in the immutable campaign ledger and pass the independent
    initial-baseline reconstruction before appearing in this report.
    """
    campaign_root = Path(campaign_root).resolve()
    rows = []
    for record in result["rounds"]:
        directory = campaign_root / "rounds" / f"round-{record['index'] + 1:04d}"
        relative = "meta-comparison/decision.json"
        path = directory / relative
        summary = {"round_index": record["index"], "outcome": record["outcome"],
                   "mechanism_inherited": record["outcome"] == "inherited",
                   "qualification": None, "promotion_allowed": False}
        if relative not in record["files"]:
            if path.exists() or record["outcome"] != "retained":
                raise ValueError("Native lineage transition lacks its pinned outer decision")
            rows.append({**summary, "status": "no_changed_mechanism",
                         "two_mechanism_accepted": None, "final_accepted": None,
                         "final_decision": "retain_parent", "initial_protection": None})
            continue
        if file_digest(path) != record["files"][relative]:
            raise ValueError("Native report outer decision differs from the completed campaign ledger")
        decision = read_json(path)
        protection, _ = verify_initial_protection(path.parent, decision, profile)
        if record["outcome"] == "inherited" and not decision["final_accepted"]:
            raise ValueError("Native campaign inherited a mechanism rejected by initial protection")
        rows.append({**summary, "status": "evaluated", "decision_report": str(path),
                     "decision_sha256": file_digest(path),
                     "two_mechanism_accepted": decision["accepted"],
                     "final_accepted": decision["final_accepted"],
                     "final_decision": decision["final_decision"],
                     "mechanism_reasons": decision["reasons"],
                     "initial_protection": {**protection, "report": decision["initial_protection"]["report"]}})
    return rows


def run(spec, workspace, *, execute=False):
    admission = admit_spec(spec)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    atomic_json(workspace / "spec.json", spec)
    atomic_json(workspace / "admission.json", admission)
    if not execute:
        return admission
    # This state belongs exclusively to the research workspace. No production
    # library pointer or global current revision is changed by this entry point.
    model = LanguageModel(ModelConfig(**spec["model"]))
    verifier = NativeLibraryVerifier(**spec["verifier"])
    proposer = LibrarySourceProposer(model=model, operation_by_task={
        item["task"]: item["operation"] for item in admission["tasks"]})
    improver = NativeLibraryImprover(verifier=verifier, proposer=proposer)
    for name, expected in admission["sources"].items():
        if file_digest(Path(name)) != expected:
            raise ValueError("Preregistered source evidence changed")
    input_receipts = spec.get("generation_receipts", [])
    if isinstance(input_receipts, dict):
        input_receipts = [name for names in input_receipts.values() for name in names]
    initial_memory = None
    if input_receipts or spec.get("experience_memory"):
        initial_memory = release_receipts(input_receipts, workspace / "initial-development",
            generation=spec.get("generation", 0), parent=spec.get("experience_memory"),
            holdout_case_digests=[case_identity(case) for cohort in admission["holdout_rounds"]
                                  for cases in cohort.values() for case in cases])["path"]
    problems = []
    for entry in admission["tasks"]:
        bundle = freeze_library(entry["library"], workspace / "initial" / entry["task"], name=entry["task"])
        problems.append({"id": entry["task"], "task": entry["task"], "initial_bundle": bundle,
                         "start_revision": verify_harness(bundle),
                         "development_records": entry["development_records"],
                         "development_cases": entry["development_cases"]})
    # Keep an immutable copy for the extra original-baseline gate.  The
    # generation provider may advance each problem's executable starting
    # capability after a survivor is committed; that rolling capability must
    # never silently replace the baseline used for regression protection.
    protected_initial = {problem["id"]: deepcopy(problem["initial_bundle"]) for problem in problems}
    evaluator = NativeProtectedMetaEvaluator(improver=improver, verifier=verifier,
        protected_initial=protected_initial, profile=spec["profile"], budget=spec["budget"],
        check_catalogue=CHECKS)
    state = HarnessState(workspace / "research-state")
    generation_context = NativeGenerationContext(
        state=state, campaign_root=workspace / "campaign", problems=problems,
        holdout_rounds=admission["holdout_rounds"], initial_memory=initial_memory,
        generation=spec.get("generation", 0))
    bundle_proposer = BundleMetaProposer(problems=problems, proposal=MetaPolicyProposer(model),
        check_catalogue=CHECKS, proposal_budget=spec["budget"]["proposals"],
        problem_provider=generation_context)
    bundle_evaluator = BundleMetaEvaluator(evaluator=evaluator, problems=problems,
        case_factory=PreregisteredCases(admission["holdout_rounds"]), problem_provider=generation_context)
    profile = {"tasks": {entry["task"]: {"weight": 1,
                   "episodes": len(admission["holdout_rounds"][0][entry["task"]]),
                   "score_range": [0, 1], "maximum_regression": 0} for entry in admission["tasks"]},
               "minimum_gain": spec["profile"]["minimum_gain"], "tie_tolerance": 1e-10}
    # SelfHarness freezes its profile across rounds, including per-task counts.
    for cases in admission["holdout_rounds"]:
        if any(len(cases[task]) != settings["episodes"] for task, settings in profile["tasks"].items()):
            raise ValueError("Preregistered holdout coverage must remain equal across generations")
    initial = policy_bundle(workspace / "meta-initial", "meta-initial", spec["meta_policy"])
    def build_loop(directory):
        return SelfHarness(directory, state=state, proposer=bundle_proposer, evaluator=bundle_evaluator,
            selector=NativeProtectedMetaSelector(profile=spec["profile"]), profile=profile,
            protocol={"identity": bundle_evaluator.revision, "evaluation_kind": "meta_native_evaluation"},
            scope=SCOPE, max_candidates=1)
    campaign = ImprovementCampaign(workspace / "campaign", state=state,
                                   build_loop=build_loop, max_rounds=spec["rounds"])
    try:
        result = campaign.run(initial)
        round_decisions = native_round_decisions(result, workspace / "campaign", spec["profile"])
    except BaseException as exc:
        atomic_json(workspace / "incomplete.json", {"state": "incomplete", "error": str(exc),
            "qualification": None, "promotion_allowed": False, "scope": SCOPE})
        raise
    inherited = HarnessState(workspace / "research-state").resolve()
    report = {"schema": "dsr.native-meta-report/v1", "campaign": result,
              "inherited_meta_policy": policy_from_bundle(inherited["harness"], CHECKS),
              "inherited_native_capabilities": generation_context.capabilities(inherited),
              "native_round_decisions": round_decisions,
              "final_accepted": round_decisions[-1]["final_accepted"] if round_decisions else None,
              "initial_protection": round_decisions[-1]["initial_protection"] if round_decisions else None,
              "qualification": None, "promotion_allowed": False, "scope": SCOPE,
              "coverage": admission["coverage"]}
    # Release only completed development trial receipts.  The next invocation
    # may pass this path as ``experience_memory``; holdout and initial-protection
    # receipts are intentionally absent from the released memory.
    # The completed campaign ledger is the release authority.  Do not sweep
    # arbitrary receipt files (including interrupted or outer holdout traces)
    # into the next generation's proposer memory.
    receipts, _ = generation_context._released(len(result["rounds"]))
    memory_dir = workspace / "released-development"
    released = release_receipts(receipts, memory_dir,
        generation=generation_context.generation + max(0, len(result["rounds"]) - 1),
        holdout_case_digests=[case_identity(case) for cohort in admission["holdout_rounds"]
                             for cases in cohort.values() for case in cases], parent=initial_memory)
    report["development_memory"] = {"path": released["path"], "sha256": file_digest(Path(released["path"])),
                                     "generation": released["memory"]["generation"],
                                     "added": released["added"], "qualification": None}
    atomic_json(workspace / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Execute real model and native simulation trials")
    args = parser.parse_args()
    result = run(read_json(args.spec), args.workspace, execute=args.execute)
    print(json.dumps({"workspace": str(args.workspace), "scope": result["scope"],
                      "executed": args.execute, "promotion_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
