"""Meta-policy ports for the existing SelfHarness and HarnessState lineage.

The mutable mechanism lives in a frozen bundle. The executor, native verifier,
cost limits, final paired selector and CAS commit remain fixed across rounds.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

from dsr.research.meta_improvement import MetaEvaluator, plan_improvement, validate_meta_policy
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness import now, selection
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness


def policy_from_bundle(bundle, check_catalogue):
    verify_harness(bundle)
    if "meta_policy.json" not in bundle["components"]["control"]:
        raise ValueError("Meta policy must be in the frozen control closure")
    return validate_meta_policy(read_json(Path(bundle["root"]) / "meta_policy.json"),
                                check_catalogue=check_catalogue)


def policy_bundle(root, name, policy, *, parent=None, metadata=None):
    root = Path(root).resolve()
    if parent is None:
        root.mkdir(parents=True, exist_ok=False)
        for kind in CLOSURE:
            atomic_json(root / (kind + ".json"), {"kind": kind, "scope": "meta-policy-development"})
    else:
        verify_harness(parent)
        shutil.copytree(parent["root"], root)
    atomic_json(root / "meta_policy.json", policy)
    components = (deepcopy(parent["components"]) if parent is not None else
                  {kind: {kind + ".json": file_digest(root / (kind + ".json"))} for kind in CLOSURE})
    components["control"]["meta_policy.json"] = file_digest(root / "meta_policy.json")
    return {"id": name, "root": str(root), "components": components, **(metadata or {})}


class BundleMetaProposer:
    """Bridge development-only proposals to versioned meta-policy bundles.

    ``proposal`` must expose identity() and propose(parent_policy,
    development_records, check_catalogue, output), as MetaPolicyProposer does.
    Problems must be obtained from trusted development evidence by the caller.
    """
    def __init__(self, *, problems, proposal, check_catalogue, proposal_budget=4, problem_provider=None):
        if type(proposal_budget) is not int or not 1 <= proposal_budget <= 128:
            raise ValueError("Explicit bounded proposal allocation required")
        self.problems, self.proposal = deepcopy(problems), proposal
        self.check_catalogue = tuple(check_catalogue)
        self.proposal_budget = proposal_budget
        self.problem_provider = problem_provider

    def identity(self):
        return {"kind": "dsr.bundle-meta-proposer/v1", "source": file_digest(Path(__file__)),
                "problem_sha256": digest(self.problems), "proposal": self.proposal.identity(),
                "checks": list(self.check_catalogue), "proposal_budget": self.proposal_budget,
                "problem_provider": self.problem_provider.identity() if self.problem_provider else None}

    def develop(self, parent, output):
        policy = policy_from_bundle(parent, self.check_catalogue)
        problems = (self.problem_provider.for_round(parent, output) if self.problem_provider else self.problems)
        records = [row for problem in problems for row in problem["development_records"]]
        if not records or any(row.get("split") != "development" for row in records):
            raise ValueError("Only recorded development experience can drive meta proposals")
        plans = {}
        for problem in problems:
            plans[problem["id"]] = plan_improvement(
                policy, problem["development_records"], proposal_budget=self.proposal_budget,
                cases=problem["development_cases"], check_catalogue=self.check_catalogue)
        path = output / "meta-development.json"
        atomic_json(path, {"parent_policy": policy, "records": records, "plans": plans,
                           "problems_sha256": digest(problems)})
        return {"split": "evolve", "evidence": {str(path): file_digest(path)},
                "records": records, "plans": plans}

    def propose(self, parent, feedback, output):
        parent_policy = policy_from_bundle(parent, self.check_catalogue)
        proposed = self.proposal.propose(parent_policy=parent_policy, development_records=feedback["records"],
            check_catalogue=self.check_catalogue, output=output / "meta-proposal")
        child_policy = validate_meta_policy(proposed["policy"], check_catalogue=self.check_catalogue)
        if child_policy == parent_policy:
            return []
        return [policy_bundle(output / "meta-child", "meta-" + output.name, child_policy, parent=parent,
            metadata={"parent_sha256": verify_harness(parent), "method": "experience_grounded_meta_proposal",
                      "changes": {"control": proposed}, "environment": {"kind": "shadow_campaign"},
                      "evidence": feedback["evidence"], "costs": {"proposals": 1}})]


class BundleMetaEvaluator:
    """Run paired downstream campaigns only after SelfHarness freezes the pool."""
    def __init__(self, *, evaluator: MetaEvaluator, problems, case_factory, problem_provider=None):
        self.evaluator, self.problems, self.case_factory = evaluator, deepcopy(problems), case_factory
        self.problem_provider = problem_provider
        self.revision = digest(self.identity())

    def identity(self):
        return {"kind": "dsr.bundle-meta-evaluator/v1", "source": file_digest(Path(__file__)),
                "meta_evaluator": self.evaluator.identity(), "problems_sha256": digest(self.problems),
                "cases": self.case_factory.identity(),
                "problem_provider": self.problem_provider.identity() if self.problem_provider else None}

    def admit(self, candidate, output):
        policy_from_bundle(candidate, self.evaluator.check_catalogue)
        return {"accepted": True, "freeze_sha256": verify_harness(candidate),
                "evidence": {"meta_policy_closure": True}}

    def validation(self, comparison, output):
        cases = self.case_factory.cases(comparison)
        path = output / "meta-holdout.json"
        atomic_json(path, cases)
        return {"split": "validation", "generated_at": now(), "comparison_sha256": digest(comparison),
                "layouts": {task: [digest(case) for case in rows] for task, rows in cases.items()},
                "admission_evidence": {str(path): file_digest(path)}}

    def evaluate(self, candidate, comparison, cohort, output):
        path = output / "meta-comparison/decision.json"
        parent = read_json(output / "parent.json")["harness"]
        children = read_json(output / "propose.json")["output"]
        if len(children) != 1:
            raise ValueError("Meta lineage supports one independently compared child per round")
        child = children[0]
        parent_policy = policy_from_bundle(parent, self.evaluator.check_catalogue)
        child_policy = policy_from_bundle(child, self.evaluator.check_catalogue)
        problems = (self.problem_provider.for_round(parent, output) if self.problem_provider else self.problems)
        if read_json(output / "meta-development.json")["problems_sha256"] != digest(problems):
            raise ValueError("Meta proposer and evaluator used different development problems")
        if not path.exists():
            self.evaluator.compare(parent_policy, child_policy, problems=problems,
                holdout_cases=read_json(output / "meta-holdout.json"), output=path.parent)
        decision = read_json(path)
        if (decision["identity"] != self.evaluator.identity()
                or decision["parent_policy_sha256"] != digest(parent_policy)
                or decision["candidate_policy_sha256"] != digest(child_policy)
                or read_json(path.parent / "comparison.json")["problems_sha256"] != digest(problems)):
            raise ValueError("Meta comparison changed or belongs to another policy pair")
        label = "parent" if candidate["id"] == parent["id"] else "candidate"
        rows = []
        for row in decision["metrics"][label]["episodes"]:
            evidence = {}
            for name, expected in row["evidence_sha256"].items():
                source = Path(name).resolve()
                if not source.is_relative_to(output.resolve()) or file_digest(source) != expected:
                    raise ValueError("Meta native trajectory changed")
                evidence[str(source.relative_to(output))] = expected
            rows.append({"task": row["task"], "layout_sha256": row["case_sha256"],
                         "state": "completed", "score": float(row["success"]), "success": row["success"],
                         "evidence_sha256": evidence})
        return {"candidate_id": candidate["id"], "kind": "meta_native_evaluation", "native_exit_code": 0,
                "comparison_sha256": digest(comparison), "cohort_sha256": digest(cohort),
                "freeze_sha256": verify_harness(candidate), "evaluator_revision": self.revision,
                "episodes": rows}


class MetaLineageSelector:
    """Recompute outcomes and meta constraints before ordinary paired selection."""
    def __init__(self, *, profile):
        if profile.get("maximum_task_regression") != 0:
            raise ValueError("Meta lineage requires zero task regression")
        self.profile = deepcopy(profile)

    def identity(self):
        return {"kind": "dsr.fixed-meta-selector/v1", "source": file_digest(Path(__file__)),
                "paired_selector": file_digest(Path(selection.__file__)), "profile": self.profile}

    def select(self, comparison, cohort, results, *, evidence_root):
        root = Path(evidence_root).resolve()
        decision = read_json(root / "meta-comparison/decision.json")
        if decision["identity"]["profile"] != self.profile:
            raise ValueError("Meta outer acceptance profile changed")
        for name, expected in decision["evidence"].items():
            source = Path(name).resolve()
            if not source.is_relative_to(root) or file_digest(source) != expected:
                raise ValueError("Meta experiment evidence changed before lineage selection")
        parent_id = comparison["parent_id"]
        child_ids = set(comparison["candidates"]) - {parent_id}
        if len(child_ids) != 1:
            raise ValueError("Exactly one meta child required")
        metrics = {}
        for result in results:
            label = "parent" if result["candidate_id"] == parent_id else "candidate"
            native_rows, false_positive, selected_count = [], 0, 0
            for row in result["episodes"]:
                receipts = []
                for relative, expected in row["evidence_sha256"].items():
                    path = root / relative
                    if file_digest(path) != expected:
                        raise ValueError("Native receipt changed")
                    if path.suffix != ".json":
                        continue  # JSONL, PNG, logs remain hash-verified evidence.
                    item = read_json(path)
                    if isinstance(item, dict) and item.get("schema") == "dsr.meta-native-evaluation/v1":
                        receipts.append((path, item))
                if len(receipts) != 1:
                    raise ValueError("Each meta outcome needs exactly one independent native receipt")
                receipt_path, receipt = receipts[0]
                checks = receipt.get("checks", {})
                if (receipt.get("state") != "completed" or type(receipt.get("native_success")) is not bool
                        or set(checks) != set(self.profile["required_checks"])
                        or any(type(value) is not bool for value in checks.values())):
                    raise ValueError("Native verifier outcome incomplete")
                success = receipt["native_success"] and all(checks.values())
                if (row["success"] != success or row["score"] != float(success)
                        or row["task"] != receipt["task"] or row["layout_sha256"] != receipt["case_sha256"]):
                    raise ValueError("Supplied score differs from independently persisted native outcome")
                campaign = read_json(receipt_path.parents[2] / "campaign.json")
                if digest(campaign["descendant"]) != receipt["descendant_sha256"]:
                    raise ValueError("Native evaluation differs from selected descendant")
                verify_harness(campaign["descendant"])
                selected_count += campaign["selected"]
                false_positive += campaign["selected"] and not success
                native_rows.append({**row, "native_success": receipt["native_success"]})
            costs = {key: 0 for key in self.profile["maximum_cost_ratio"]}
            for path in (root / "meta-comparison" / label).glob("*/campaign.json"):
                report = read_json(path)
                for key in costs:
                    costs[key] += report["cost"][key]
            per_task = {task: sum(row["success"] for row in native_rows if row["task"] == task)
                        / sum(row["task"] == task for row in native_rows) for task in comparison["profile"]["tasks"]}
            metrics[label] = {"per_task": per_task, "cost": costs,
                              "false_positive_rate": false_positive / max(1, selected_count),
                              "cases": {(row["task"], row["layout_sha256"]):
                                        (row["success"], row["native_success"]) for row in native_rows}}
            if any(metrics[label][key] != decision["metrics"][label][key]
                   for key in ("per_task", "cost", "false_positive_rate")):
                raise ValueError("Meta summary disagrees with independent native/campaign receipts")
        parent, child = metrics["parent"], metrics["candidate"]
        tasks = comparison["profile"]["tasks"]
        gain = sum(child["per_task"][task] - parent["per_task"][task] for task in tasks) / len(tasks)
        reasons = []
        if gain < self.profile["minimum_gain"]:
            reasons.append("insufficient_downstream_gain")
        reasons += ["task_regression:" + task for task in sorted(tasks)
                    if parent["per_task"][task] - child["per_task"][task] > self.profile["maximum_task_regression"]]
        if set(parent["cases"]) != set(child["cases"]):
            raise ValueError("Meta case outcomes are not paired")
        reasons += ["case_regression:" + task + ":" + case_sha for task, case_sha in sorted(parent["cases"])
                    if any(before and not after for before, after in
                           zip(parent["cases"][(task, case_sha)], child["cases"][(task, case_sha)]))]
        if child["false_positive_rate"] > self.profile["maximum_false_positive_rate"]:
            reasons.append("false_positive_limit")
        reasons += ["cost_regression:" + key for key, ratio in self.profile["maximum_cost_ratio"].items()
                    if child["cost"][key] > parent["cost"][key] * ratio]
        if decision["accepted"] != (not reasons) or sorted(decision["reasons"]) != sorted(reasons):
            raise ValueError("Meta decision differs from independently recomputed acceptance")
        selected = selection.select_survivor(comparison, cohort, results, evidence_root=root)
        if reasons:
            child_id = next(iter(child_ids))
            selected.update(survivor_id=parent_id, survivor_freeze_sha256=comparison["candidates"][parent_id],
                            decision="retain_parent")
            selected["rejected"][child_id] = {"reason": "fixed_meta_constraints", "details": reasons}
        return selected
