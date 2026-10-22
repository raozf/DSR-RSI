"""Independently protect initial native capability during meta inheritance.

Both mechanisms can produce descendants worse than the shared starting robot.
This adds a third, unchanged native baseline on the same held-out cases. The
initial baseline never reaches either proposer and never changes their equal
development budgets. Missing or unbound measurements fail closed.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import time

from dsr.research.meta_improvement import MetaEvaluator
from dsr.research.meta_lineage import MetaLineageSelector
from dsr.robodojo.native_campaign import library_from_bundle
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness


DIRECTORY = "initial-protection"


def protection_identity():
    return {"kind": "dsr.native-initial-protection/v1", "source": file_digest(Path(__file__)),
            "rule": "reject_any_initial_native_or_checked_success_lost_by_candidate",
            "score": "native boolean success; no unmeasured continuous score",
            "qualification": None}


def _evidence(files, root):
    if not isinstance(files, dict) or not files:
        raise ValueError("Initial protection requires complete evidence")
    checked = {}
    for name, sha in files.items():
        path = Path(name)
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_digest(path) != sha:
            raise ValueError("Initial protection evidence changed or escaped its experiment")
        checked[str(path)] = sha
    return checked


def _receipt(path, *, directory, task, case, descendant, verifier, checks):
    path = Path(path).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Initial protection native receipt is outside its case")
    receipt = read_json(path)
    expected = {"schema": "dsr.meta-native-evaluation/v1", "state": "completed", "task": task,
                "case_sha256": digest(case), "descendant_sha256": digest(descendant),
                "verifier_sha256": digest(verifier)}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Initial protection native receipt is incomplete or incorrectly bound")
    actual = receipt.get("checks")
    if (type(receipt.get("native_success")) is not bool or not isinstance(actual, dict)
            or set(actual) != set(checks) or any(type(value) is not bool for value in actual.values())):
        raise ValueError("Initial protection requires every native boolean check")
    evidence = _evidence(receipt.get("evidence"), directory)
    evidence[str(path)] = file_digest(path)
    return {"native_success": receipt["native_success"],
            "success": receipt["native_success"] and all(actual.values())}, evidence


def verify_initial_protection(output, decision, profile):
    """Reconstruct the extra gate from fixed manifests and raw native receipts."""
    output = Path(output).resolve()
    directory = output / DIRECTORY
    request_path, report_path = directory / "request.json", directory / "report.json"
    for path in (request_path, report_path, output / "comparison.json"):
        if decision["evidence"].get(str(path)) != file_digest(path):
            raise ValueError("Initial protection request or report is outside frozen evidence")
    request, report = read_json(request_path), read_json(report_path)
    frozen = read_json(output / "comparison.json")
    if (request.get("identity") != protection_identity()
            or decision["identity"].get("initial_protection") != protection_identity()
            or request.get("verifier") != decision["identity"]["verifier"]
            or request.get("required_checks") != profile["required_checks"]
            or digest(request.get("problems")) != frozen["problems_sha256"]
            or digest(request.get("holdout")) != frozen["holdout_sha256"]
            or request.get("comparison_sha256") != file_digest(output / "comparison.json")):
        raise ValueError("Initial protection protocol or starting problem binding changed")
    _evidence(decision["evidence"], output)
    outcomes, reasons, all_evidence = [], [], {}
    candidate_rows = decision["metrics"]["candidate"]["episodes"]
    by_case = {(row["problem"], row["case_sha256"]): row for row in candidate_rows}
    if len(by_case) != len(candidate_rows):
        raise ValueError("Initial protection candidate cases are duplicated")
    expected_keys = set()
    for problem in request["problems"]:
        problem_id, task = problem["id"], problem["task"]
        initial = request["initial_snapshots"][problem_id]
        start_freeze = frozen["initial_bundle_freezes"][problem_id]
        protected = decision["identity"].get("protected_initial_freezes")
        expected_freeze = protected[problem_id] if protected is not None else start_freeze
        if (verify_harness(initial) != expected_freeze
                or verify_harness(problem["initial_bundle"]) != start_freeze
                or problem["start_revision"] != start_freeze
                or initial["components"]["foundation"] != problem["initial_bundle"]["components"]["foundation"]
                or not Path(initial["root"]).resolve().is_relative_to(directory)):
            raise ValueError("Protected initial capability differs from the pinned original baseline")
        library_from_bundle(initial)
        campaign_path = output / "candidate" / problem_id / "campaign.json"
        if decision["evidence"].get(str(campaign_path)) != file_digest(campaign_path):
            raise ValueError("Protected candidate campaign is not pinned")
        campaign = read_json(campaign_path)
        descendant = campaign["descendant"]
        verify_harness(descendant)
        library_from_bundle(descendant)
        if (campaign["start_revision"] != start_freeze
                or descendant["components"]["foundation"] != initial["components"]["foundation"]):
            raise ValueError("Protected candidate changed the fixed starting foundation")
        for index, case in enumerate(request["holdout"][task]):
            key = (problem_id, digest(case))
            expected_keys.add(key)
            if key not in by_case:
                raise ValueError("Initial protection holdout case is absent from candidate evaluation")
            before_dir = directory / problem_id / "holdout" / f"case-{index:04d}"
            before, initial_evidence = _receipt(before_dir / "receipt.json", directory=before_dir,
                task=task, case=case, descendant=initial, verifier=request["verifier"],
                checks=profile["required_checks"])
            after_dir = output / "candidate" / problem_id / "holdout" / f"case-{index:04d}"
            row = by_case[key]
            receipts = []
            for name in row["evidence_sha256"]:
                path = Path(name)
                if path.suffix == ".json" and path.is_relative_to(after_dir):
                    value = read_json(path)
                    if isinstance(value, dict) and value.get("schema") == "dsr.meta-native-evaluation/v1":
                        receipts.append(path)
            if len(receipts) != 1:
                raise ValueError("Protected candidate needs exactly one independent native receipt")
            after, candidate_evidence = _receipt(receipts[0], directory=after_dir, task=task,
                case=case, descendant=descendant, verifier=request["verifier"], checks=profile["required_checks"])
            if any(row.get(name) is not value for name, value in after.items()):
                raise ValueError("Protected candidate summary differs from its native receipt")
            for name, sha in {**initial_evidence, **candidate_evidence}.items():
                if decision["evidence"].get(name) != sha:
                    raise ValueError("Protected native evidence is outside frozen comparison")
            lost = any(before[name] and not after[name] for name in before)
            if lost:
                reasons.append("initial_case_regression:" + task + ":" + digest(case))
            outcomes.append({"problem": problem_id, "task": task, "case_sha256": digest(case),
                             "initial": before, "candidate": after, "regressed": lost})
            all_evidence.update(initial_evidence)
    if set(by_case) != expected_keys:
        raise ValueError("Initial protection did not cover every candidate holdout case")
    expected = {"accepted": not reasons, "reasons": reasons, "outcomes": outcomes,
                "episodes": len(outcomes), "qualification": None}
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError("Initial protection decision differs from independently measured outcomes")
    if decision.get("initial_protection") != {**expected, "report": str(report_path)}:
        raise ValueError("Initial protection comparison summary changed")
    if (decision.get("final_accepted") is not (decision["accepted"] and not reasons)
            or decision.get("final_decision") != ("inherit_candidate" if decision["accepted"] and not reasons else "retain_parent")):
        raise ValueError("Final native inheritance decision bypasses initial protection")
    return expected, {str(request_path): file_digest(request_path), str(report_path): file_digest(report_path), **all_evidence}


class NativeProtectedMetaEvaluator(MetaEvaluator):
    """MetaEvaluator with one extra unchanged-initial trial per held-out case."""
    def __init__(self, *, protected_initial=None, **kwargs):
        self.protected_initial = deepcopy(protected_initial)
        super().__init__(**kwargs)

    def identity(self):
        return {**super().identity(), "initial_protection": protection_identity(),
                "protected_initial_freezes": ({name: verify_harness(bundle)
                    for name, bundle in self.protected_initial.items()}
                    if self.protected_initial is not None else None)}

    def compare(self, parent_policy, candidate_policy, *, problems, holdout_cases, output):
        decision = super().compare(parent_policy, candidate_policy, problems=problems,
                                   holdout_cases=holdout_cases, output=output)
        output = Path(output).resolve()
        directory = output / DIRECTORY
        directory.mkdir()
        snapshots, evidence = {}, {}
        if self.protected_initial is not None and set(self.protected_initial) != {p["id"] for p in problems}:
            raise ValueError("Every problem requires its pinned original baseline")
        for problem in problems:
            initial = (self.protected_initial[problem["id"]] if self.protected_initial is not None
                       else problem["initial_bundle"])
            library_from_bundle(initial)
            snapshot_root = directory / problem["id"] / "initial"
            shutil.copytree(initial["root"], snapshot_root)
            snapshot = {**deepcopy(initial), "root": str(snapshot_root)}
            if verify_harness(snapshot) != verify_harness(initial):
                raise ValueError("Initial baseline changed during snapshot")
            snapshots[problem["id"]] = snapshot
            for component in snapshot["components"].values():
                evidence.update({str(snapshot_root / name): sha for name, sha in component.items()})
        request_path = directory / "request.json"
        atomic_json(request_path, {"identity": protection_identity(), "problems": problems,
            "holdout": holdout_cases, "initial_snapshots": snapshots, "verifier": self.verifier.identity(),
            "required_checks": self.profile["required_checks"],
            "comparison_sha256": file_digest(output / "comparison.json")})
        evidence[str(request_path)] = file_digest(request_path)
        outcomes, reasons = [], []
        started = time.monotonic()
        for problem in problems:
            initial, task = snapshots[problem["id"]], problem["task"]
            for index, case in enumerate(holdout_cases[task]):
                self._stable()
                destination = directory / problem["id"] / "holdout" / f"case-{index:04d}"
                destination.mkdir(parents=True)
                returned = Path(self.verifier.evaluate(descendant=deepcopy(initial), task=task,
                    case=deepcopy(case), required_checks=deepcopy(self.profile["required_checks"]),
                    output=destination)).resolve()
                # Require the standard receipt location so a selector can find
                # every baseline independently, including unsuccessful ones.
                if returned != destination / "receipt.json":
                    raise ValueError("Initial protection requires its standard native receipt path")
                self._stable()
                before, native_evidence = _receipt(returned, directory=destination, task=task,
                    case=case, descendant=initial, verifier=self.verifier.identity(), checks=self.profile["required_checks"])
                evidence.update(native_evidence)
                matches = [row for row in decision["metrics"]["candidate"]["episodes"]
                           if row["problem"] == problem["id"] and row["case_sha256"] == digest(case)]
                if len(matches) != 1:
                    raise ValueError("Initial protection requires exactly paired candidate cases")
                after = {key: matches[0][key] for key in ("success", "native_success")}
                lost = any(before[name] and not after[name] for name in before)
                if lost:
                    reasons.append("initial_case_regression:" + task + ":" + digest(case))
                outcomes.append({"problem": problem["id"], "task": task, "case_sha256": digest(case),
                                 "initial": before, "candidate": after, "regressed": lost})
        report_path = directory / "report.json"
        summary = {"accepted": not reasons, "reasons": reasons, "outcomes": outcomes,
                   "episodes": len(outcomes), "qualification": None}
        atomic_json(report_path, {**summary, "wall_seconds": time.monotonic() - started,
            "cost_scope": "extra fixed outer verification; excluded from equal improvement budgets"})
        evidence[str(report_path)] = file_digest(report_path)
        decision["evidence"].update(evidence)
        decision["initial_protection"] = {**summary, "report": str(report_path)}
        decision["final_accepted"] = decision["accepted"] and not reasons
        decision["final_decision"] = "inherit_candidate" if decision["final_accepted"] else "retain_parent"
        # BundleMetaEvaluator forwards this closure into lineage, including raw
        # action traces and binary observations. Its selector strips these extra
        # baseline receipts only for the existing two-mechanism metric check.
        for label in ("parent", "candidate"):
            for row in decision["metrics"][label]["episodes"]:
                row["evidence_sha256"].update(evidence)
        verify_initial_protection(output, decision, self.profile)
        self._stable()
        atomic_json(output / "decision.json", decision)
        return decision


class NativeProtectedMetaSelector(MetaLineageSelector):
    def identity(self):
        return {**super().identity(), "initial_protection": protection_identity()}

    def select(self, comparison, cohort, results, *, evidence_root):
        root = Path(evidence_root).resolve()
        decision = read_json(root / "meta-comparison/decision.json")
        protection, evidence = verify_initial_protection(root / "meta-comparison", decision, self.profile)
        filtered = deepcopy(results)
        for original, result in zip(results, filtered):
            for before, row in zip(original["episodes"], result["episodes"]):
                for name, sha in evidence.items():
                    if before["evidence_sha256"].get(str(Path(name).relative_to(root))) != sha:
                        raise ValueError("Initial protection evidence is missing from lineage closure")
                row["evidence_sha256"] = {name: sha for name, sha in row["evidence_sha256"].items()
                    if not (root / name).resolve().is_relative_to(root / "meta-comparison" / DIRECTORY)}
        selected = super().select(comparison, cohort, filtered, evidence_root=root)
        if not protection["accepted"]:
            parent = comparison["parent_id"]
            child = next(name for name in comparison["candidates"] if name != parent)
            selected.update(survivor_id=parent, survivor_freeze_sha256=comparison["candidates"][parent],
                            decision="retain_parent")
            selected["rejected"][child] = {"reason": "initial_native_capability_regression",
                                           "details": protection["reasons"]}
        return selected
