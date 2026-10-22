"""Resolve next native generation exclusively from completed CAS/ledger evidence.

The current HarnessState selects executable descendants.  Completed rejected
rounds can contribute development observations, but never select capabilities.
The campaign ledger remains the authority for release timing; all holdout cases
remain embargoed from both proposers for the entire campaign.
"""
from copy import deepcopy
from pathlib import Path

from dsr.research.meta_improvement import case_identity
from dsr.research.native_experience import release_receipts, read_memory, receipt_record
from dsr.robodojo.native_campaign import library_from_bundle
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.campaign import _manifest


class NativeGenerationContext:
    def __init__(self, *, state, campaign_root, problems, holdout_rounds, initial_memory=None,
                 generation=0):
        self.state = state
        self.campaign_root = Path(campaign_root).resolve()
        self.problems = deepcopy(problems)
        self.holdout = sorted({case_identity(case) for cohort in holdout_rounds
                              for cases in cohort.values() for case in cases})
        self.initial_memory = str(Path(initial_memory).resolve()) if initial_memory else None
        self.initial_memory_sha = file_digest(Path(initial_memory)) if initial_memory else None
        if type(generation) is not int or generation < 0:
            raise ValueError("Native generation must be a nonnegative integer")
        self.generation = generation

    def identity(self):
        return {"kind": "dsr.native-generation-context/v1", "source": file_digest(Path(__file__)),
                "state_root": str(self.state.root), "campaign_root": str(self.campaign_root),
                "initial_problems_sha256": digest(self.problems), "holdout_physical_cases": self.holdout,
                "generation": self.generation,
                "initial_memory": self.initial_memory, "initial_memory_sha256": self.initial_memory_sha}

    def capabilities(self, anchor=None):
        anchor = self.state.resolve() if anchor is None else anchor
        # Reopen the content-addressed CAS record; supplied summaries are not authority.
        if self.state.read(anchor["revision"]) != anchor:
            raise ValueError("Native generation anchor differs from CAS state")
        selected = {problem["id"]: deepcopy(problem["initial_bundle"]) for problem in self.problems}
        if not anchor.get("evidence"):
            return selected
        decisions = [Path(name) for name in anchor["evidence"]
                     if name.endswith("/meta-comparison/decision.json")]
        if len(decisions) != 1:
            raise ValueError("Inherited native generation lacks exactly one committed comparison")
        path = decisions[0]
        decision = read_json(path)
        if (decision.get("final_accepted") is not True
                or anchor.get("decision", {}).get("decision") != "inherit_child"):
            raise ValueError("Rejected native descendant cannot become an inherited capability")
        meta = read_json(Path(anchor["harness"]["root"]) / "meta_policy.json")
        if decision["candidate_policy_sha256"] != digest(meta):
            raise ValueError("Inherited capability belongs to another meta policy")
        for problem in self.problems:
            campaign = path.parent / "candidate" / problem["id"] / "campaign.json"
            if anchor["evidence"].get(str(campaign)) != file_digest(campaign):
                raise ValueError("Inherited descendant campaign is outside committed evidence")
            descendant = read_json(campaign)["descendant"]
            verify_harness(descendant)
            library_from_bundle(descendant)
            for component in descendant["components"].values():
                if any(anchor["evidence"].get(str(Path(descendant["root"]) / name)) != sha
                       for name, sha in component.items()):
                    raise ValueError("Inherited capability bytes are outside committed evidence")
            if descendant["components"]["foundation"] != problem["initial_bundle"]["components"]["foundation"]:
                raise ValueError("Inherited native capability changed the original foundation")
            selected[problem["id"]] = deepcopy(descendant)
        return selected

    def _released(self, index):
        ledger_path = self.campaign_root / "campaign.json"
        if not ledger_path.exists():
            if index != 0:
                raise ValueError("Previous native generation ledger is missing")
            return [], {}
        ledger = read_json(ledger_path)
        completed = ledger["rounds"][:index]
        if len(completed) != index:
            raise ValueError("Development receipts cannot be released before generation completion")
        receipts, sources = [], {}
        expected_parent = ledger["anchor_revision"]
        for generation, record in enumerate(completed):
            if record["index"] != generation or record["parent_revision"] != expected_parent:
                raise ValueError("Native generation order changed")
            directory = self.campaign_root / "rounds" / f"round-{generation + 1:04d}"
            if record["files"] != _manifest(directory):
                raise ValueError("Completed native generation evidence changed")
            result = self.state.read(record["result_revision"])
            if ((result["revision"] != expected_parent and result["previous"] != expected_parent)
                    or record["outcome"] != ("retained" if result["revision"] == expected_parent else "inherited")):
                raise ValueError("Native generation survivor differs from its completed ledger")
            expected_parent = result["revision"]
            for relative, expected in record["files"].items():
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory) or path.is_symlink() or file_digest(path) != expected:
                    raise ValueError("Completed native generation evidence changed")
                # Only development trial receipts are released.  Neither outer
                # holdouts nor initial-protection cases become proposer inputs.
                parts = Path(relative).parts
                if (len(parts) >= 7 and parts[0] == "meta-comparison"
                        and parts[1] in {"parent", "candidate"} and parts[3:5] == ("development", "trials")
                        and parts[-1] == "receipt.json"):
                    receipts.append({"path": str(path), "generation": self.generation + generation})
                    sources[str(path)] = expected
        return receipts, sources

    def _problems(self, selected, memory):
        problems = deepcopy(self.problems)
        for problem in problems:
            initial = selected[problem["id"]]
            problem.update(initial_bundle=initial, start_revision=verify_harness(initial))
            records = {row["id"]: row for row in problem["development_records"]}
            for row in memory["records"]:
                if row["task"] == problem["task"]:
                    records.setdefault(row["id"], deepcopy(row))
            problem["development_records"] = list(records.values())
        return problems

    def _validate_memory(self, memory, receipts, index):
        """Rebuild source records, rather than trusting a rehashed snapshot."""
        initial = read_memory(self.initial_memory, holdout_case_digests=self.holdout) if self.initial_memory else None
        records = deepcopy(initial["records"]) if initial else []
        seen = {row["id"] for row in records}
        added = []
        for item in receipts:
            row = receipt_record(item["path"], generation=item["generation"], holdout_case_digests=self.holdout)
            if row["id"] not in seen:
                records.append(row)
                seen.add(row["id"])
                added.append(row["id"])
        def public_source(row):
            value = deepcopy(row)
            value["provenance"].pop("copied_evidence", None)
            value["provenance"].pop("receipt_relpath", None)
            return value
        if ([public_source(row) for row in memory["records"]] != [public_source(row) for row in records]
                or memory["generation"] != self.generation + index
                or memory["parent_memory_sha256"] != self.initial_memory_sha
                or memory["added_record_ids"] != added):
            raise ValueError("Native generation memory differs from its completed source receipts")

    def for_round(self, parent, output):
        output = Path(output).resolve()
        index = int(output.name.rsplit("-", 1)[1]) - 1
        anchor = read_json(output / "parent.json") if (output / "parent.json").exists() else self.state.resolve()
        if verify_harness(parent) != anchor["freeze_sha256"]:
            raise ValueError("Native generation parent is not the pinned CAS capability")
        selected = self.capabilities(anchor)
        receipts, sources = self._released(index)
        if self.initial_memory:
            if file_digest(Path(self.initial_memory)) != self.initial_memory_sha:
                raise ValueError("Initial released development memory changed")
            read_memory(self.initial_memory, holdout_case_digests=self.holdout)
        memory_path = output / "released-development" / "development-memory.json"
        snapshot = output / "native-generation.json"
        if snapshot.exists():
            saved = read_json(snapshot)
            memory = read_memory(memory_path, holdout_case_digests=self.holdout)
            self._validate_memory(memory, receipts, index)
            if (saved["anchor_revision"] != anchor["revision"] or saved["sources"] != sources
                    or saved["memory_sha256"] != file_digest(memory_path)
                    or saved["capabilities"] != selected
                    or saved["problems"] != self._problems(selected, memory)
                    or saved["generation"] != self.generation + index
                    or saved["holdout_released"] is not False):
                raise ValueError("Native generation context changed during the round")
            return saved["problems"]
        memory = release_receipts(receipts, memory_path.parent, generation=self.generation + index,
                                  holdout_case_digests=self.holdout, parent=self.initial_memory)["memory"]
        self._validate_memory(memory, receipts, index)
        problems = self._problems(selected, memory)
        atomic_json(snapshot, {"schema": "dsr.native-generation/v1", "generation": self.generation + index,
            "anchor_revision": anchor["revision"], "capabilities": selected, "sources": sources,
            "memory_sha256": file_digest(memory_path), "problems": problems,
            "qualification": None, "holdout_released": False})
        return problems
