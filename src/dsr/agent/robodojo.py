"""Native DSR-RSI controller with one prompt-guided retry after failure."""

import json
import inspect
import subprocess
from pathlib import Path
import uuid

from dsr.diagnosis import DiagnosisContext, FailureDiagnoser
from dsr.legacy_evolution import CandidateGenerator, EvidenceGate, VersionManager
from dsr.memory import ExperienceLedger, ExperienceRecord
from dsr.schema import CandidateEvaluation, SINGLE_RETRY_PROTOCOL
from dsr.system2 import System2Agent
from dsr.system2.codex_backend import CodexBackend, InvalidSupervisorResponse, write_json
from dsr.robodojo.adapter import compact, instruction_of
from dsr.harness.runtime import merge_native_harness, normalize_native_harness


class DSRRSIRuntime:
    def __init__(
        self, output_dir, *, mode="adapt", codex_home=None, memory_path=None,
        max_agent_calls_per_task=None, backend=None,
    ):
        if mode not in ("adapt", "frozen"):
            raise ValueError("mode must be adapt or frozen")
        self.mode = mode
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ExperienceLedger(self.output_dir / "experience_ledger.json")
        self.ledger.flush()
        self.source_memory = ExperienceLedger(memory_path) if memory_path else None
        self.versions = VersionManager(self.output_dir / "versions")
        self.backend = backend or (
            CodexBackend(self.output_dir / "agent_calls", config_home=codex_home)
            if mode == "adapt" else None
        )
        self.system2 = System2Agent(
            FailureDiagnoser(), CandidateGenerator(), self.ledger, backend=self.backend
        )
        self.gate = EvidenceGate(min_improvement=0.01)
        # Kept as a compatibility field for old manifests. Native adaptation
        # deliberately has no per-task Agent budget: every failed episode is
        # eligible for diagnosis and a harness proposal.
        self.max_agent_calls_per_task = None
        self.agent_calls = {}
        state_path = self.output_dir / "runtime_state.json"
        if state_path.exists():
            self.agent_calls = json.loads(state_path.read_text()).get("agent_calls", {})

    def _save_state(self):
        write_json(self.output_dir / "runtime_state.json", {"agent_calls": self.agent_calls})

    def _related_experience(self, instruction):
        rows = []
        for ledger in (self.source_memory, self.ledger):
            if ledger:
                rows.extend(ledger.retrieve(task=instruction, top_k=4))
        unique = {row.record_id: row for row in rows}
        return [row for row in unique.values() if row.task == instruction][:6]

    def _initial_harness(self, memories):
        if memories:
            try:
                return normalize_native_harness(json.loads(memories[0].recovery))
            except (TypeError, ValueError, AttributeError):
                pass
        return normalize_native_harness({})

    @staticmethod
    def _run(adapter, harness, *, output_dir, retry=False, record_training=False, failure_type=None):
        """Call native adapters with the failure event, keeping test doubles compatible."""
        kwargs = {"output_dir": output_dir, "retry": retry, "record_training": record_training}
        try:
            accepts_context = "failure_type" in inspect.signature(adapter.run_episode).parameters
        except (TypeError, ValueError):
            accepts_context = False
        if accepts_context:
            kwargs["failure_type"] = failure_type
        return adapter.run_episode(harness, **kwargs)

    def run_episode(self, adapter):
        identity = adapter.identity()
        episode_id = f"layout_{identity['layout_id']}_{uuid.uuid4().hex[:8]}"
        directory = self.output_dir / "episodes" / episode_id
        directory.mkdir(parents=True)
        observation = adapter.preview()
        instruction = instruction_of(observation)
        # Adaptive repair memory is read only AFTER a failed first attempt.
        # A frozen deployment keeps its explicitly supplied baseline memory.
        baseline_memories = self._related_experience(instruction) if self.mode == "frozen" else []
        memories = []
        harness = self._initial_harness(baseline_memories)
        validator = getattr(adapter, "validate_harness", None)
        if callable(validator):
            validator(harness)
        write_json(self.output_dir / "progress.json", {**identity, "episode_id": episode_id, "phase": "first"})
        first = self._run(
            adapter, harness, output_dir=directory / "first",
            record_training=self.mode == "adapt",
        )
        attempts = [first]
        report = {
            **identity, "episode_id": episode_id, "mode": self.mode, "status": "RUNNING",
            "first_attempt": first, "initial_harness": harness, "accepted": False,
            "protocol": SINGLE_RETRY_PROTOCOL, "retry_attempted": False,
            "metrics_protocol": "first attempt only; repair replays are excluded from benchmark SR and score",
            "recovery_metrics_protocol": "first success OR one successful same-layout retry",
        }
        write_json(directory / "report.json", report)

        if self.mode == "adapt" and not first["success"]:
            memories = self._related_experience(instruction)
            # Every failed native episode is eligible for one diagnosis. There
            # is intentionally no per-task proposal budget in this protocol.
            self.agent_calls[identity["task"]] = int(self.agent_calls.get(identity["task"], 0)) + 1
            self._save_state()
            context = DiagnosisContext(
                instruction,
                compact(observation),
                info={
                    "episode_id": episode_id,
                    "terminal": {"success": False, "score": first["score"], "layout_id": first["layout_id"]},
                    "harness": harness,
                    "experience": [row.to_dict() for row in memories],
                    "images": first["images"],
                    "agent_directory": str(directory / "diagnosis"),
                },
                history=tuple(first.get("trace_summary", ())),
            )
            write_json(self.output_dir / "progress.json", {**identity, "episode_id": episode_id, "phase": "diagnosis"})
            try:
                diagnosis, proposal = self.system2.reflect(
                    context, generation=len(self.versions.history),
                    parent_version=self.versions.current_version,
                )
            except InvalidSupervisorResponse as exc:
                # The robot's first rollout is valid evidence even when the
                # supervisor's correction is malformed. Reject it explicitly
                # and continue without another first attempt.
                diagnosis, proposal = None, None
                report["rejected_reason"] = f"invalid supervisor response after correction: {exc}"
                report["supervisor_error"] = str(exc)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                # Repair infrastructure cannot erase a valid first result or
                # cause the caller to re-run that first attempt.
                diagnosis, proposal = None, None
                report["recovery_error"] = f"{type(exc).__name__}: {exc}"
                report["recovery_status"] = "INFRASTRUCTURE_FAILED"
            report["diagnosis"] = diagnosis.to_dict() if diagnosis else None
            report["proposal"] = proposal.to_dict() if proposal else None
            if proposal:
                try:
                    candidate_harness = merge_native_harness(harness, proposal.changes)
                    if callable(validator):
                        validator(candidate_harness)
                except (TypeError, ValueError) as exc:
                    report["rejected_reason"] = f"candidate admission failed: {exc}"
                    candidate_harness = None
                if candidate_harness is not None and candidate_harness != harness:
                    write_json(self.output_dir / "progress.json", {**identity, "episode_id": episode_id, "phase": "candidate"})
                    report["retry_attempted"] = True
                    try:
                        candidate = self._run(
                            adapter, candidate_harness, output_dir=directory / "candidate",
                            retry=True, record_training=True,
                            failure_type=diagnosis.failure_type.value,
                        )
                    except Exception as exc:
                        report["recovery_error"] = f"{type(exc).__name__}: {exc}"
                        report["recovery_status"] = "INFRASTRUCTURE_FAILED"
                    else:
                        attempts.append(candidate)
                        same_layout = all(candidate[key] == first[key] for key in ("task", "seed", "layout_id"))
                        before = float(first["success"])
                        decision = self.gate.decide(CandidateEvaluation(
                            episode_id, proposal,
                            before_metrics={"success_rate": before},
                            after_metrics={"success_rate": float(candidate["success"])},
                            regression_metrics={"success_rate": before},
                            metadata={
                                "protocol": SINGLE_RETRY_PROTOCOL,
                                "evidence_strength": "observed recovery; no unchanged control replay",
                                "causal_effect_verified": False,
                                "cross_task_regression_tested": False,
                                "first": first["trace_path"],
                                "candidate": candidate["trace_path"],
                            },
                        ))
                        if not same_layout:
                            decision = decision.__class__(
                                **{**decision.__dict__, "accepted": False, "rejected_reason": "layout mismatch"}
                            )
                        self.versions.promote(decision)
                        self.versions.save_history()
                        report["verification"] = decision.to_dict()
                        report["accepted"] = decision.accepted
                        if decision.accepted:
                            self.ledger.append(ExperienceRecord(
                                task=instruction, scene=identity["task"], goal=instruction,
                                failure_type=diagnosis.failure_type.value,
                                failed_strategy=json.dumps(harness), recovery=json.dumps(candidate_harness),
                                precondition="same native task and saved layout",
                                postcondition="initial attempt fails and sole prompt-guided retry succeeds; causal effect unverified",
                                evidence=[first["trace_path"], candidate["trace_path"]],
                                confidence=diagnosis.confidence, source_episode=episode_id,
                            ))
                            report["verified_training_episode"] = candidate["training_jsonl"]
                elif candidate_harness is not None:
                    report["rejected_reason"] = "agent proposed no harness change"
            else:
                report.setdefault("rejected_reason", "agent returned UNKNOWN/no update")

        retry = attempts[1] if len(attempts) == 2 else None
        report.update(
            status="COMPLETED", attempts=attempts, physical_rollouts=1 + int(report["retry_attempted"]),
            completed_rollouts=len(attempts),
            physical_steps=sum(row["steps"] for row in attempts),
            physical_steps_complete="recovery_error" not in report or not report["retry_attempted"],
            agent_calls=int(self.mode == "adapt" and not first["success"]),
            reused_experience_ids=[row.record_id for row in baseline_memories],
            consulted_repair_experience_ids=[row.record_id for row in memories],
            retry_success=bool(retry["success"] and same_layout) if retry else False if report["retry_attempted"] else None,
            final_success=bool(first["success"] or (retry and same_layout and retry["success"])),
        )
        self._save_state()
        write_json(directory / "report.json", report)
        write_json(self.output_dir / "progress.json", {**identity, "episode_id": episode_id, "phase": "completed"})
        with (self.output_dir / "episode_events.jsonl").open("a") as stream:
            stream.write(json.dumps({
                **identity, "episode_id": episode_id, "report": str(directory / "report.json"),
                "first_success": first["success"], "first_score": first["score"],
                "accepted": report["accepted"], "physical_rollouts": report["physical_rollouts"],
                "completed_rollouts": report["completed_rollouts"],
                "physical_steps": report["physical_steps"],
                "physical_steps_complete": report["physical_steps_complete"],
                "candidate_success": report["retry_success"],
                "retry_success": report["retry_success"], "retry_attempted": report["retry_attempted"],
                "final_success": report["final_success"], "protocol": SINGLE_RETRY_PROTOCOL,
                "agent_calls": report["agent_calls"],
            }) + "\n")
        return report
