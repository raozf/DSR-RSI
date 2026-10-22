"""Public self-assessment derived from the current frozen native library.

Inventory presence and observed outcomes are separate. This report neither
infers a physical root cause from actions nor grants deployment authority.
Inputs must already pass the source proposer's native receipt admission.
"""
from __future__ import annotations

from PhysicalRSI_core.infra.storage import digest


def assess_development(feedback):
    """Turn admitted development outcomes into falsifiable diagnostic plans.

    Unobserved causes stay unknown. Checker disagreement establishes a
    verification gap, not that perception or control caused it. Each hypothesis
    names the observation or paired experiment needed to distinguish it.
    """
    diagnoses = []
    for row in feedback:
        if (row.get("split") != "development" or type(row.get("success")) is not bool
                or type(row.get("native_success")) is not bool
                or not isinstance(row.get("checks"), dict)
                or any(type(value) is not bool for value in row["checks"].values())):
            raise ValueError("Self-assessment requires admitted development outcomes")
        failed_checks = sorted(name for name, passed in row["checks"].items() if not passed)
        domain, hypotheses = "unknown", []
        if row["native_success"] and failed_checks:
            domain = "verification"
            hypotheses.append({
                "hypothesis": "Official task completion omits a required observed condition.",
                "target": "meta_policy",
                "experiment": "Use the failed registered checks in candidate development; compare resulting descendants on fresh outer cases.",
            })
        elif not row["native_success"]:
            hypotheses = [
                {"hypothesis": "Object grounding or geometry became stale before contact.",
                 "target": "skill",
                 "experiment": "Record fresh before-contact RGB and compare grounded target across stage boundaries."},
                {"hypothesis": "The stage sequence omitted a necessary precondition or postcondition.",
                 "target": "code_policy",
                 "experiment": "Compare expected and observed progress at every stage; test one changed precondition on paired layouts."},
                {"hypothesis": "The object did not follow the gripper through lift or release.",
                 "target": "skill",
                 "experiment": "Track the object before contact, during lift and after release; end-effector motion alone cannot establish object stability."},
            ]
        traces = row.get("action_traces", [])
        stage_traces = row.get("stage_traces", [])
        deviations = [trace["earliest_observed_deviation"] for trace in stage_traces
                      if trace.get("earliest_observed_deviation") is not None]
        # Each physical attempt keeps its first evidenced discrepancy even if
        # the visible event projection is truncated. It locates a symptom; it
        # does not establish which hidden process caused it.
        deviation = deviations[0] if len(deviations) == 1 else None
        if deviation:
            hypotheses.insert(0, {
                "hypothesis": "The recorded stage precondition or postcondition diverged before the terminal outcome.",
                "target": "code_policy",
                "experiment": "Replay paired layouts with a fresh observation and one changed stage guard at the first evidenced discrepancy; retain every baseline success.",
            })
        diagnoses.append({
            "case_id": row["case_id"], "evidence_sha256": [row["receipt_sha256"]],
            "failure_domain": domain, "root_cause_established": False,
            "stage": (deviation.get("stage") or "unknown") if deviation else "unknown",
            "native_success": row["native_success"],
            "failed_checks": failed_checks,
            "action_trace_available": bool(traces),
            "action_trace_truncated": any(trace.get("truncated") for trace in traces),
            "stage_trace_available": bool(stage_traces),
            "stage_trace_truncated": any(trace.get("truncated") for trace in stage_traces),
            "stage_claim_scope": "controller-reported observed discrepancy; physical root cause unestablished",
            "earliest_observed_deviation": deviation,
            "observed_stages": sorted({stage for trace in stage_traces for stage in trace.get("observed_stages", [])}),
            "observation_age_steps": [trace["maximum_observation_age_steps"] for trace in stage_traces
                                      if trace.get("maximum_observation_age_steps") is not None],
            "hypotheses": hypotheses,
            "missing_evidence": ["stage-aligned object observations and expected/observed transitions"],
        })
    return diagnoses


def assess_library(*, library, descriptor, operation, revision, feedback):
    """Expose current policies, skills, tools and model bindings without paths.

    Called after the complete executable/dependency closure has been verified.
    Arbitrary descriptor fields (including hidden metadata and credentials) are
    deliberately not forwarded to the proposal model.
    """
    registry = library.get("program_registry", {})
    if operation not in registry:
        raise ValueError("Self-assessment operation is not registered")
    policies = [{"name": name,
                 "action_type": entry.get("action_type"),
                 "input": entry.get("input"), "output": entry.get("output"),
                 "qualification": None}
                for name, entry in sorted(registry.items())]
    skills = [{"name": name, "implementation": entry.get("implementation"),
               "qualification": None}
              for name, entry in sorted(library.get("skills", {}).items())]
    report = {
        "schema": "dsr.capability-assessment/v1",
        "capability_revision": revision, "current_operation": operation,
        "code_policies": policies, "skills": skills,
        "model_bindings": [row for row in skills
                           if row["implementation"] in {"dm05", "pi05", "pi05-sparse-memory"}],
        "tools": {"model_methods": list(descriptor.get("transport", {}).get("allowed_methods", [])),
                  "perception_services": [service["name"] for service in descriptor.get("services", [])
                                          if isinstance(service, dict) and isinstance(service.get("name"), str)]},
        "memory_sha256": digest(library.get("memory", {})),
        "assessment_scope": "verified source presence; per-case development evidence; no global capability qualification",
        "mutable_targets": ["code_policy", "skill", "memory"],
        "fixed_boundaries": ["learned weights", "observation/action interface", "native task checker",
                             "outer acceptance thresholds", "holdout layouts"],
        "diagnoses": assess_development(feedback),
        "limitations": [
            "A complete action trace does not establish a root cause or object stability.",
            "Unknown failures require additional observation before causal claims.",
            "Infrastructure failures require repair and rerun; they are not capability failures.",
        ],
    }
    return {**report, "assessment_sha256": digest(report)}
