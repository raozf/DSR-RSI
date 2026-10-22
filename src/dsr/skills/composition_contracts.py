"""Offline review of two-controller proposals; this module emits no actions.

Only a compatibility record reviewed outside the Agent can authorize a pair.
Descriptions, shared wire types and low historical SR alone are insufficient.
The live PhysicalRSI selector does not import this module yet.
"""
import math


def review_composition(plan, catalog, historical_metrics, pair_contracts):
    def reject(reason):
        return dict(accepted=False, reason=reason, native_success=None, executed=False)

    task = plan.get('task')
    stages = plan.get('stages')
    if not isinstance(stages, list) or len(stages) != 2:
        return reject('exactly_two_stages_required')
    metric = historical_metrics.get(task)
    rate = metric.get('sr') if isinstance(metric, dict) else None
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate < 10:
        return reject('historical_task_sr_not_below_10_percent')
    ids = [s.get('id') for s in stages if isinstance(s, dict)]
    if len(ids) != 2 or any(not isinstance(i, str) or i not in catalog for i in ids):
        return reject('unregistered_controller')
    if ids[0] == ids[1]:
        return reject('duplicate_controller_is_not_complementarity')
    controllers = [catalog[i] for i in ids]
    if tuple(c.get('kind') for c in controllers) not in (
        ('code','code'), ('code','vla'), ('vla','code')):
        return reject('unsupported_controller_combination')
    for selected, contract in zip(stages, controllers):
        if (contract.get('enabled') is not True or task not in contract.get('tasks', ())
                or selected.get('revision') != contract.get('revision')):
            return reject('disabled_task_incompatible_or_revision_mismatch')
        if contract.get('stage_api') != 'dsr.stage/v1':
            return reject('no_phase_resume_interface')
    pair = pair_contracts.get(tuple(ids))
    if not pair or pair.get('reviewed') is not True or task not in pair.get('tasks', ()):
        return reject('no_task_specific_complementarity_review')
    if pair.get('revisions') != [s['revision'] for s in stages]:
        return reject('pair_review_revision_mismatch')
    a, b = controllers
    if a.get('exit_contract') is None or a['exit_contract'] != b.get('entry_contract'):
        return reject('physical_handoff_contract_mismatch')
    if plan.get('handoff') != pair.get('handoff') or not pair.get('handoff'):
        return reject('unreviewed_handoff')
    if pair.get('max_switches') != 1 or pair.get('reset_environment') is not False:
        return reject('reset_or_unbounded_switching')
    costs = [s.get('max_steps') for s in stages]
    overhead = pair.get('handoff_steps')
    limit = pair.get('native_step_limit')
    if any(type(x) is not int or x <= 0 for x in costs + [limit]):
        return reject('invalid_step_budget')
    if type(overhead) is not int or overhead < 0 or sum(costs)+overhead > limit:
        return reject('native_step_budget_exceeded')
    if not pair.get('observable_predicates') or not pair.get('preserved_state'):
        return reject('missing_observable_handoff_evidence')
    if not isinstance(plan.get('rationale'), str) or not plan['rationale'].strip():
        return reject('missing_instance_specific_rationale')
    return dict(accepted=True, reason='proposal_contracts_match; runtime_checks_still_required',
        historical_sr_percent=rate, estimated=metric.get('estimated', False),
        baseline_source=metric.get('source'), max_steps=sum(costs)+overhead,
        executed=False, native_success=None)
