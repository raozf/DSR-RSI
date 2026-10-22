"""Verify formal agent admission against the native first-attempt outcome."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from dsr.robodojo.stage_evidence import _save_once
from dsr.robodojo.strategy_evidence import validate_strategy
from dsr.robodojo.task_policy_routing import CODE_COMPOSITION, TaskPolicyRouter


def formal_strategy_library(library):
    return bool(library.get('task_policy_routing')) and any(
        entry.get('deployment_status') == 'experimentally_validated'
        for entry in library.get('program_registry', {}).values())


def validate_formal_run(library, keys, args):
    """Validate the formal schedule before creating any policy/simulator."""
    if not formal_strategy_library(library):
        return
    if (getattr(args, 'probe_bindings', None) or args.policy_name != 'DSRRSI'
            or args.agent_strategy != 'physicalrsi' or args.agent_mode != 'frozen'
            or args.attempts != 1 or getattr(args, 'episode_retry', False)
            or getattr(args, 'sequential_seeds', False)):
        raise ValueError('Formal adaptive measurement requires one frozen attempt without episode restart')
    for key in keys:
        TaskPolicyRouter(library, dict(task_name=key.split('/', 1)[1]))


def validate_formal_episode(event, task, seed, library, agent_output, native_detail):
    """Archive actual binding, strategy consultations and native checker parity."""
    report_path = Path(event['report']).resolve(strict=True)
    if not report_path.is_relative_to(Path(agent_output).resolve() / 'episodes'):
        raise ValueError('Formal report is outside this task run')
    report = json.loads(report_path.read_text())
    evidence = report_path.parent / 'evidence'
    bound_path = evidence / 'bound-plan.json'
    bound = json.loads(bound_path.read_text())
    manifest = json.loads((evidence / 'manifest.json').read_text())
    identity = {key: report.get(key) for key in ('task', 'seed', 'layout_id', 'episode_id')}
    if (identity['task'] != task or type(identity['seed']) is not int or identity['seed'] != seed
            or type(identity['layout_id']) is not int or not identity['episode_id']
            or manifest.get('identity') != identity or manifest.get('attempt') != 1):
        raise ValueError('Formal evidence identity or first-attempt mismatch')
    first = report.get('first_attempt') or {}
    for record in (event, report):
        if (any(record.get(key) != value for key, value in identity.items())
                or record.get('physical_rollouts') != 1 or record.get('retry_attempted', False)
                or record.get('protocol') != 'physicalrsi_frozen_episode_start/v1'):
            raise ValueError('Formal result contains a changed episode or restart')
    if (report.get('status') != 'COMPLETED' or first.get('status') != 'COMPLETED'
            or any(first.get(key) != value for key, value in identity.items())
            or type(first.get('success')) is not bool or type(first.get('score')) not in (int, float)
            or not math.isfinite(first['score']) or not 0 <= first['score'] <= 1
            or type(first.get('steps')) is not int or first['steps'] < 1
            or event.get('first_success') is not first['success']
            or event.get('first_score') != first['score']
            or native_detail.get('success') is not first['success']
            or native_detail.get('layout_id') != identity['layout_id']
            or not math.isclose(native_detail.get('score', -1), first['score'], abs_tol=1e-8)
            or any(record.get('physical_steps') != first['steps'] for record in (event, report))):
        raise ValueError('Formal report differs from the native checker outcome')

    router = TaskPolicyRouter(library, dict(task_name=task))
    _, _, expected = router.view(library['composition_definitions'], dict(
        operations={name: {} for name in library['program_registry']}))
    route = report.get('task_policy_route') or {}
    if (any(route.get(key) != value for key, value in expected.items())
            or route.get('status') != 'SELECTED' or route.get('selected_plan') != bound.get('plan')
            or bound.get('attempt') != 1 or bound.get('step') != 0):
        raise ValueError('Formal route differs from frozen admission or first-action plan')
    plan = bound['plan']
    names = plan.get('operations', [])
    if plan.get('composition') not in expected['allowed_compositions']:
        raise ValueError('Formal composition was not admitted')
    operation = None
    settings = None
    if plan['composition'] == CODE_COMPOSITION:
        if not isinstance(names, list) or len(names) != 1 or names[0] not in expected['allowed_operations']:
            raise ValueError('Formal Code Policy was not admitted')
        operation = names[0]
        entry = library['program_registry'][operation]
        source = entry.get('source_name', operation)
        implementation = 'code-policy'
        settings = entry.get('execution_strategy')
        if route.get('selected_candidate') != operation:
            raise ValueError('Formal selected operation changed')
    else:
        if names or route.get('selected_candidate') is not None:
            raise ValueError('Formal VLA cannot attach Code operations')
        source = library['composition_definitions'][plan['composition']]['steps'][-1]
        implementation = library['skills'][source]['implementation']
    if (bound.get('controller', {}).get('name') != source
            or bound['controller'].get('implementation') != implementation
            or route.get('selected_executor', {}).get('name') != source
            or report.get('binding', {}).get('groups') != {'0': source}
            or event.get('binding') != report.get('binding')
            or route.get('selected_stage_composition') is not None):
        raise ValueError('Formal first-action executor differs from the registry')
    strategy = validate_strategy(route, report, source, report_path, settings)
    result = dict(schema='dsr.formal-episode-evidence/v1', identity=identity,
        operation=operation, source=source, composition=plan['composition'],
        native_success=first['success'], native_score=first['score'], physical_steps=first['steps'],
        physical_rollouts=1, retry_attempted=False,
        report=str(report_path), report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        bound_plan_sha256=hashlib.sha256(bound_path.read_bytes()).hexdigest(),
        routing_sha256=expected['routing_sha256'])
    if strategy is not None:
        result['execution_strategy'] = deepcopy(strategy)
    _save_once(evidence / 'formal-admission.json',
               (json.dumps(result, indent=2, sort_keys=True) + '\n').encode())
    return result
