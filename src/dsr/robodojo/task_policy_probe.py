"""Isolated native measurement of exact-task Code candidates and portfolios.

This is deliberately separate from the production qualification router. A
probe is a measurement, not admission or an agent-visible success prior.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from dsr.robodojo.task_policy_routing import (CODE_COMPOSITION, descriptor_task,
    entry_identity, read_reference)


SCHEMA = 'dsr.exact-task-policy-probe/v1'
PORTFOLIO = '@agent-portfolio'


class TaskProbeRouter:
    def __init__(self, library, deployment):
        if 'task_policy_routing' in library:
            raise ValueError('A probe cannot replace or bypass the formal qualification route')
        manifest = deepcopy(library['task_policy_probe'])
        if manifest.get('schema') != SCHEMA or manifest.get('promotion_allowed') is not False:
            raise ValueError('An isolated, non-promoting probe manifest is required')
        metrics = read_reference(manifest['historical_metrics'])['task_metrics']
        routes = manifest['tasks']
        if not isinstance(routes, dict) or not routes:
            raise ValueError('Probe needs exact task routes')
        sources = {}
        for task, route in routes.items():
            if not isinstance(task, str) or not isinstance(route, dict):
                raise ValueError('Invalid exact probe task')
            rate = metrics.get(task, {}).get('sr')
            if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate < 50:
                raise ValueError('Probe task is not in the historical low-SR set: ' + task)
            candidates = route.get('candidates')
            if not isinstance(candidates, dict) or not candidates:
                raise ValueError('Probe needs at least one candidate for ' + task)
            for operation, candidate in candidates.items():
                key = candidate['source_library']['sha256']
                if key not in sources:
                    sources[key] = read_reference(candidate['source_library'])
                self._validate_candidate(library, task, operation, candidate, sources[key])
                command=library['program_registry'][operation]['configuration']['command']
                descriptor=json.loads(Path(command[command.index('--descriptor')+1]).read_text())
                if descriptor.get('model_config',{}).get('stage_composition') and rate>=10:
                    raise ValueError('Stage composition requires exact historical SR below 10 percent')
                stages = descriptor.get('model_config', {}).get('stage_composition', [])
                if stages != library['program_registry'][operation].get('stage_composition', []):
                    raise ValueError('Runtime descriptor and catalogue stage composition disagree')
                strategy=descriptor.get('model_config',{}).get('execution_strategy')
                if strategy!=library['program_registry'][operation].get('execution_strategy'):
                    raise ValueError('Runtime descriptor and catalogue execution strategy disagree')
                if strategy:
                    from dsr.harness.strategy import PROTOCOL,StrategyLimits
                    if stages or set(strategy)!={'protocol','limits'} or strategy['protocol']!=PROTOCOL:
                        raise ValueError('Distinct frozen execution strategy contract required')
                    StrategyLimits(**strategy['limits'])
                if stages and (not isinstance(stages, list) or len(stages) != 2
                               or any(not isinstance(s, str) or not s for s in stages)
                               or stages[0] == stages[1]):
                    raise ValueError('Stage composition needs two distinct controllers')
        self.manifest = manifest
        self.library = library
        self.task = deployment.get('task_name')
        if self.task not in routes:
            raise ValueError('Trusted deployment task is outside this isolated probe')
        configured = deployment.get('experiment_candidate')
        available = routes[self.task]['candidates']
        self.candidate = configured
        self.portfolio = configured == PORTFOLIO and routes[self.task].get('selection') == 'agent_portfolio'
        if not self.portfolio and self.candidate not in available:
            raise ValueError('Deployment candidate is not registered for the exact task')

    @classmethod
    def _validate_candidate(cls, library, task, operation, candidate, source_library):
        entry = library['program_registry'].get(operation)
        if not isinstance(entry, dict) or entry.get('deployment_status') not in {
                'isolated_unqualified', 'disabled_pending_native_validation'}:
            raise ValueError('Probe may select only an unqualified candidate')
        if entry_identity(entry) != candidate['entry_sha256']:
            raise ValueError('Probe executable identity changed')
        if descriptor_task(entry) != task:
            raise ValueError('Probe descriptor is for a different exact task')
        original = candidate['source_operation']
        source_entry = source_library['program_registry'].get(original)
        if not isinstance(source_entry, dict) or entry_identity(source_entry) != candidate['source_entry_sha256']:
            raise ValueError('Probe source entry changed')
        cls._check_descriptor_rebind(source_entry, entry, task)
        if entry.get('source_name', operation) != source_entry.get('source_name', original):
            raise ValueError('Probe controller source identity changed')
        skill = library['skills'].get(operation)
        if (not isinstance(skill, dict) or skill.get('implementation') != 'code-policy'
                or skill.get('configuration') != entry['configuration']):
            raise ValueError('Probe skill and operation disagree')

    @staticmethod
    def _check_descriptor_rebind(source_entry, entry, task):
        before = source_entry['configuration']['command']
        after = entry['configuration']['command']
        index = before.index('--descriptor') + 1
        old_path, new_path = Path(before[index]), Path(after[index])
        expected = deepcopy(source_entry)
        # Normalize the two wire contract identifiers. Descriptive suffixes in
        # the expansion registry are not protocol names and fail the planner.
        expected['input'] = 'robodojo.observation-batch/v1'
        expected['output'] = 'robodojo.action-chunks/v1'
        if new_path != old_path:
            old_descriptor = json.loads(old_path.read_text())
            new_descriptor = json.loads(new_path.read_text())
            old_descriptor['model_config']['task_name'] = task
            old_descriptor['environment']['RDJ_CONFIG'] = task
            if old_descriptor != new_descriptor:
                raise ValueError('Derived probe descriptor changed beyond the exact task binding')
            expected['configuration']['command'][index] = str(new_path)
            expected['configuration']['files'][str(new_path)] = hashlib.sha256(new_path.read_bytes()).hexdigest()
        if entry_identity(expected) != entry_identity(entry):
            raise ValueError('Derived probe operation changed beyond the descriptor binding')

    def allows_vla(self, name):
        return False

    def view(self, compositions, operation_program):
        allowed = (list(self.manifest['tasks'][self.task]['candidates'])
                   if self.portfolio else [self.candidate])
        operations = (operation_program or {}).get('operations', {})
        composition = compositions.get(CODE_COMPOSITION)
        if (any(name not in operations for name in allowed) or not composition
                or composition.get('steps', [])[-1:] != ['memory-program-library']):
            raise ValueError('Probe candidate is absent from the runtime catalogue')
        program = deepcopy(operation_program)
        program['operations'] = {name: deepcopy(operations[name]) for name in allowed}
        stage_candidates = {}
        strategy_candidates = {}
        for name, entry in program['operations'].items():
            strategy=self.library['program_registry'][name].get('execution_strategy')
            if strategy:
                entry['execution_strategy']=deepcopy(strategy)
                strategy_candidates[name]=deepcopy(strategy)
            stages = self.library['program_registry'][name].get('stage_composition')
            if stages:
                entry['stage_composition'] = dict(controllers=stages, max_handoffs=1,
                    shared_physical_budget=True, reset_environment=False,
                    boundary='three fresh observations, empty measured grippers, stable subgoal')
                stage_candidates[name] = deepcopy(entry['stage_composition'])
        receipt = dict(schema='dsr.task-policy-probe-selection/v1',
                       exact_configuration=self.task, probe_arm=self.manifest['arm'],
                       selection_mode='agent_portfolio' if self.portfolio else 'fixed_candidate',
                       selected_candidate=None if self.portfolio else self.candidate,
                       allowed_operations=allowed, allowed_compositions=[CODE_COMPOSITION],
                       stage_candidates=stage_candidates,
                       strategy_candidates=strategy_candidates,
                       required_stage_operations=sorted(stage_candidates) if self.portfolio else [],
                       promotion_allowed=False, native_success=None)
        return {CODE_COMPOSITION: deepcopy(composition)}, program, receipt
