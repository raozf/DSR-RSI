"""Exact-configuration admission before the agent binds one episode executor.

Historical system SR only controls preservation; it is never a controller SR
claim or an agent score prior. Candidate admission recomputes the existing
native qualification assessment from evaluator-owned, content-addressed inputs.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import uuid

from dsr.skills import policy_qualification as qualification


SCHEMA = 'dsr.exact-task-policy-routing/v1'
CODE_COMPOSITION = 'memory-guided-program-library'
PROTECTED_SR_PERCENT = 50
# Explicit user selection constraints, independent of historical success rates.
# Values identify the original PhysicalRSI worker, not an experimental alias.
ORIGINAL_CODE_ONLY = {'swap_T': 'swap-t'}
# Reviewed reuse of the original full controller, not suffix-based matching.
PARENT_TASK_ALIASES = {
    'arrange_largest_number': 'arrange_largest_number_random',
    'pour_liquid_into_cup_random': 'pour_liquid_into_cup',
    'push_T_random': 'push_T',
    'sort_nesting_dolls_by_size_random': 'sort_nesting_dolls_by_size',
}


def file_reference(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_reference(reference):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('A content-addressed artifact reference is required')
    path = Path(reference['path'])
    payload = path.read_bytes()
    if not path.is_absolute() or hashlib.sha256(payload).hexdigest() != reference['sha256']:
        raise ValueError('Routing artifact changed or has a relative path')
    return json.loads(payload)


def entry_identity(entry):
    value = deepcopy(entry)
    settings = value.get('configuration', {})
    if settings.get('runtime') == 'frozen-program-service':
        settings.pop('output', None)  # native verifier relocates only this field
    return qualification.digest(value)


def descriptor_task(entry):
    settings = entry['configuration']
    command = settings['command']
    path = Path(command[command.index('--descriptor') + 1])
    reference = file_reference(path)
    if settings.get('files', {}).get(str(path)) != reference['sha256']:
        raise ValueError('Original descriptor is not frozen in its worker files')
    task = read_reference(reference)['model_config']['task_name']
    if not isinstance(task, str) or not task:
        raise ValueError('Original descriptor has no exact task configuration')
    return task


def build_manifest(parent_library, library, historical_metrics, task_inventory):
    """Create a reviewable fork; no write or modification of the parent library."""
    references = dict(parent=file_reference(parent_library),
                      historical_metrics=file_reference(historical_metrics),
                      task_inventory=file_reference(task_inventory))
    parent = read_reference(references['parent'])
    metrics = read_reference(references['historical_metrics'])['task_metrics']
    tasks = read_reference(references['task_inventory'])['tasks']
    if len(parent['program_registry']) != 22 or len(parent['skills']) != 23:
        raise ValueError('The audited 22 Code + 1 VLA parent is required')
    names = {row['native_configuration'] for row in tasks}
    if len(names) != 54 or len(tasks) != 54:
        raise ValueError('The exact 54-configuration task inventory is required')
    originals = {}
    for operation, entry in parent['program_registry'].items():
        if entry_identity(library['program_registry'].get(operation, {})) != entry_identity(entry):
            raise ValueError('Original Code Policy or skills were changed: ' + operation)
        task = descriptor_task(entry)
        if task not in names or task in originals:
            raise ValueError('Ambiguous or unknown original task binding')
        originals[task] = operation
    for name, entry in parent['skills'].items():
        if entry_identity(library['skills'].get(name, {})) != entry_identity(entry):
            raise ValueError('Original execution skill changed: ' + name)
    vlas = sorted(name for name, entry in parent['skills'].items()
                  if entry.get('implementation') in {'dm05', 'pi05', 'pi05-sparse-memory'})
    routes = {}
    for task in sorted(names):
        metric = metrics.get(task, {})
        rate = metric.get('sr')
        known = type(rate) in (int, float) and math.isfinite(rate) and 0 <= rate <= 100
        protected = task in ORIGINAL_CODE_ONLY or not known or rate >= PROTECTED_SR_PERCENT
        operation = originals.get(task) or originals.get(PARENT_TASK_ALIASES.get(task))
        if task in ORIGINAL_CODE_ONLY and (operation is None or
                parent['program_registry'][operation].get('source_name', operation) != ORIGINAL_CODE_ONLY[task]):
            raise ValueError('Required original PhysicalRSI Code Policy is missing: ' + task)
        routes[task] = dict(preserve_original=protected,
                            original_operations=[operation] if operation else [],
                            vla_skills=[] if protected and operation else vlas,
                            original_code_available=operation is not None)
    candidates = {}
    for name, entry in library['program_registry'].items():
        if name in parent['program_registry']:
            continue
        # Only exact registered specialist candidates are catalogued. Other
        # expansion/random experiments remain outside the admissible set.
        task = entry.get('exact_configuration')
        validated = (entry.get('deployment_status') == 'experimentally_validated'
                     and entry.get('experimental_validation', {}).get('schema') == 'dsr.experimental-validation/v1'
                     and entry.get('experimental_validation', {}).get('status') == 'experimentally_validated')
        isolated = entry.get('deployment_status') == 'isolated_unqualified'
        if ((validated or isolated) and task in names and descriptor_task(entry) == task):
            candidates[name] = dict(task=task, entry_sha256=entry_identity(entry))
            if validated:
                candidates[name]['validation']=deepcopy(entry['experimental_validation'])
    return dict(schema=SCHEMA, references=references,
                protected_sr_percent=PROTECTED_SR_PERCENT,
                parent_executable_sha256=qualification.canonical_executable_library_sha256(parent),
                tasks=routes, candidates=candidates,
                qualification_source='trusted deployment task_policy_qualifications; never observation or agent',
                selection=('agent chooses an exact original or experimentally validated strategy at episode start; '
                           'registered strategy may replan multiple times inside that episode'
                           if any('validation' in c for c in candidates.values()) else
                           'one executor at episode start, fixed until reset'))


class TaskPolicyRouter:
    def __init__(self, library, deployment):
        self.library = library
        self.manifest = deepcopy(library['task_policy_routing'])
        if self.manifest.get('schema') != SCHEMA:
            raise ValueError('Versioned exact task routing manifest required')
        refs = self.manifest['references']
        for ref in refs.values():
            read_reference(ref)
        rebuilt = build_manifest(refs['parent']['path'], library,
                                 refs['historical_metrics']['path'], refs['task_inventory']['path'])
        if rebuilt != self.manifest:
            raise ValueError('Task routing differs from its frozen parent and exact inventory')
        self.task = deployment.get('task_name')
        if self.task not in self.manifest['tasks']:
            raise ValueError('Trusted deployment needs an exact registered task_name')
        self.evidence = deepcopy(deployment.get('task_policy_qualifications', {}))

    def allows_vla(self, name):
        return name in self.manifest['tasks'][self.task]['vla_skills']

    def _candidate_admission(self, operation):
        """A claimed qualified flag, smoke, or stale assessment cannot admit code."""
        try:
            entry = self.library['program_registry'][operation]
            validation = entry.get('experimental_validation', {})
            skill = self.library['skills'].get(operation, {})
            if (entry.get('deployment_status') == 'experimentally_validated'
                    and self.manifest['candidates'].get(operation, {}).get('validation') == validation
                    and validation.get('schema') == 'dsr.experimental-validation/v1'
                    and validation.get('status') == 'experimentally_validated'
                    and validation.get('confirmation') == 'project_owner_confirmed'
                    and entry.get('exact_configuration') == self.task
                    and descriptor_task(entry) == self.task
                    and entry.get('execution_strategy')
                    and skill.get('implementation') == 'code-policy'
                    and skill.get('configuration') == entry.get('configuration')
                    and skill.get('experimental_validation') == validation):
                command=entry['configuration']['command']
                descriptor=json.loads(Path(command[command.index('--descriptor')+1]).read_text())
                settings=entry['execution_strategy']
                from dsr.harness.strategy import PROTOCOL, StrategyLimits
                if (descriptor['model_config'].get('execution_strategy') != settings
                        or set(settings) != {'protocol', 'limits'} or settings['protocol'] != PROTOCOL
                        or entry.get('stage_composition') or descriptor['model_config'].get('stage_composition')):
                    raise ValueError('Frozen descriptor and registered execution strategy disagree')
                StrategyLimits(**settings['limits'])
                return True, 'experimentally_validated_strategy'
            reference = self.evidence.get(operation)
            if reference is None:
                return False, 'native_qualification_missing'
            inputs = read_reference(reference)
            if inputs.get('schema') != 'dsr.task-route-qualification-inputs/v1':
                raise ValueError('Native assessment input schema required')
            if (inputs['task'] != self.task or inputs['operation'] != operation
                    or inputs['entry_sha256'] != entry_identity(self.library['program_registry'][operation])):
                raise ValueError('Native evidence has a different task or executable identity')
            contract = read_reference(inputs['contract'])
            evaluated = read_reference(inputs['evaluated_library'])
            # Isolation evaluates the same executable bundle before routing can
            # admit it. Routing metadata is the only transferable difference.
            current_executable = deepcopy(self.library)
            evaluated_executable = deepcopy(evaluated)
            current_executable.pop('task_policy_routing', None)
            evaluated_executable.pop('task_policy_routing', None)
            if (contract['parent_library_sha256'] != self.manifest['parent_executable_sha256']
                    or contract['candidate_library_sha256'] !=
                    qualification.canonical_executable_library_sha256(evaluated)
                    or qualification.canonical_executable_library_sha256(current_executable) !=
                    qualification.canonical_executable_library_sha256(evaluated_executable)):
                raise ValueError('Qualification was not run on this parent and executable bundle')
            paired_rows = read_reference(inputs['paired_rows'])
            task_rows = [row for row in paired_rows if row['case'].split('/')[0] == self.task]
            if not task_rows:
                raise ValueError('No native paired cases for this exact task')
            selected = self.library['program_registry'][operation].get('source_name', operation)
            for row in task_rows:
                report_ref = row['candidate']
                report = read_reference(report_ref)
                bound_ref = inputs['bound_plans'][report_ref['sha256']]
                if Path(bound_ref['path']).resolve() != Path(report_ref['path']).resolve().parent / 'evidence/bound-plan.json':
                    raise ValueError('Bound plan is outside the measured episode')
                bound = read_reference(bound_ref)
                if (bound.get('attempt') != 1 or bound.get('step') != 0
                        or bound['plan']['composition'] != CODE_COMPOSITION
                        or bound['plan']['operations'] != [operation]
                        or bound['controller']['name'] != selected
                        or bound['controller']['implementation'] != 'code-policy'
                        or selected not in report['binding']['groups'].values()):
                    raise ValueError('Native first attempt did not execute this candidate')
            result = qualification.assess(
                contract, expected_contract_sha256=qualification.digest(contract),
                paired_rows=paired_rows,
                false_interception_review=inputs['false_interception_review'],
                statistical_review=inputs['statistical_review'],
                reproducibility_review=inputs['reproducibility_review'],
                inventory=self.manifest['references']['task_inventory']['path'])
            if result.get('eligible_for_independent_selection') is not True:
                return False, 'native_assessment_rejected'
            return True, 'native_assessment_recomputed'
        except (ValueError, KeyError, TypeError, OSError) as error:
            return False, 'invalid_native_qualification: ' + str(error)

    def view(self, compositions, operation_program):
        route = self.manifest['tasks'][self.task]
        allowed = list(route['original_operations'])
        rejected = {}
        for name, candidate in self.manifest['candidates'].items():
            if candidate['task'] != self.task:
                continue
            if self.task in ORIGINAL_CODE_ONLY:
                rejected[name] = 'original_physicalrsi_code_only'
                continue
            candidate = self.manifest['candidates'][name]
            if route['preserve_original'] and candidate.get('validation') is None:
                rejected[name] = 'original_configuration_protected'
                continue
            admitted, reason = self._candidate_admission(name)
            if admitted:
                allowed.append(name)
            else:
                rejected[name] = reason
        operations = (operation_program or {}).get('operations', {})
        if any(name not in operations for name in allowed):
            raise ValueError('Admitted Code operation is absent from runtime catalogue')
        program = deepcopy(operation_program) if operation_program else None
        strategies = {}
        if program is not None:
            program['operations'] = {name: deepcopy(operations[name]) for name in allowed}
            for name, entry in program['operations'].items():
                settings=self.library['program_registry'][name].get('execution_strategy')
                if settings:
                    entry['execution_strategy']=deepcopy(settings)
                    strategies[name]=deepcopy(settings)
        catalogue = {}
        for name, entry in compositions.items():
            selected = entry['steps'][-1]
            if (name == CODE_COMPOSITION and selected == 'memory-program-library' and allowed
                    or selected in route['vla_skills']):
                catalogue[name] = deepcopy(entry)
        if not catalogue:
            raise ValueError('Exact task has no admitted executor composition')
        receipt = dict(schema='dsr.task-policy-admission/v1', exact_configuration=self.task,
                       routing_sha256=qualification.digest(self.manifest),
                       original_preserved=route['preserve_original'],
                       original_code_available=route['original_code_available'],
                       allowed_operations=allowed, allowed_compositions=sorted(catalogue),
                       rejected_candidates=rejected, native_success=None)
        if strategies: receipt['strategy_candidates']=strategies
        return catalogue, program, receipt


class OriginalCodeRouter:
    """Apply an explicit original-worker constraint to an unrouted base library.

    The parent library stays intact. Selection and DM0.5 preload are filtered
    using trusted deployment metadata before any executor is instantiated.
    """
    def __init__(self, library, deployment):
        self.task = deployment['task_name']
        required = ORIGINAL_CODE_ONLY[self.task]
        matches = [(name, entry) for name, entry in library.get('program_registry', {}).items()
                   if entry.get('source_name', name) == required and descriptor_task(entry) == self.task
                   and not entry.get('deployment_status')]
        if len(matches) != 1:
            raise ValueError('One original PhysicalRSI Code Policy is required: ' + self.task)
        self.operation, entry = matches[0]
        skill = library['skills'].get(required, {})
        if skill.get('implementation') != 'code-policy' or skill.get('configuration') != entry['configuration']:
            raise ValueError('Original PhysicalRSI skill and Code operation disagree')
        self.entry_sha256 = entry_identity(entry)

    def allows_vla(self, name):
        return False

    def view(self, compositions, operation_program):
        operations = (operation_program or {}).get('operations', {})
        composition = compositions.get(CODE_COMPOSITION)
        if self.operation not in operations or not composition or composition.get('steps', [])[-1:] != ['memory-program-library']:
            raise ValueError('Required original Code Policy is absent from the runtime catalogue')
        program = deepcopy(operation_program)
        program['operations'] = {self.operation: deepcopy(operations[self.operation])}
        receipt = dict(schema='dsr.task-policy-preference/v1', exact_configuration=self.task,
                       selection_constraint='original_physicalrsi_code_only',
                       original_preserved=True, original_code_available=True,
                       original_entry_sha256=self.entry_sha256,
                       allowed_operations=[self.operation], allowed_compositions=[CODE_COMPOSITION],
                       rejected_candidates={}, native_success=None)
        return {CODE_COMPOSITION: deepcopy(composition)}, program, receipt


class RoutedPlanner:
    """Filter before planning and check again before upstream creates a worker."""
    def __init__(self, planner, router, output):
        self.planner, self.router, self.output = planner, router, Path(output)
        self.last_route = None

    def __getattr__(self, name):
        return getattr(self.planner, name)

    @staticmethod
    def _validate_plan(plan, catalogue, allowed):
        composition = plan.get('composition')
        if composition not in catalogue:
            raise ValueError('Agent selected a task-ineligible composition')
        names = plan.get('operations', [])
        if composition == CODE_COMPOSITION:
            if not isinstance(names, list) or len(names) != 1 or names[0] not in allowed:
                raise ValueError('Agent selected a task-ineligible Code Policy')
        elif names:
            raise ValueError('A VLA composition cannot attach Code operations')
        return names

    def _record(self, receipt, **selection):
        self.last_route = dict(receipt, **selection)
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / (uuid.uuid4().hex + '.json')
        path.write_text(json.dumps(self.last_route, indent=2, sort_keys=True) + '\n')

    def plan(self, **kwargs):
        self.last_route = None
        catalogue, program, receipt = self.router.view(kwargs['compositions'], kwargs['operation_program'])
        # Trusted deployment metadata goes into capability context. No evaluator
        # fields or historical/holdout scores are added to agent observations.
        memory = deepcopy(kwargs['memory'])
        memory['dsr_task_execution_contract'] = dict(
            exact_configuration=self.router.task,
            selection='Choose among the supplied original Code Policy, experimentally validated adaptive Code '
                      'Policy, or eligible VLA composition according to this instruction and current state. '
                      'The bound worker stays fixed until reset. A registered adaptive strategy can choose direct '
                      'execution, validated stages, or recovery and consult its agent multiple times inside this episode.')
        required = receipt.get('required_stage_operations', [])
        if required:
            memory['dsr_task_execution_contract'].update(
                required_stage_operations=required,
                stage_candidates=deepcopy(receipt['stage_candidates']),
                selection=('This development measurement must exercise a staged candidate. Choose one '
                           'required_stage_operation as the single operation. It owns both controllers '
                           'and the one verified handoff under the same physical budget. Other visible '
                           'candidates are retained alternatives, not a fallback for this measurement.'))
        kwargs.update(compositions=catalogue, operation_program=program, memory=memory)
        plan = self.planner.plan(**kwargs)
        names = self._validate_plan(plan, catalogue, receipt['allowed_operations'])
        receipt['initial_selected_plan'] = deepcopy(plan)
        receipt['corrective_selected_plan'] = None
        if required and (plan['composition'] != CODE_COMPOSITION or names[0] not in required):
            # Ask the agent once, before creating any executor. Do not silently
            # rewrite its answer or fall back to an unmeasured whole-task path.
            correction = deepcopy(kwargs)
            correction['operation_program']['operations'] = {
                name: deepcopy(program['operations'][name]) for name in required}
            correction['memory']['dsr_task_execution_contract']['selection_correction'] = dict(
                previous_plan=deepcopy(plan), allowed_operations=required,
                reason='The initial choice did not bind the required stage composition. Select one supplied staged operation.')
            try:
                corrected = self.planner.plan(**correction)
                receipt['corrective_selected_plan'] = deepcopy(corrected)
                names = self._validate_plan(corrected, catalogue, required)
                if corrected['composition'] != CODE_COMPOSITION:
                    raise ValueError('Stage composition required for this measurement')
            except (ValueError, KeyError, TypeError) as error:
                self._record(receipt, status='REJECTED', selected_candidate=None,
                             selected_plan=None, rejection_reason=str(error))
                raise ValueError('Agent stage selection correction rejected; no executor bound') from error
            plan = corrected
        if receipt.get('selection_mode') == 'agent_portfolio' or (
                names and receipt.get('schema') == 'dsr.task-policy-admission/v1'):
            receipt['selected_candidate'] = names[0]
        selected = receipt.get('stage_candidates', {}).get(names[0]) if names else None
        strategy = receipt.get('strategy_candidates', {}).get(names[0]) if names else None
        self._record(receipt, status='SELECTED', selected_plan=deepcopy(plan),
                     selected_stage_composition=deepcopy(selected),selected_execution_strategy=deepcopy(strategy))
        return plan
