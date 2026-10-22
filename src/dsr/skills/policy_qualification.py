"""Fail-closed scope gate for proposed RoboDojo policy/guard changes.

A smoke result never qualifies a policy. This gate checks the installed official
schedule, raw native episode reports, repeated paired outcomes, explicit guard
annotations and independent statistical review. It never updates a live library.
The trusted evaluator owns the manifest, layout hashes and annotations; candidate
code must not produce its own judging inputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def native_library_sha256(library):
    """Match PhysicalRSI's digest of the exact loaded JSON configuration."""
    return hashlib.sha256(json.dumps(library,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def canonical_executable_library_sha256(library):
    """Ignore only relocated service output directories, preserving execution.

    Command arguments, memory, source/file identities, model settings, routing
    and every other value remain in the digest. Output fields elsewhere remain
    significant. The native verifier separately checks executable-file closure.
    """
    if not isinstance(library,dict):
        raise ValueError('An actual executable library JSON object is required')
    value=deepcopy(library)
    for group in ('skills','program_registry'):
        collection=value.get(group)
        if not isinstance(collection,dict):
            raise ValueError('Actual library needs skills and program_registry objects')
        for entry in collection.values():
            if not isinstance(entry,dict):
                raise ValueError('Malformed actual library entry')
            settings=entry.get('configuration',{})
            if not isinstance(settings,dict):
                raise ValueError('Malformed actual library configuration')
            if settings.get('runtime')=='frozen-program-service':
                settings.pop('output',None)
    return native_library_sha256(value)


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(char in '0123456789abcdef' for char in value)


def official_schedule(inventory=None):
    root=Path(__file__).resolve().parents[3]
    path=Path(inventory or root/'metadata/current_task_contracts.json')
    data=json.loads(path.read_text())
    tasks=data['tasks']
    if len(tasks)!=54 or len({item['native_configuration'] for item in tasks})!=54:
        raise ValueError('Full installed 54-configuration inventory is required')
    schedule=[]
    for task in tasks:
        seeds=task['official_seeds']
        if len(seeds)<3 or len(set(seeds))!=len(seeds):
            raise ValueError('Official three-seed schedule is required')
        count=task['official_episodes_per_seed']
        if count not in (25,50):
            raise ValueError('Unexpected official episode schedule; audit the new benchmark version')
        source=task['configuration_source']
        if hashlib.sha256((root/source['path']).read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Native task configuration changed since inventory extraction')
        schedule.extend(f"{task['native_configuration']}/{seed}/{layout}" for seed in seeds for layout in range(count))
    if len(schedule)!=6300 or data['counts']['nominal_episodes']!=6300:
        raise ValueError('Full official 6300-case schedule is required')
    return schedule,hashlib.sha256(path.read_bytes()).hexdigest()


def official_checker_sha256(inventory=None):
    root=Path(__file__).resolve().parents[3]
    path=Path(inventory or root/'metadata/current_task_contracts.json')
    data=json.loads(path.read_text())
    sources=[data['protocol_source'],*[task['task_source'] for task in data['tasks']]]
    identities={}
    for source in sources:
        actual=hashlib.sha256((root/source['path']).read_bytes()).hexdigest()
        if actual!=source['sha256']:
            raise ValueError('Native task/checker protocol changed since source audit')
        identities[source['path']]=actual
    for path in sorted((root/'vendor/RoboDojo/env/reward_manager').rglob('*.py')):
        identities[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(identities)


def statistical_preflight(schedule, holdout_layout_file_sha256, *, repeats,
                          familywise_alpha, max_regression_risk, candidate_comparisons):
    """Plan independent sample capacity; never predict measured successes.

    The loss bound is conditional on a parent success in at least one complete
    repetition. Thus even unique held-out layouts are only capacity, not the
    effective sample. Robust gains use different, parent-failure layouts.
    """
    tasks=sorted({case.split('/')[0] for case in schedule})
    alpha=familywise_alpha/(len(tasks)*candidate_comparisons)
    required=(None if max_regression_risk==0 else
              math.ceil(math.log(alpha)/math.log1p(-max_regression_risk)))
    gain_required=math.ceil(math.log(familywise_alpha/candidate_comparisons)/math.log(.5))
    capacities={task:len({value for case,value in holdout_layout_file_sha256.items()
                         if case.split('/')[0]==task}) for task in tasks}
    official={task:sum(case.split('/')[0]==task for case in schedule) for task in tasks}
    insufficient=(tasks if required is None else
                  [task for task in tasks if capacities[task]<required])
    official_insufficient=(tasks if required is None else
                          [task for task in tasks if official[task]<required])
    risk_total=None if required is None else len(tasks)*required
    statistics_total=None if risk_total is None else risk_total+gain_required
    reasons=[]
    if required is None:
        reasons.append('zero_risk_cannot_be_proven_with_finite_samples')
    else:
        if insufficient:
            reasons.append('insufficient_per_task_independent_holdout_capacity')
        if sum(capacities.values())<statistics_total:
            reasons.append('insufficient_independent_holdout_capacity_for_risk_and_gain')
    return dict(status='qualification_unavailable' if reasons else 'capacity_sufficient_pending_measured_evidence',
        capacity_can_support_qualification=not reasons,reasons=reasons,
        familywise_adjusted_alpha=alpha,max_regression_risk=max_regression_risk,
        required_independent_parent_success_holdout_layouts_per_task=required,
        required_robust_gain_layouts=gain_required,
        planned_unique_holdout_layouts_per_task=capacities,
        insufficient_holdout_configurations=insufficient,
        official_layout_capacity_upper_bound_per_task=official,
        official_layout_capacity_insufficient_configurations=official_insufficient,
        independent_new_layouts_required=bool(official_insufficient),
        minimum_risk_holdout_layouts=risk_total,minimum_statistical_holdout_layouts=statistics_total,
        minimum_native_rollouts_for_risk_bound=(None if risk_total is None else
            2*repeats*(len(schedule)+risk_total)),
        minimum_native_rollouts_for_statistics=(None if statistics_total is None else
            2*repeats*(len(schedule)+statistics_total)),
        planned_native_rollouts=2*repeats*(len(schedule)+len(holdout_layout_file_sha256)),
        assumptions='Optimistic lower bounds require complete repetitions, independent layouts, zero losses, '
          'and enough measured parent-success layouts in every configuration; repeated or aliased layouts '
          'do not increase the effective sample. Robust gain layouts do not count as parent-success layouts.')


def preregister(*, parent_library_sha256, candidate_library_sha256, layout_file_sha256,
                development_layout_file_sha256, checker_sha256, holdout_layout_file_sha256,
                required_holdout_pairs, repeats=2,
                max_step_ratio=1.0, max_elapsed_ratio=1.2,
                familywise_alpha=.05, max_regression_risk=.01, candidate_comparisons=1,
                initial_position_tolerance_m=.001, initial_angle_tolerance_rad=.001,
                initial_joint_tolerance_rad=.001, initial_velocity_tolerance=.001,
                intervention_mode='active', inventory=None):
    """Freeze all official cases and budgets before independent evaluation.

    Layout-file fingerprints must come from the evaluator. They identify the
    layout recipe, not the actual physical checkpoint. Per-pair checkpoint
    artifacts are separately required and compared at fixed tolerances.
    Parent/candidate library hashes pin canonical executable identities, which
    ignore only frozen-program-service configuration.output relocation. Each
    measured pair must also supply its actual loaded skill_library.json file.
    At least two complete repeats are needed to examine reproducibility.
    """
    schedule,inventory_sha=official_schedule(inventory)
    if set(layout_file_sha256)!=set(schedule) or any(not _sha(value) for value in layout_file_sha256.values()):
        raise ValueError('Every official case needs its actual initial-layout SHA256')
    if any(not _sha(value) for value in development_layout_file_sha256):
        raise ValueError('Development layout identities must be SHA256 fingerprints')
    tasks={key.split('/')[0] for key in schedule}
    if not holdout_layout_file_sha256 or any(not _sha(value) for value in holdout_layout_file_sha256.values()):
        raise ValueError('Independent holdout layout fingerprints are required')
    if set(holdout_layout_file_sha256.values()) & set(development_layout_file_sha256):
        raise ValueError('Development-seen layouts cannot be presented as held-out evaluation')
    if {key.split('/')[0] for key in holdout_layout_file_sha256} != tasks:
        raise ValueError('Independent validation must cover all task configurations')
    for task in tasks:
        keys=[key.split('/') for key in holdout_layout_file_sha256 if key.split('/')[0]==task]
        if any(len(key)!=3 for key in keys) or len({key[1] for key in keys})<3:
            raise ValueError('Independent validation requires at least three seeds per task')
        if any(len({key[2] for key in keys if key[1]==seed})<2 for seed in {key[1] for key in keys}):
            raise ValueError('Independent validation requires multiple layouts per seed')
    if type(required_holdout_pairs) is not int or required_holdout_pairs<len(tasks)*3*2 or len(holdout_layout_file_sha256)<required_holdout_pairs:
        raise ValueError('Independent holdout size must meet the preregistered statistical sample requirement')
    if not all(_sha(value) for value in (parent_library_sha256,candidate_library_sha256,checker_sha256)):
        raise ValueError('Parent, candidate and immutable checker content hashes are required')
    if checker_sha256!=official_checker_sha256(inventory):
        raise ValueError('Checker fingerprint must identify the installed native task/checker sources')
    if parent_library_sha256==candidate_library_sha256:
        raise ValueError('A proposal must identify a distinct candidate revision')
    if type(repeats) is not int or repeats<2:
        raise ValueError('At least two independent complete repeats are required')
    if not 0<max_step_ratio<=1 or not 0<max_elapsed_ratio<=2:
        raise ValueError('Explicit bounded preregistered execution-cost limits are required')
    if not 0<familywise_alpha<=.05 or not 0<=max_regression_risk<1:
        raise ValueError('Prespecified familywise error and residual-risk bounds are required')
    if type(candidate_comparisons) is not int or candidate_comparisons<1:
        raise ValueError('Prespecify the number of candidate comparisons')
    tolerances=(initial_position_tolerance_m,initial_angle_tolerance_rad,initial_joint_tolerance_rad,initial_velocity_tolerance)
    if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<value<=.001 for value in tolerances):
        raise ValueError('Initial physical-state tolerances must be fixed at or below 1e-3')
    if intervention_mode not in ('active','shadow','none'):
        raise ValueError('Unknown intervention mode')
    preflight=statistical_preflight(schedule,holdout_layout_file_sha256,repeats=repeats,
        familywise_alpha=familywise_alpha,max_regression_risk=max_regression_risk,
        candidate_comparisons=candidate_comparisons)
    contract=dict(schema='dsr.robodojo-full-scope-contract/v1',created_at=datetime.now(timezone.utc).isoformat(),
        schedule=schedule,inventory_sha256=inventory_sha,layout_file_sha256=dict(layout_file_sha256),
        development_layout_file_sha256=sorted(development_layout_file_sha256),repeats=repeats,
        seen_regression_cases=sorted(key for key,value in layout_file_sha256.items() if value in development_layout_file_sha256),
        holdout_layout_file_sha256=dict(holdout_layout_file_sha256),required_holdout_pairs=required_holdout_pairs,
        parent_library_sha256=parent_library_sha256,candidate_library_sha256=candidate_library_sha256,
        library_identity_kind='canonical_executable_library_without_service_output/v1',
        checker_sha256=checker_sha256,max_step_ratio=max_step_ratio,max_elapsed_ratio=max_elapsed_ratio,
        familywise_alpha=familywise_alpha,max_regression_risk=max_regression_risk,candidate_comparisons=candidate_comparisons,
        layout_identity_kind='official_layout_file_sha256',
        initial_position_tolerance_m=initial_position_tolerance_m,initial_angle_tolerance_rad=initial_angle_tolerance_rad,
        initial_joint_tolerance_rad=initial_joint_tolerance_rad,initial_velocity_tolerance=initial_velocity_tolerance,
        intervention_mode=intervention_mode,per_repeat_native_rollouts=12600,
        minimum_native_rollouts=(12600+2*len(holdout_layout_file_sha256))*repeats,first_attempt_only=True,
        statistical_preflight=preflight,
        requirements=['zero_observed_parent_success_regressions','no_per_case_score_regression',
          'no_cross_task_compensation','explicit_independent_false_interception_annotations',
          'repeatability_review','independent_statistical_review','measured_execution_cost'])
    return contract


def _artifact(reference):
    path=Path(reference['path']).resolve(strict=True)
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=reference['sha256']:
        raise ValueError('Judging artifact changed: '+str(path))
    return json.loads(payload)


def _intervention_coverage(reference, *, contract_sha256, expected, candidate_measurements):
    """Read evaluator-owned complete physical-action coverage for mode none.

    Measurements bind each candidate's raw native report, cover all physical
    actions and enumerate actually intercepted action indices. A review's zero
    count cannot replace these logs or override an observed event. As with the
    other judging inputs, the evaluator must own this artifact outside the
    candidate's writable capability; this reader is not an OS trust boundary.
    """
    log=_artifact(reference)
    if (not isinstance(log,dict) or log.get('schema')!='dsr.independent-intervention-log/v1'
            or log.get('contract_sha256')!=contract_sha256 or log.get('independent_validator') is not True):
        raise ValueError('Intervention coverage has no independent frozen-contract provenance')
    records=log.get('records')
    if not isinstance(records,list):
        raise ValueError('Complete intervention coverage records are required')
    seen=set();total=0
    for record in records:
        if not isinstance(record,dict):
            raise ValueError('Malformed intervention coverage record')
        key=(record['split'],record['case'],record['repeat'])
        if type(record['repeat']) is not int or key not in expected or key in seen:
            raise ValueError('Unregistered or duplicate intervention coverage case/repeat')
        seen.add(key)
        report_reference,steps=candidate_measurements[key]
        if record['candidate_report_sha256']!=report_reference['sha256']:
            raise ValueError('Intervention coverage does not bind the actual candidate report')
        if type(record['observed_actions']) is not int or record['observed_actions']!=steps or steps<=0:
            raise ValueError('Intervention coverage does not cover every measured physical action')
        events=record['interventions']
        if not isinstance(events,list):
            raise ValueError('Measured intervention events are required, including explicit empty lists')
        if any(not isinstance(event,dict) for event in events):
            raise ValueError('Malformed measured intervention event')
        indices=[event['action_index'] for event in events]
        if any(type(index) is not int or not 0<=index<steps for index in indices) or len(set(indices))!=len(indices):
            raise ValueError('Intervention events must identify distinct measured physical action indices')
        total+=len(events)
    if seen!=expected:
        raise ValueError('Intervention coverage does not cover the complete repeated scope')
    return total


def _review_interventions(review, *, mode, contract_sha256, expected, candidate_measurements):
    if mode=='active' and (type(review['actual_interventions_assessed']) is not int
                           or review['actual_interventions_assessed']<=0):
        raise ValueError('No active interventions were independently assessed')
    if mode=='none':
        count=review['actual_interventions_assessed']
        if type(count) is not int or count<0:
            raise ValueError('An explicit nonnegative measured intervention count is required')
        measured=_intervention_coverage(review['intervention_log'],contract_sha256=contract_sha256,
            expected=expected,candidate_measurements=candidate_measurements)
        if measured!=count:
            raise ValueError('Review intervention count differs from the actual coverage log')
        if measured!=0:
            raise ValueError('Actual interceptions cannot qualify under intervention mode none')


def _outcome(reference, *, library_reference, task, seed, layout, revision):
    report=_artifact(reference)
    library=_artifact(library_reference)
    first=report['first_attempt']
    if (report['task'],report['seed'],report['layout_id'])!=(task,seed,layout):
        raise ValueError('Native episode report has a different case identity')
    if report['status']!='COMPLETED' or first['status']!='COMPLETED':
        raise ValueError('Infrastructure or incomplete outcomes cannot qualify a candidate')
    if report['binding']['library_sha256']!=native_library_sha256(library):
        raise ValueError('Actual loaded library artifact differs from native report binding')
    if canonical_executable_library_sha256(library)!=revision:
        raise ValueError('Actual executable library differs from the frozen candidate/parent')
    if type(first['success']) is not bool or type(first['steps']) is not int or first['steps']<=0:
        raise ValueError('First-attempt native outcome is missing or malformed')
    score=first['score'];seconds=report['wall_seconds']
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not 0<=score<=1:
        raise ValueError('Invalid first-attempt score')
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not 0<seconds<1e9:
        raise ValueError('Measured episode cost is required')
    return dict(success=first['success'],score=score,steps=first['steps'],elapsed_seconds=seconds)


def compare_initial_state(parent_reference,candidate_reference,contract,*,task,seed,layout):
    """Compare evaluator-private physical checkpoints; never policy inputs.

    The full runner must export this normalized numeric receipt from the actual
    simulator. A layout recipe hash cannot substitute for this measurement.
    No export is synthesized by this module when the measurement is absent.
    """
    import numpy as np
    states=[_artifact(reference) for reference in (parent_reference,candidate_reference)]
    for state in states:
        if (state.get('schema')!='dsr.native-pair-state/v1' or state.get('captured_before_first_action') is not True
                or state.get('frame')!='world' or state.get('task')!=task or state.get('seed')!=seed or state.get('layout_id')!=layout):
            raise ValueError('Initial physical checkpoint identity/capture contract is missing')
        if not state.get('objects') or set(state.get('robots',{}))!={'left','right'}:
            raise ValueError('Complete paired object and robot checkpoints are required')
    def vector(value,length=None):
        values=np.asarray(value,float)
        if values.ndim!=1 or not len(values) or (length is not None and len(values)!=length) or not np.isfinite(values).all():
            raise ValueError('Invalid physical checkpoint vector')
        return values
    for group in ('objects','robots'):
        left,right=states[0][group],states[1][group]
        if set(left)!=set(right):
            raise ValueError('Initial object/robot identity sets differ')
        for name in left:
            a,b=left[name],right[name]
            p,q=vector(a['pose'],7),vector(b['pose'],7)
            if np.linalg.norm(p[:3]-q[:3])>contract['initial_position_tolerance_m']:
                raise ValueError('Initial physical positions differ beyond the fixed tolerance')
            if abs(np.linalg.norm(p[3:])-1)>1e-4 or abs(np.linalg.norm(q[3:])-1)>1e-4:
                raise ValueError('Physical checkpoint orientation is not normalized')
            dot=abs(float(np.dot(p[3:],q[3:])/(np.linalg.norm(p[3:])*np.linalg.norm(q[3:]))))
            if 2*math.acos(min(1.,dot))>contract['initial_angle_tolerance_rad']:
                raise ValueError('Initial physical orientations differ beyond the fixed tolerance')
            fields=['linear_velocity','angular_velocity']
            if group=='robots' or 'joints' in a or 'joints' in b:
                fields+=['joints','joint_velocities']
            for field in fields:
                u,v=vector(a[field]),vector(b[field])
                limit=contract['initial_joint_tolerance_rad'] if field=='joints' else contract['initial_velocity_tolerance']
                if u.shape!=v.shape or np.max(np.abs(u-v))>limit:
                    raise ValueError('Initial physical '+field+' differ beyond the fixed tolerance')
    return True


def paired_statistics(contract, outcomes):
    """Compute uncertainty from real paired outcomes, independently of reviews.

    Repeats and aliases of one actual layout never increase the independent
    holdout denominator. An observed loss in any repetition counts as a loss;
    a robust gain requires the candidate to win in every collected repetition.
    Binomial coverage assumes independently sampled held-out layouts. Reports
    state that assumption and never turn a finite sample into zero-risk proof.
    """
    from scipy.stats import beta, binom
    tasks=sorted({case.split('/')[0] for case in contract['schedule']})
    alpha=contract['familywise_alpha']/(len(tasks)*contract['candidate_comparisons'])
    risk=contract['max_regression_risk']
    needed=None if risk==0 else math.ceil(math.log(alpha)/math.log1p(-risk))
    rows={task:dict(paired_observations=0,parent_successes=0,wins=0,losses=0,holdout={}) for task in tasks}
    for key,parent,candidate in outcomes:
        split,case,repeat=key;task=case.split('/')[0]
        row=rows[task]
        row['paired_observations']+=1
        row['parent_successes']+=int(parent['success'])
        row['wins']+=int(candidate['success'] and not parent['success'])
        row['losses']+=int(parent['success'] and not candidate['success'])
        if split=='holdout':
            layout=contract['holdout_layout_file_sha256'][case]
            trial=row['holdout'].setdefault(layout,[])
            trial.append((repeat,parent['success'],candidate['success']))
    total_wins=total_losses=0
    for row in rows.values():
        n=losses=wins=0
        for trial in row.pop('holdout').values():
            if {part[0] for part in trial}!=set(range(contract['repeats'])):
                continue
            if any(part[1] for part in trial):
                n+=1
                losses+=int(any(parent and not candidate for _,parent,candidate in trial))
            wins+=int(all(not parent and candidate for _,parent,candidate in trial))
        upper=(1. if n==0 or losses==n else
               (1-alpha**(1/n) if losses==0 else float(beta.ppf(1-alpha,losses+1,n-losses))))
        discordant=wins+losses
        row.update(independent_holdout_parent_success_layouts=n,independent_holdout_losses=losses,
            robust_holdout_wins=wins,one_sided_loss_risk_upper_bound=upper,
            familywise_adjusted_alpha=alpha,zero_loss_layouts_required_for_bound=needed,
            bound_satisfied=upper<=risk,
            paired_gain_p_value=float(binom.sf(wins-1,discordant,.5)) if discordant else 1.)
        total_wins+=wins;total_losses+=losses
    discordant=total_wins+total_losses
    global_p=float(binom.sf(total_wins-1,discordant,.5)) if discordant else 1.
    return dict(per_task=rows,robust_holdout_wins=total_wins,independent_holdout_losses=total_losses,
        global_paired_gain_p_value=global_p,gain_significant=global_p<=contract['familywise_alpha']/contract['candidate_comparisons'],
        all_task_risk_bounds_satisfied=all(row['bound_satisfied'] for row in rows.values()),
        method='One-sided exact binomial loss upper bound; Bonferroni over configurations and candidates. Paired exact sign test for gains.',
        assumptions='Independently sampled held-out initial layouts. Repetitions and identical layout hashes are not independent observations.',
        zero_risk_proven=False)


def assess(contract, *, expected_contract_sha256, paired_rows, false_interception_review=None,
           statistical_review=None, reproducibility_review=None, inventory=None):
    """Return explicit rejection reasons; never publish or enable a policy.

    paired_rows bind native reports and independently captured layout/checker
    identities. Reviews are content-addressed artifacts from the experiment
    validator, not the candidate. Their provenance must remain outside the
    candidate's writable capability. This function does not establish an OS
    security boundary around that validator.
    """
    reasons=[]
    if digest(contract)!=expected_contract_sha256:
        reasons.append('preregistered_contract_changed')
    try:
        checked=preregister(**{key:contract[key] for key in ('parent_library_sha256','candidate_library_sha256',
            'layout_file_sha256','development_layout_file_sha256','checker_sha256','holdout_layout_file_sha256',
            'required_holdout_pairs','repeats','max_step_ratio','max_elapsed_ratio','intervention_mode',
            'familywise_alpha','max_regression_risk','candidate_comparisons',
            'initial_position_tolerance_m','initial_angle_tolerance_rad','initial_joint_tolerance_rad','initial_velocity_tolerance')},inventory=inventory)
        checked['created_at']=contract['created_at']
        if checked!=contract:
            reasons.append('contract_does_not_match_full_scope_requirements')
        schedule,inventory_sha=official_schedule(inventory)
        if contract['inventory_sha256']!=inventory_sha or contract['schedule']!=schedule:
            reasons.append('official_schedule_or_inventory_mismatch')
        if set(contract['layout_file_sha256'])!=set(schedule):
            reasons.append('official_layout_file_fingerprints_missing')
        if set(contract['holdout_layout_file_sha256'].values()) & set(contract['development_layout_file_sha256']):
            reasons.append('development_holdout_layout_leakage')
        if len(contract['holdout_layout_file_sha256'])<contract['required_holdout_pairs']:
            reasons.append('independent_holdout_sample_size_insufficient')
        if not contract['statistical_preflight']['capacity_can_support_qualification']:
            reasons.append('statistical_preflight_qualification_unavailable')
        repeats=contract['repeats']
        if type(repeats) is not int or repeats<2:
            raise ValueError('Complete repeated evaluation is required')
        expected={(split,key,repeat) for split,cases in [('regression',schedule),('holdout',contract['holdout_layout_file_sha256'])]
                  for key in cases for repeat in range(repeats)}
    except (ValueError,KeyError,TypeError) as error:
        return dict(status='rejected',promotion_allowed=False,eligible_for_independent_selection=False,
                    reasons=sorted(set(reasons+['invalid_contract: '+str(error)])))
    actual={};gains=0;regressions=0;cost_excesses=0;valid_outcomes=[];candidate_measurements={}
    for row in paired_rows:
        try:
            key=(row['split'],row['case'],row['repeat'])
            if key not in expected:
                raise ValueError('Unregistered case/repeat')
            if key in actual:
                raise ValueError('Duplicate case/repeat')
            actual[key]=row
            layouts=contract['layout_file_sha256'] if key[0]=='regression' else contract['holdout_layout_file_sha256']
            if row['parent_layout_file_sha256']!=row['candidate_layout_file_sha256'] or row['parent_layout_file_sha256']!=layouts[key[1]]:
                raise ValueError('Paired official layout-file identities differ')
            if row['checker_sha256']!=contract['checker_sha256']:
                raise ValueError('Native checker changed')
            if datetime.fromisoformat(row['started_at'])<=datetime.fromisoformat(contract['created_at']):
                raise ValueError('Evaluation predates preregistration')
            task,seed,layout=key[1].split('/')
            compare_initial_state(row['parent_initial_state'],row['candidate_initial_state'],contract,
                                  task=task,seed=int(seed),layout=int(layout))
            outcomes=[_outcome(row[version],library_reference=row[version+'_library'],task=task,seed=int(seed),layout=int(layout),
                revision=contract[version+'_library_sha256']) for version in ('parent','candidate')]
            parent,candidate=outcomes
            valid_outcomes.append((key,parent,candidate))
            candidate_measurements[key]=(row['candidate'],candidate['steps'])
            if parent['success'] and not candidate['success']:
                regressions+=1
                reasons.append('parent_success_case_regressed')
            if candidate['score']<parent['score']:
                reasons.append('per_case_score_regression')
            gains+=int(candidate['success'] and not parent['success'])
            if candidate['steps']>parent['steps']*contract['max_step_ratio'] or candidate['elapsed_seconds']>parent['elapsed_seconds']*contract['max_elapsed_ratio']:
                cost_excesses+=1
                reasons.append('preregistered_execution_cost_exceeded')
        except (ValueError,KeyError,TypeError,OSError) as error:
            reasons.append('invalid_pair: '+str(error))
    missing=expected-set(actual)
    if missing:
        reasons.append('incomplete_official_paired_repeated_scope')
    for name,reference in [('false_interception',false_interception_review),('statistical',statistical_review),('reproducibility',reproducibility_review)]:
        if reference is None:
            reasons.append(name+'_review_missing')
            continue
        try:
            review=_artifact(reference)
            if review['contract_sha256']!=expected_contract_sha256 or review['independent_validator'] is not True:
                raise ValueError('Review has no independent frozen-contract provenance')
            if review['covered_pairs']!=len(expected) or review['passed'] is not True:
                raise ValueError('Review does not pass the complete repeated scope')
            if name=='false_interception':
                if review['assessed_case_repeats']!=len(expected) or type(review['false_interceptions']) is not int or review['false_interceptions']!=0:
                    raise ValueError('Explicit complete false-interception measurement is required')
                _review_interventions(review,mode=contract['intervention_mode'],contract_sha256=expected_contract_sha256,
                    expected=expected,candidate_measurements=candidate_measurements)
            if name=='statistical' and (review['prespecified_method'] is not True or review['multiple_comparisons_accounted'] is not True or review['holdout_power_adequate'] is not True):
                raise ValueError('Statistical selection protocol was not fixed independently')
        except (ValueError,KeyError,TypeError,OSError) as error:
            reasons.append(name+'_review_invalid: '+str(error))
    if contract.get('intervention_mode')=='shadow':
        reasons.append('shadow_diagnostics_cannot_qualify_an_active_guard')
    if gains==0:
        reasons.append('no_observed_new_first_attempt_successes')
    statistics=paired_statistics(contract,valid_outcomes)
    if not statistics['all_task_risk_bounds_satisfied']:
        reasons.append('independent_holdout_risk_bounds_not_satisfied')
    if not statistics['gain_significant']:
        reasons.append('independent_holdout_gain_not_statistically_established')
    return dict(schema='dsr.robodojo-full-scope-assessment/v1',status='rejected' if reasons else 'eligible_for_independent_selection',
        promotion_allowed=False,eligible_for_independent_selection=not reasons,reasons=sorted(set(reasons)),
        expected_paired_cases=len(expected),received_paired_cases=len(actual),missing_paired_cases=len(missing),
        observed_gains=gains,observed_regressions=regressions,cost_excesses=cost_excesses,
        statistics=statistics,
        statistical_preflight=contract['statistical_preflight'],
        conclusion_scope='Finite preregistered official suite only; no guarantee about unknown conditions.',
        seen_regression_case_count=len(contract['seen_regression_cases']),
        independent_holdout_cases=len(contract['holdout_layout_file_sha256']),deployment_performed=False)
