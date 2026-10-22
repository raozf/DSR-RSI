"""Evidence-based inventory for the current low-SR development campaign.

This is a source review of unfinished capabilities, not policy qualification.
Entries are deliberately not inferred complete from names or registrations.
Update the reviewed gaps with the implementation and its native evidence in
the same recorded change. A missing entry is a blocker, never implicit approval.
"""
import hashlib
import json
from pathlib import Path

from dsr.skills.task_specialists.catalog import DESIGNS, STAGE_DESIGNS
from dsr.skills.native_specialists.catalog import SPECS
from dsr.skills.expansion36.catalog import POLICIES


ROOT=Path(__file__).resolve().parents[3]
HISTORY=ROOT/'evidence/expansion36_20261012/physicalrsi_official_results_20260928.json'
RANDOM_OPERATIONS={
    'pour_liquid_into_cup_random':'random-pour-spout-feedback-v1',
    'push_T_random':'random-push-t-response-v1',
    'stack_bowls_random':'random-bowl-profile-nesting-v1',
}
# Concrete known gaps, beyond the common requirement for native mechanisms.
KNOWN_GAPS={
    'fasten_screws':'No new native nut/bolt adapter or measured 1 mm visual alignment.',
    'imitate_sorting_sequence':'Decision program lacks an online demonstration reader and native transfer adapter.',
    'organize_table':'No new native controller covering all four support/orientation constraints.',
    'pick_from_conveyor_by_image':'No native image-match interception controller with a continuously held raised basket.',
    'play_stacking_toy':'No ten-piece/four-peg native adapter or measured 1 mm local alignment.',
    'pour_by_language':'Frozen v9 native pilot (seed 0/layout 1) failed 0/1: initial_upright_unconfirmed at step 53, then 747 failure holds. Recorded fragment-union RGB replay passed 0/5 geometry fits. See evidence/low_sr_pour_v9c_20261014/receipt.json. Axis/attachment observability, fluid-transfer feedback and measured three-pair budget closure remain unresolved; this pilot is not a population SR estimate.',
    'store_tools_in_toolbox':'Original one-tool truncation and incomplete tool/slot reader are not replaced by a four-tool adapter.',
    'match_and_pick_from_conveyor':'Decision program lacks an online first-return reader, physical clock and native interception adapter.',
    'make_toast':'Recorded reader v7 resolves four bodies but fails faces/slots/rack; insertion floor and lever remain ungrounded.',
    'make_toast_random':'New SAM3 instance reader has no random RGB/native mechanism evidence; floor/lever geometry unresolved.',
}


def reference(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load(path):
    return json.loads(Path(path).read_text())


def source_drift(config):
    """Compare declared frozen Python dependencies with the current sources."""
    packages={'dsr_native_specialists':'native_specialists','dsr_expansion36':'expansion36','dsr_random3':'random_adaptation'}
    drift=set();checked=set()
    for entry in config.get('program_registry',{}).values():
        for name,digest in entry.get('configuration',{}).get('files',{}).items():
            path=Path(name)
            if path.suffix!='.py' or name in checked:continue
            checked.add(name)
            for package,folder in packages.items():
                if package not in path.parts:continue
                relative=Path(*path.parts[path.parts.index(package)+1:])
                if package=='dsr_native_specialists' and relative.name=='strategy_runtime.py':
                    current=ROOT/'src/dsr/harness/strategy.py'
                elif package=='dsr_native_specialists' and relative.name=='strategy_agent.py':
                    current=ROOT/'src/dsr/system2/strategy_agent.py'
                elif relative.parts[0]=='analytical':current=ROOT/'src/dsr/skills/task_specialists'/Path(*relative.parts[1:])
                elif package=='dsr_expansion36' and relative.name=='depth_calibration.py':
                    current=ROOT/'src/dsr/skills/native_specialists/depth_calibration.py'
                else:current=ROOT/'src/dsr/skills'/folder/relative
                if not current.is_file() or reference(current)['sha256']!=digest:
                    drift.add(str(current.relative_to(ROOT)))
                if not path.is_file() or reference(path)['sha256']!=digest:
                    drift.add('modified_frozen_source:'+name)
    return sorted(drift)


def audit(libraries=()):
    historical=load(HISTORY)['task_metrics']
    arm_rows=[];bindings={};blockers=[]
    for path in libraries:
        path=Path(path).resolve(strict=True);library=load(path);probe=library.get('task_policy_probe',{})
        tasks=probe.get('tasks',{});drift=source_drift(library)
        arm_rows.append(dict(library=reference(path),arm=probe.get('arm'),tasks=sorted(tasks),source_drift=drift))
        if not tasks:blockers.append('Library has no exact probe task bindings: '+str(path))
        for task,row in tasks.items():
            for operation in row.get('candidates',{}):bindings.setdefault(task,[]).append(dict(arm=probe.get('arm'),operation=operation))
        if drift:blockers.append('Frozen arm does not contain latest reviewed Python sources: '+str(path))
    rows=[]
    for design in DESIGNS:
        for task in design['tasks']:
            rate=historical.get(task,{}).get('sr')
            if type(rate) not in (int,float) or not 0<=rate<50:
                blockers.append('Design has unknown/protected historical configuration: '+task)
                continue
            earlier=[name for name,spec in POLICIES.items() if task in spec['tasks']]
            native=('specialist-'+task.replace('_','-')+'-v1') if task in SPECS else None
            random=RANDOM_OPERATIONS.get(task)
            if native:state='new_native_candidate_unverified'
            elif random:state='isolated_random_candidate_unverified'
            elif earlier:state='earlier_expansion_is_not_latest_redesign'
            else:state='new_native_adapter_missing'
            gaps=[]
            if task in KNOWN_GAPS:gaps.append(KNOWN_GAPS[task])
            if state=='earlier_expansion_is_not_latest_redesign':
                gaps.append('Earlier generic expansion does not establish preservation/integration of the latest specialized design.')
            gaps.append('No passing native mechanism review bound to the latest complete design; software/loader/RGB replay is insufficient.')
            planned=bindings.get(task,[])
            if libraries and not planned:gaps.append('No selected measurement arm binds this exact configuration.')
            rows.append(dict(design=design['id'],task=task,historical_sr=rate,
                estimated=historical[task].get('estimated',False),status=state,
                native_candidate=native,earlier_expansion=earlier,isolated_random_candidate=random,
                decision_program=design['implemented_decision_program'],planned_bindings=planned,
                latest_design_complete=False,mechanism_verified=False,native_success_rate=None,gaps=gaps))
    combinations=[]
    for pair in STAGE_DESIGNS:
        for task in pair['tasks']:
            combinations.append(dict(task=task,pair=pair['pair'],code_to_code='native_stage_api_implemented_unverified',
                code_vla='guarded_live_adapter_missing',historical_sr=historical[task]['sr'],
                planned_bindings=[x for x in bindings.get(task,[]) if x['operation'].startswith('composition-')],
                launch_ready=False))
    blockers.extend(f"{row['task']}: "+'; '.join(row['gaps']) for row in rows)
    blockers.append('Code/VLA guarded live handoff and bound native stage evidence remain missing.')
    blockers.append('Analytical skills/decision programs require actual per-controller integration receipts; registration alone is not execution.')
    return dict(schema='dsr.low-sr-readiness/v1',historical_metrics=reference(HISTORY),
        design_source=reference(ROOT/'src/dsr/skills/task_specialists/catalog.py'),review_source=reference(__file__),
        design_count=len(DESIGNS),exact_design_tasks=len(rows),rows=rows,arms=arm_rows,combinations=combinations,
        preserved_high_sr='Original routing/Code/skills remain unchanged; excluded from this measurement.',
        three_random_candidates='Separate unpromoted arms only; never production fallback.',
        protocol=dict(attempts=1,episode_retry=False,seed_restart_loop=False),
        launch_ready=not blockers,blockers=blockers,native_success_rate=None)


def markdown(record):
    lines=['# Low-SR capability and launch review','',
        'Eight-GPU launch: **'+('READY' if record['launch_ready'] else 'BLOCKED')+'**.',
        'This inventory distinguishes implementations from verified native mechanisms. It does not report task success rates.','',
        '| Exact task | Historical SR | Latest design | Implementation state | Planned arms |',
        '|---|---:|---|---|---|']
    for row in record['rows']:
        lines.append(f"| {row['task']} | {row['historical_sr']:g}% | {row['design']} | {row['status']} | "+
                     ', '.join(b['arm'] or 'unknown' for b in row['planned_bindings'])+' |')
    lines+=['','## Current blockers','']+['- '+item for item in record['blockers']]
    lines+=['','## Stage combinations','',
        '| Task | Pair | Code to Code | Code/VLA |','|---|---|---|---|']
    lines.extend(f"| {x['task']} | {x['pair']} | {x['code_to_code']} | {x['code_vla']} |" for x in record['combinations'])
    lines+=['','## Source freshness','']
    for arm in record['arms']:
        lines.append('- '+str(arm['arm'])+': '+((', '.join(arm['source_drift'])) or 'Declared Python sources match; no native capability inference.'))
    lines+=['','Protocol: one attempt; no episode retry; no failed-seed restart loop. High-SR originals are preserved. The three older random candidates stay isolated.','']
    return '\n'.join(lines)


def enforce_large_probe(library, workers, *, allow_unverified_measurement=False):
    """Keep the capability gate; permit an explicit, audited measurement only."""
    if workers<4 and not allow_unverified_measurement:return None
    result=audit([library])
    if allow_unverified_measurement:
        # A task-local RSI revision intentionally differs from workspace HEAD.
        # Keep that distinction in the audit, while rejecting any mutation of
        # the declared immutable package itself. Exact route/source provenance
        # is independently checked by TaskProbeRouter before this gate.
        mutated=[item for arm in result['arms'] for item in arm['source_drift']
                 if item.startswith('modified_frozen_source:')]
        if mutated:
            raise ValueError('Unverified low-SR measurement has changed frozen sources: '+', '.join(mutated))
        return result
    if not result['launch_ready']:
        missing=[r['task'] for r in result['rows'] if r['status']=='new_native_adapter_missing']
        raise ValueError('Eight-GPU low-SR campaign is not ready: '+str(len(result['blockers']))+
            ' unresolved requirements; new native adapters missing: '+', '.join(missing)+
            '. Run tools/audit_low_sr_readiness.py for the full review. This is a capability gate, not a permission request.')
    return result
