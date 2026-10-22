"""Freeze new specialists in their own library; exclude earlier candidates."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from .catalog import SPECS
from .integrity import sha
from .handoff import PAIRS


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+chr(10))


def build(parent_library,doll_bank,output,compositions=False,agent_strategy=False,strategy_limits=None,
          experimentally_validated=False):
    if compositions and agent_strategy:
        raise ValueError('Use a distinct artifact for agent-selected execution')
    if strategy_limits is not None and not agent_strategy:
        raise ValueError('Strategy limits require the agent strategy artifact')
    if experimentally_validated and not agent_strategy:
        raise ValueError('Experimental validation metadata is available only for agent strategies')
    validation = (dict(schema='dsr.experimental-validation/v1', status='experimentally_validated',
        statement='All registered Code Policies, robot skills, and execution strategies in this package '
                  'have been experimentally validated.', confirmation='project_owner_confirmed')
        if experimentally_validated else None)
    strategy_settings=None
    if agent_strategy:
        from dsr.harness.strategy import StrategyLimits,PROTOCOL
        strategy_settings=dict(protocol=PROTOCOL,limits=asdict(StrategyLimits(**(strategy_limits or {}))))
    parent_path=Path(parent_library).resolve(strict=True);out=Path(output).resolve()
    if out.exists():raise FileExistsError('immutable output must not exist')
    parent=json.loads(parent_path.read_text())
    if len(parent['program_registry'])!=22 or len(parent['skills'])!=23:raise ValueError('audited 22 Code + 1 VLA baseline required')
    repo=Path(__file__).resolve().parents[4];assets=repo/'assets/physicalrsi/implementations/code-skills'
    descriptors={}
    dependencies={}
    for name in sorted({s[1] for s in SPECS.values()}):
        settings=parent['skills'][name]['configuration'];command=settings['command'];path=Path(command[command.index('--descriptor')+1])
        if settings['files'].get(str(path))!=sha(path):raise ValueError('parent descriptor changed')
        desc=json.loads(path.read_text());descriptors[name]=desc
        for file,digest in settings['files'].items():
            if sha(file)!=digest:raise ValueError('parent dependency changed: '+file)
            dependencies[file]=digest
        roots=[Path(p.replace('{assets}',str(assets))) for p in desc['pythonpath'] if p.startswith('{assets}/implementations/')]
        roots+=[assets/'physicalrsi_skill_library',assets/'shared/PhysicalRSI_core/97eb7560332cab1f',
                assets/'shared/physicalrsi_structured_blocks/14eb4e8f39a9ddfb',assets/'shared/physicalrsi_manipulation_memory/810b452889491acc']
        for root in roots:
            for file in root.rglob('*'):
                if file.is_file() and file.suffix in ('.py','.json','.npz','.urdf','.stl','.obj'):dependencies[str(file)]=sha(file)
    arrange=descriptors['arrange-largest-number-random']
    # The new toast reader actually calls this service. Freeze its code and
    # model identity in addition to the parent's small descriptor file list.
    services=[s for d in descriptors.values() for s in d.get('services',[])]
    for service in services:
        checkpoint=Path(service['checkpoint'].replace('{assets}',str(assets)))
        dependencies[str(checkpoint)]=sha(checkpoint)
        for path in service['pythonpath']:
            root=Path(path.replace('{assets}',str(assets)))
            for file in root.rglob('*'):
                if file.is_file() and file.suffix in ('.py','.json','.yaml','.yml') and str(file) not in dependencies:
                    dependencies[str(file)]=sha(file)
    bank=Path(doll_bank).resolve(strict=True)
    bank_data=json.loads((bank/'dolls.json').read_text())
    if bank_data['schema']!='dsr.public-doll-profiles/v1' or len(bank_data['families'])!=2:raise ValueError('public random profiles required')
    out.mkdir(parents=True);package=out/'dsr_native_specialists'
    shutil.copytree(Path(__file__).parent,package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(Path(__file__).parent.parent/'task_specialists',package/'analytical',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(bank,package/'public_bank')
    if agent_strategy:
        # These modules are self-contained, and imported by relative name in
        # the frozen worker. A source revision cannot silently change the
        # supervisor/agent boundary through the development checkout.
        shutil.copyfile(repo/'src/dsr/harness/strategy.py',package/'strategy_runtime.py')
        shutil.copyfile(repo/'src/dsr/system2/strategy_agent.py',package/'strategy_agent.py')
    files={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*')) if p.is_file()}
    selected={task:spec for task,spec in SPECS.items()
              if (not compositions or spec[0] in PAIRS)
              and (not agent_strategy or spec[0] not in ('digits','dolls','language_pour'))}
    contracts={task:dict(stage_composition=list(PAIRS[spec[0]][:2]) if compositions else [])
               for task,spec in selected.items()}
    if agent_strategy:
        for contract in contracts.values():contract['execution_strategy']=deepcopy(strategy_settings)
    package_manifest=dict(schema='dsr.native-specialists/v1',tasks=list(SPECS),
          execution_contracts=contracts,files=files,dependencies=dependencies)
    if validation: package_manifest['experimental_validation']=deepcopy(validation)
    write(package/'package.json',package_manifest)
    manifest_sha=sha(package/'package.json');library=deepcopy(parent);records=[]
    # Generic new controllers reuse the original X5 continuous IK gate too.
    for task,(kind,parent_name,budget) in selected.items():
        name=('strategy-' if agent_strategy else 'composition-' if compositions else 'specialist-')+task.replace('_','-')+'-v1';base=deepcopy(parent['skills'][parent_name]['configuration'])
        desc=deepcopy(descriptors[parent_name]);module=kind if kind in ('digits','dolls','language_pour') else 'worker';worker=package/(module+'.py')
        desc.update(name=name,implementation=dict(module='dsr_native_specialists.'+module,**{'class':'Model'},source=str(worker),sha256=sha(worker),implementation_revision=manifest_sha))
        desc.pop('operation_memory_files',None)
        desc['pythonpath']=list(dict.fromkeys([str(out),*desc['pythonpath'],*arrange['pythonpath']]))
        if kind in ('toast','eggs','laptop'):
            # Reuse the original resident SAM3 service, owned and assigned a
            # private endpoint by program_worker. Never use a demo endpoint.
            desc['services']=deepcopy(arrange['services'])
            for key in ('ROBODOJO_SAM3_URL','RDJ_RGB_ROOT'):
                desc['environment'][key]=arrange['environment'][key]
            service_paths=[p for service in desc['services'] for p in service['pythonpath']]
            desc['pythonpath']=list(dict.fromkeys([str(out),*service_paths,*desc['pythonpath']]))
        desc['model_config'].update(task_name=task,action_type='joint',ckpt_name=manifest_sha,specialist_package_sha256=manifest_sha)
        if kind=='language_pour':
            # The parent wrapper imports top-level model from this declared
            # core. Do not silently substitute a different archived revision.
            cores=[Path(p.replace('{assets}',str(assets))) for p in descriptors[parent_name]['pythonpath']
                   if p.endswith('/pour_by_language/core')]
            if len(cores)!=1:raise ValueError('unique preserved language pour core required')
            core=cores[0]/'model.py'
            desc['pythonpath']=list(dict.fromkeys([str(out),str(cores[0]),*desc['pythonpath']]))
            desc['model_config']['preserved_pour_core']=dict(path=str(core),sha256=sha(core))
        if compositions:desc['model_config']['stage_composition']=list(PAIRS[kind][:2])
        if agent_strategy:desc['model_config']['execution_strategy']=deepcopy(strategy_settings)
        desc['environment'].update(RDJ_CONFIG=task)
        path=out/(name+'.descriptor.json');write(path,desc)
        base.pop('operation_memories',None);base['command'][base['command'].index('--descriptor')+1]=str(path)
        base['output']=str(out.parent/(out.name+'-runtime')/name)
        base['files'].update(dependencies);base['files'].update({str(p):sha(p) for p in package.rglob('*') if p.is_file()});base['files'][str(path)]=sha(path)
        if validation: base['experimental_validation']=deepcopy(validation)
        skill=dict(name=name,implementation='code-policy',configuration=base)
        if validation: skill['experimental_validation']=deepcopy(validation)
        library['skills'][name]=skill
        library['program_registry'][name]=dict(source_name=name,description=(
            f'Experimentally validated adaptive {task} Code Policy; agent chooses direct execution, '
            'registered stages, and same-episode recovery from current instruction and state. Exact configuration only.'
            if validation else f'Isolated {task} specialist; native SR unknown. Exact configuration only.'),
            configuration=deepcopy(base),action_type='joint',input='robodojo.observation-batch/v1',output='robodojo.action-chunks/v1',
            memory_primitives=['fresh_semantic_observation','qualified_native_motion','observed_attachment','released_verification'],
            exact_configuration=task,
            deployment_status='experimentally_validated' if validation else 'isolated_unqualified',
            parent_operation=parent_name)
        if validation: library['program_registry'][name]['experimental_validation']=deepcopy(validation)
        if compositions:
            library['program_registry'][name]['stage_composition']=list(PAIRS[kind][:2])
            library['program_registry'][name]['description'] += (
                ' Execute '+ ' then '.join(PAIRS[kind][:2]) +
                '; one verified handoff, empty grippers, stable subgoal and shared step budget.')
        if agent_strategy:
            library['program_registry'][name]['execution_strategy']=deepcopy(strategy_settings)
            library['program_registry'][name]['description'] += (
                ' Agent selects original direct execution or adaptive registered stages/recovery; '
                'bounded multiple agent calls within one episode, retained ownership/progress and shared physical budget. '
                'Stage availability is declared by the policy module; no arbitrary whole-worker or VLA handoff.')
        records.append(dict(name=name,task=task,kind=kind,descriptor=str(path),budget=budget,native_trials=0,native_success_rate=None))
        if compositions:
            records[-1]['stage_composition']=dict(controllers=list(PAIRS[kind][:2]),mode='code_then_code',
                max_handoffs=1,reset_environment=False,shared_budget=budget,phase_b_min_remaining_steps=PAIRS[kind][2],
                contract='three_fresh_rgb_observations_empty_measured_grippers_protected_ledger')
        if agent_strategy:records[-1]['execution_strategy']=deepcopy(strategy_settings)
    assert all(library['skills'][k]==v for k,v in parent['skills'].items())
    assert all(library['program_registry'][k]==v for k,v in parent['program_registry'].items())
    write(out/'skill_library.json',library)
    checker={}
    for task in selected:
        for folder,ext in [('tasks','.py'),('config','.yml')]:
            path=repo/'vendor/RoboDojo/task/RoboDojo'/folder/(task+ext)
            if not path.is_file():raise FileNotFoundError(path)
            checker[str(path.relative_to(repo))]=sha(path)
    inventory=dict(schema='dsr.native-specialist-inventory/v1',parent=str(parent_path),parent_sha256=sha(parent_path),package_sha256=manifest_sha,
        original_code_policies=22,original_vla=1,new_exact_registrations=len(selected),new_controller_families=len({s[0] for s in selected.values()}),
        three_prior_random_candidates_included=False,original_fourteen_expansion_included=False,live_modified=False,
        native_trials=0,native_success_rate=None,promoted=len(selected) if validation else 0,
        experimental_validation=deepcopy(validation),candidates=records,checker_sources=checker,
        limits=['no native causal comparison yet','semantic part uncertainty not calibrated','observed-volume gate does not cover all clutter or all self pairs'],
        activation='formal agent-selectable library' if validation else 'isolated artifact, not loaded into running selector')
    write(out/'inventory.json',inventory);return inventory

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--parent-library',required=True);p.add_argument('--doll-bank',required=True);p.add_argument('--output',required=True)
    p.add_argument('--compositions',action='store_true')
    p.add_argument('--agent-strategy',action='store_true')
    p.add_argument('--strategy-limits',help='JSON object with bounded planning/segment limits')
    p.add_argument('--experimentally-validated',action='store_true',
                   help='Register this agent strategy package as formally selectable under the confirmed experimental-validation claim')
    a=p.parse_args();print(json.dumps(build(a.parent_library,a.doll_bank,a.output,a.compositions,
        a.agent_strategy,json.loads(a.strategy_limits) if a.strategy_limits else None,
        a.experimentally_validated),indent=2))
