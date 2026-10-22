"""Build a separate 22+14 library using PhysicalRSI's native descriptors."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from .catalog import POLICIES, memory_for
from .program import specification


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


def controller_revision(package, name, dependencies):
    """Bind policy, shared implementation, memory and reused frozen sources.

    Generated program/manifest files are excluded to avoid a hash cycle.
    The package manifest separately binds those generated files.
    """
    package = Path(package)
    identity = dict(policy=name, parameters=sha(package/(name+'.memory.json')),
        sources={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py'))},
        dependencies=dependencies)
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def build(library, output):
    library=Path(library).resolve(strict=True);output=Path(output).resolve()
    original=json.loads(library.read_text());config=deepcopy(original)
    code={k:v for k,v in original["skills"].items() if v["implementation"]=="code-policy"}
    if len(code)!=22 or len(original["program_registry"])!=22:
        raise ValueError("Expansion requires the audited 22-program parent library")
    if output.exists():raise FileExistsError("Use a new immutable output directory")
    output.mkdir(parents=True)
    package=output/"dsr_expansion36"
    shutil.copytree(Path(__file__).parent,package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__).parent.parent/'native_specialists/depth_calibration.py',
                 package/'depth_calibration.py')
    shutil.copy2(Path(__file__).parent.parent/'native_specialists/trace.py', package/'trace.py')
    source_hashes={name:sha(package/'policies'/(entry['module']+'.py')) for name,entry in POLICIES.items()}
    if len(set(source_hashes.values()))!=len(POLICIES):raise ValueError("Candidate entry aliases detected")
    base=deepcopy(original['skills']['general-pickup']['configuration'])
    command=base['command'];descriptor_index=command.index('--descriptor')+1
    descriptor=json.loads(Path(command[descriptor_index]).read_text())
    assets=Path(command[command.index('--assets')+1])
    # Pin both reused RGB backends and their local templates. No model is loaded
    # during packaging; pretrained weights retain upstream asset locations.
    perception_roots=[assets/'implementations/rdj_master/7ad18230694dd7d4',
                      assets/'implementations/arrange_rgb/7990151605e857df']
    dependencies={str(p):sha(p) for root in perception_roots for p in root.rglob('*')
                  if p.is_file() and p.suffix in ('.py','.json','.npz')}
    dependencies.update(base['files'])
    revisions={}
    for name in POLICIES:
        write(package/(name+'.memory.json'),memory_for(name))
        revisions[name]=controller_revision(package,name,dependencies)
        write(package/(name+'.program.json'),specification(name,revisions[name]))
    manifest={"schema":"dsr.code-policy-source/v1","files":{
        str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*')) if p.is_file()}}
    write(package/'package.json',manifest)
    baseline_path=Path(__file__).resolve().parents[4]/'evidence/expansion36_20261012/physicalrsi_official_results_20260928.json'
    baseline=json.loads(baseline_path.read_text()) if baseline_path.is_file() else None
    records=[]
    for name,entry in POLICIES.items():
        desc=deepcopy(descriptor)
        worker=package/'worker.py';revision=sha(worker)
        desc.update(name=name,implementation=dict(module='dsr_expansion36.worker',
            **{'class':'Model'},source=str(worker),sha256=revision,implementation_revision=revision))
        desc['pythonpath']=list(dict.fromkeys([str(output),*[str(p) for p in perception_roots],*desc['pythonpath']]))
        desc.pop('operation_memory_files',None)
        desc['model_config'].update(task_name=entry['tasks'][0],action_type='ee',
            expansion_policy=name,expansion_controller_revision=revisions[name],expansion_package_sha256=sha(package/'package.json'))
        desc['environment']['RDJ_CONFIG']=entry['tasks'][0]
        path=output/(name+'.descriptor.json');write(path,desc)
        settings=deepcopy(base)
        settings.pop('operation_memories',None)
        settings['command'][descriptor_index]=str(path)
        settings['output']=str(output/'code_skills'/name)
        settings['files'].update(dependencies)
        settings['files'].update({str(p):sha(p) for p in package.rglob('*') if p.is_file()})
        settings['files'][str(path)]=sha(path)
        config['skills'][name]=dict(name=name,implementation='code-policy',configuration=settings)
        config['program_registry'][name]=dict(source_name=name,
            description=entry['hypothesis']+' Supported tasks: '+', '.join(entry['tasks'])+'. Development candidate, native qualification pending.',
            memory_primitives=['observation_grounding','identity_ledger','bounded_acquisition',
                               'relative_placement','released_stability'],
            action_type='ee',input=entry['input'],output=entry['output'],configuration=deepcopy(settings),
            deployment_status='disabled_pending_native_validation',
            handoff_contract=deepcopy(entry['handoff_contract']),
            checker_contract=deepcopy(entry['checker_contract']))
        records.append(dict(name=name,**entry,published_task_metrics={t:baseline['task_metrics'][t] for t in entry['tasks']} if baseline else None,controller_sha256=source_hashes[name],descriptor=str(path),
                            memory_sha256=sha(package/(name+'.memory.json'))))
    # Parent source/configuration remains byte-equivalent as JSON objects.
    assert all(config['skills'][k]==v for k,v in original['skills'].items())
    assert all(config['program_registry'][k]==v for k,v in original['program_registry'].items())
    write(output/'skill_library.json',config)
    count=len(code)+len(POLICIES)
    inventory=dict(schema='dsr.expansion36/v1',parent_library=str(library),parent_sha256=sha(library),
        package_sha256=sha(package/'package.json'),code_programs=count,original_programs=len(code),
        new_programs=len(POLICIES),trained_skills=len(config['skills'])-count,parent_entries_preserved=True,
        native_trials=0,native_success_rate=None,promoted_programs=0,composition_implemented=False,
        candidates=records,activation='isolated_artifact_only; not loaded into live selector',
        launch_contract='NativeLibraryVerifier relocates all worker outputs before launch; do not reuse parent output directories')
    write(output/'inventory.json',inventory)
    return inventory
