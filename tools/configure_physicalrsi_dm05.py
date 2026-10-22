#!/usr/bin/env python3
"""Use the official code-skill configurator, substituting the learned VLA."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

from dsr.paths import repository_root, physicalrsi_root
from dsr.runtime_config import code_asset_root, interpreter, checkpoint_root
ROOT = repository_root()
UPSTREAM=physicalrsi_root()
sys.path.insert(0,str(UPSTREAM))

def configure(output,evidence,checkpoint,*,allow_vla_only=False):
    source=UPSTREAM/'PhysicalRSI_baselines/robodojo/configure_skills.py'
    tree=ast.parse(source.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure')
    # Remove precisely the official pi05 checkpoint binding loop. All code
    # program registration, path/hash validation and capability descriptions
    # remain the pinned upstream implementation.
    loops=[n for n in function.body if isinstance(n,ast.For)
           and isinstance(n.target,ast.Tuple)
           and [x.id for x in n.target.elts]==['name','folder']]
    if len(loops)!=1:raise RuntimeError('Upstream VLA configurator changed')
    position=function.body.index(loops[0])
    function.body[position:position+1]=ast.parse("config['skills']={}\nconfig['compositions']=[]").body
    namespace={'__file__':str(source),'__name__':'dsr_rsi_configure'}
    exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),namespace)
    assets=code_asset_root()
    programs=assets/'implementations/code-skills'
    if not allow_vla_only and not (programs/'programs.json').is_file():
        raise FileNotFoundError('Official executable code-skill assets are not prepared')
    # Official code assets require NumPy 1.x and pinned robotics dependencies.
    # Keep these isolated from the independently verified DM0.5 environment.
    config=namespace['configure'](assets,python=interpreter('code'),
        output=output,evidence=evidence,endpoint='local-codex-transport',model='codex-configured-model',
        framework=ROOT/'vendor/XPolicyLab')
    if not allow_vla_only and not config.get('program_registry'):
        raise ValueError('Strict PhysicalRSI evaluation requires the real operation library')
    checkpoint=Path(checkpoint).resolve()
    import yaml
    deployment=yaml.safe_load((ROOT/'vendor/XPolicyLab/policy/DSRRSI/deploy.yml').read_text())
    deployment.update(model_path=str(checkpoint),norm_stats_path=str(checkpoint/'norm_stats.json'),
                      experiment_path=str(ROOT/'vendor/XPolicyLab/policy/OpenDM/scripts/robodojo_dm05_history.py'))
    config['skills']['dm05-mem']={'name':'dm05-mem','implementation':'dm05','configuration':deployment}
    config.setdefault('composition_definitions',{})['memory-guided-dm05']={
        'description':'DM0.5-Mem joint-action inference from current RGB, proprioception, original instruction and causal visual history.',
        'steps':['memory.snapshot','dm05-mem'],
        'capability_memory':{
            'input_contract':'robodojo.observation-batch/v1','output_contract':'robodojo.action-chunks/v1',
            'description':'General learned visual manipulation with 20 causal head-camera history slots at 1 Hz.',
            'limits':['No guaranteed success for an instruction.','Requires the Dual ARX5 joint-action contract.']}}
    config['compositions'].insert(0,'memory-guided-dm05')
    config['model_substitution']={'official_vla':['pi05','pi05-sparse-memory'],
        'replacement_vla':'DM05-MEM-Robodojo-Sim','agent':'Codex from DSR_CODEX_HOME',
        'upstream_commit':json.loads((UPSTREAM/'UPSTREAM_REVISION.json').read_text())['commit'],
        'configurator_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'full_code_library':bool(config.get('program_registry'))}
    temp=Path(output).with_suffix('.tmp');temp.write_text(json.dumps(config,indent=2)+'\n');temp.replace(output)
    return config

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=checkpoint_root()/'DM05-MEM-Robodojo-Sim')
    a=p.parse_args();c=configure(a.output,a.evidence,a.checkpoint)
    print('Registered operations:',len(c['program_registry']))
