"""Native observation-batch -> joint-action-chunks isolated specialist worker."""
from collections.abc import Mapping
from copy import deepcopy
import importlib
import os
from pathlib import Path
import uuid
from .boundary import clean_batch
from .catalog import SPECS,limits
from .control import PolicyFailure
from .episode import Episode
from .scene import SemanticScene
from .joint_port import JointPort


class Model:
    def __init__(self,cfg):
        from .integrity import verify
        verify(cfg)
        self.task=cfg['task_name'];self.kind=SPECS[self.task][0]
        if self.kind in ('digits','dolls') or cfg['action_type']!='joint' or cfg['env_cfg_type']!='arx_x5':
            raise ValueError('semantic specialist joint descriptor required')
        module=importlib.import_module('.policies.'+self.kind,__package__)
        self.module=module
        self.policy=module.policy
        self.strategy_settings=cfg.get('execution_strategy')
        self.revision=cfg['specialist_package_sha256']
        if self.strategy_settings and cfg.get('stage_composition'):
            raise ValueError('Legacy staged and agent-selected execution contracts cannot be mixed')
        if cfg.get('stage_composition'):
            from .handoff import composed,PAIRS
            if cfg['stage_composition']!=list(PAIRS[self.kind][:2]):
                raise ValueError('Unreviewed stage composition')
            self.policy=composed(self.kind,module)
        self.output=Path(os.environ['RDJ_OUTPUT_DIR']);self.output.mkdir(parents=True,exist_ok=True)
        self.port=JointPort();self.reset()

    def reset(self):self.episodes={};self.indices=();self.consumed=set();self.latest={}

    def update_obs_batch(self,observations):
        self.indices=();self.consumed=set();self.latest={}
        rows=clean_batch(observations)
        for obs in rows:
            index=obs['env_idx']
            if index not in self.episodes:
                memory=limits(self.task)
                log=self.output/(f'env{index}-'+uuid.uuid4().hex+'.jsonl')
                strategy=agent=None
                startup=lambda robot,context:self.policy(robot,robot.memory)
                if self.strategy_settings:
                    from .strategy import supervisor
                    if __package__ == 'dsr.skills.native_specialists':
                        from dsr.system2.strategy_agent import StrategyAgent
                    else:
                        from .strategy_agent import StrategyAgent
                    strategy=supervisor(self.module,self.revision,self.strategy_settings)
                    agent=StrategyAgent(log.with_suffix('.agent'))
                    startup=strategy.policy
                self.episodes[index]=Episode(self.task,memory,SemanticScene(self.kind),
                    log,startup=startup,strategy=strategy,strategy_agent=agent)
            self.episodes[index].observe(obs);self.latest[index]=obs
        self.indices=tuple(o['env_idx'] for o in rows)

    def update_obs(self,obs):return self.update_obs_batch([dict(obs,env_idx=0) if isinstance(obs,Mapping) else obs])

    def get_action_batch(self,indices=None):
        indices=self.indices if indices is None else tuple(indices)
        if not indices or any(type(i) is not int or i not in self.indices or i in self.consumed for i in indices) or len(set(indices))!=len(indices):
            raise ValueError('fresh unique environment observations required')
        self.consumed.update(indices);actions=[]
        for index in indices:
            episode=self.episodes[index];action=episode.action();state=self.latest[index]['state']
            try:joint=self.port.convert(action,state)
            except PolicyFailure as error:
                episode.reject_action(error)
                joint={a+s:list(state[a+s]) for a in ('left','right') for s in ('_arm_joint_state','_ee_joint_state')}
            actions.append([joint])
        return actions

    def get_action(self):return self.get_action_batch([0])[0]
