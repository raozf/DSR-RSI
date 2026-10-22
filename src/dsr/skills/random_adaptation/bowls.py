"""Descriptor-compatible EE worker for a single complete random bowl policy."""
from copy import deepcopy
from collections.abc import Mapping
import json
import os
from pathlib import Path
import uuid

from .bowl_reader import RGBBowlScene
from .bowl_policy import policy
from .native_contract import clean_batch
from .support.episode import Episode


class Model:
    def __init__(self,model_cfg):
        if model_cfg['task_name']!='stack_bowls_random' or model_cfg['action_type']!='ee' or model_cfg['env_cfg_type']!='arx_x5':
            raise ValueError('random bowls require arx_x5 EE descriptor')
        from .integrity import verify_package
        root=verify_package(model_cfg)
        self.memory=json.loads((root/'bowl.memory.json').read_text())
        self.bank=root/'public_bank'
        self.output=Path(os.environ['RDJ_OUTPUT_DIR']);self.output.mkdir(parents=True,exist_ok=True)
        self.revision=model_cfg['random_controller_revision']
        self.reset()

    def reset(self):
        self.episodes={};self.indices=();self.consumed=set()

    def update_obs_batch(self,observations):
        self.indices=();self.consumed=set()
        rows=clean_batch(observations)
        for o in rows:
            index=o['env_idx']
            if index not in self.episodes:
                perception=RGBBowlScene(self.bank)
                ep=Episode('bowl-profile-nesting-v3',deepcopy(self.memory),perception,
                           self.output/(f'{index}-'+uuid.uuid4().hex+'.jsonl'),
                           startup=lambda robot,context:policy(robot,deepcopy(self.memory)))
                self.episodes[index]=ep
            self.episodes[index].observe(o)
        self.indices=tuple(o['env_idx'] for o in rows)

    def update_obs(self,observation):
        return self.update_obs_batch([dict(observation,env_idx=0) if isinstance(observation,Mapping) else observation])

    def get_action_batch(self,indices=None):
        indices=self.indices if indices is None else tuple(indices)
        if not indices or any(type(i) is not int for i in indices) or len(set(indices))!=len(indices) or any(i not in self.indices or i in self.consumed for i in indices):
            raise ValueError('requested environments require fresh observation')
        self.consumed.update(indices)
        return [[self.episodes[i].action()] for i in indices]

    def get_action(self):return self.get_action_batch([0])[0]
