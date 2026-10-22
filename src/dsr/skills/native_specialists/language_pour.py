"""Native batch adapter retaining the exact frozen PhysicalRSI pour core."""
from copy import deepcopy
import hashlib
import importlib
from pathlib import Path

import numpy as np

from .boundary import clean_batch
from .pour_control import LanguagePourMixin, JOINT_FIELDS


class Model:
    def __init__(self, cfg):
        from .integrity import verify
        verify(cfg)
        if cfg['task_name'] != 'pour_by_language' or cfg['action_type'] != 'joint':
            raise ValueError('exact language pour joint descriptor required')
        parent = importlib.import_module('model')
        identity = cfg['preserved_pour_core']
        source = Path(parent.__file__).resolve(strict=True)
        if str(source) != identity['path'] or hashlib.sha256(source.read_bytes()).hexdigest() != identity['sha256']:
            raise ValueError('preserved pour core resolved to a different source')
        if not all(hasattr(parent.Model, method) for method in
                   ('_memory_motion_visual_gates', '_memory_motion_commit', 'register_visible_body')):
            raise ValueError('preserved structured pour stages unavailable')
        self.episode_type = type('LanguagePourEpisode', (LanguagePourMixin, parent.Model), {})
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.episodes = {}; self.fresh = (); self.observations = {}

    def update_obs_batch(self, observations):
        self.fresh = (); self.observations = {}
        rows = clean_batch(observations)
        for obs in rows:
            index = obs['env_idx']
            if index not in self.episodes:
                self.episodes[index] = self.episode_type(self.cfg)
            self.episodes[index].update_obs(obs)
            self.observations[index] = obs
        self.fresh = tuple(o['env_idx'] for o in rows)

    def get_action_batch(self, indices=None):
        indices = self.fresh if indices is None else tuple(indices)
        if (not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or i not in self.fresh for i in indices)):
            raise ValueError('fresh unique environment observations required')
        self.fresh = tuple(i for i in self.fresh if i not in indices)
        result = []
        for index in indices:
            episode = self.episodes[index]
            try:
                chunk = episode.get_action()
                if len(chunk) != 1 or set(chunk[0]) != set(JOINT_FIELDS):
                    raise ValueError('one bounded joint action per observation required')
                for key, value in chunk[0].items():
                    values = np.asarray(value, float)
                    n = 1 if '_ee_joint' in key else 6
                    if values.shape != (n,) or not np.isfinite(values).all():
                        raise ValueError('invalid joint action')
                    if n == 1 and not 0 <= values[0] <= 1:
                        raise ValueError('invalid gripper action')
            except (ValueError, KeyError) as error:
                if episode.run is not None:
                    episode.reject('language_pour_adapter', str(error))
                chunk = [{k:deepcopy(self.observations[index]['state'][k]) for k in JOINT_FIELDS}]
            result.append(chunk)
        return result

    def update_obs(self, observation):
        return self.update_obs_batch([dict(observation, env_idx=0)])

    def get_action(self):
        return self.get_action_batch([0])[0]
