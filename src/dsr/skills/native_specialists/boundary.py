"""Fresh indexed RGB/proprioception boundary around unchanged native workers."""
from copy import deepcopy
from collections.abc import Mapping
from numbers import Real

import numpy as np


STATE_FIELDS = frozenset(a+s for a in ('left','right') for s in
    ('_arm_joint_state','_ee_joint_state','_ee_pose','_tcp_pose','_delta_ee_pose'))
CAMERA_FIELDS = frozenset(('color','intrinsic_matrix','extrinsic_matrix','shape'))
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def _numbers(value, shape, label):
    """Check before casting so strings and mixed boolean lists cannot pass."""
    try:
        raw = np.asarray(value, dtype=object)
        if raw.shape != shape or any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) for v in raw.flat):
            raise ValueError(f'{label} requires numeric shape {shape}')
        result = np.asarray(value, dtype=float)
    except (TypeError, OverflowError) as error:
        raise ValueError(f'{label} requires numeric shape {shape}') from error
    if not np.isfinite(result).all():
        raise ValueError(f'{label} requires finite numbers')
    return result


def _camera(values, camera):
    if not isinstance(values, Mapping) or 'color' not in values:
        raise ValueError(f'{camera} RGB required')
    rgb = np.asarray(values['color'])
    # Frozen RGB registration uses this resolution and camera calibration.
    if rgb.dtype != np.uint8 or rgb.shape not in ((480, 640, 3), (480, 640, 4)):
        raise ValueError(f'{camera} requires 480x640 uint8 RGB or RGBA')
    if 'shape' in values:
        shape = np.asarray(values['shape'], dtype=object)
        if (shape.shape not in ((2,), (3,)) or
                any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in shape) or
                tuple(shape) != rgb.shape[:len(shape)]):
            raise ValueError(f'{camera} shape metadata disagrees with RGB')
    if 'intrinsic_matrix' in values:
        k = _numbers(values['intrinsic_matrix'], (3, 3), camera + ' intrinsic_matrix')
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-6):
            raise ValueError(f'{camera} invalid intrinsic matrix')
    if 'extrinsic_matrix' in values:
        t = _numbers(values['extrinsic_matrix'], (4, 4), camera + ' extrinsic_matrix')
        if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6) or
                not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or
                np.linalg.det(t[:3, :3]) < .999):
            raise ValueError(f'{camera} invalid rigid extrinsic matrix')
    return {k: deepcopy(v) for k, v in values.items() if k in CAMERA_FIELDS}


def clean_batch(observations):
    if isinstance(observations, (Mapping, str, bytes)):
        raise ValueError('observation batch required')
    try:
        rows = list(observations)
    except TypeError as error:
        raise ValueError('observation batch required') from error
    if any(not isinstance(o, Mapping) for o in rows):
        raise ValueError('observation rows must be mappings')
    indices = [o.get('env_idx') for o in rows]
    if not rows or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('unique nonnegative integer environment indices required')
    clean=[]
    for o in rows:
        if (not isinstance(o.get('vision'), Mapping) or not isinstance(o.get('state'), Mapping) or
                not isinstance(o.get('instruction'), str) or not o['instruction'].strip()):
            raise ValueError('RGB/proprioception/instruction observation required')
        if 'cam_head' not in o['vision']:
            raise ValueError('cam_head RGB required')
        vision = {camera: _camera(o['vision'][camera], camera) for camera in CAMERAS if camera in o['vision']}
        for arm in ('left', 'right'):
            for suffix, size in (('_ee_pose', 7), ('_arm_joint_state', 6), ('_ee_joint_state', 1),
                                 ('_tcp_pose', 7), ('_delta_ee_pose', 7)):
                key = arm + suffix
                if suffix in ('_tcp_pose', '_delta_ee_pose') and key not in o['state']:
                    continue
                values = _numbers(o['state'].get(key), (size,), key)
                if suffix in ('_ee_pose', '_tcp_pose') and abs(np.linalg.norm(values[3:]) - 1) > .001:
                    raise ValueError(f'{key} requires a unit wxyz quaternion')
        state={k:deepcopy(v) for k,v in o['state'].items() if k in STATE_FIELDS}
        clean.append(dict(env_idx=o['env_idx'],vision=vision,state=state,instruction=o['instruction']))
    return clean


class FreshNativeMixin:
    def _reset_fresh(self):
        self._fresh_indices = ()
        self._latest_native = {}
        self._native_counts = {}

    def update_obs_batch(self, observations):
        self._fresh_indices = ()
        self._latest_native = {}
        rows = clean_batch(observations)
        self._latest_native = {o['env_idx']: o for o in rows}
        result = super().update_obs_batch(rows)
        self._fresh_indices = tuple(o['env_idx'] for o in rows)
        return result

    def update_obs(self, observation):
        return self.update_obs_batch([dict(observation, env_idx=0) if isinstance(observation, Mapping) else observation])

    def get_action_batch(self, env_idx_list=None):
        indices = self._fresh_indices if env_idx_list is None else tuple(env_idx_list)
        if (not indices or any(type(i) is not int for i in indices) or len(set(indices)) != len(indices) or
                any(i not in self._fresh_indices for i in indices)):
            raise ValueError('requested environments require fresh observations')
        # The retained pour executor consumes the whole batch on emission.
        self._fresh_indices = ()
        actions = super().get_action_batch(indices)
        for index, chunk in zip(indices, actions, strict=True):
            self._native_counts[index] = self._native_counts.get(index, 0) + len(chunk)
            for n, action in enumerate(chunk):
                chunk[n] = self._adapt_action(index, action)
        return actions

    def _adapt_action(self, index, action):
        return action

    def get_action(self):
        return self.get_action_batch([0])[0]
