"""Private simulator checkpoint for a same-layout restart, never policy input.

Checkpoint the settled first-attempt scene rather than re-sampling contacts from
the saved layout recipe. Deformable objects retain their native reset path.
"""
from copy import deepcopy
import random
import sys

import numpy as np


def capture_reset_state(env):
    objects = {}
    for kind in ('rigid', 'articulation'):
        for name, obj in env.scene_manager.get_objects(object_type=kind).items():
            row = dict(pose=deepcopy(obj.get_local_pose()),
                       linear_velocity=deepcopy(obj.get_linear_velocity()),
                       angular_velocity=deepcopy(obj.get_angular_velocity()))
            if kind == 'articulation':
                row.update(joints=deepcopy(obj.get_current_joint_positions()),
                           joint_velocities=deepcopy(obj.get_joint_velocities()),
                           drive_targets=deepcopy(obj._capture_drive_targets()))
            objects[name] = row
    robots = [dict(root=key.data.root_state_w.clone(),
                   joints=key.data.joint_pos.clone(), velocities=key.data.joint_vel.clone(),
                   position_target=key.data.joint_pos_target.clone(),
                   velocity_target=key.data.joint_vel_target.clone())
              for key in env.robot_manager.robot_key]
    torch = sys.modules.get('torch')
    return dict(seeds=list(env.env_seeds), objects=objects, robots=robots,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state() if torch is not None else None,
                cuda_rng=torch.cuda.get_rng_state_all() if torch is not None and torch.cuda.is_available() else None)


def restore_reset_state(env, state, *, rng=False):
    if list(env.env_seeds) != state['seeds']:
        raise ValueError('Checkpoint layout seeds do not match this restart')
    objects = env.scene_manager.get_objects()
    if set(state['objects']) - set(objects):
        raise ValueError('Checkpoint object identities changed during restart')
    for name, row in state['objects'].items():
        obj = objects[name]
        obj.set_local_pose(*deepcopy(row['pose']))
        obj.set_linear_velocity(deepcopy(row['linear_velocity']))
        obj.set_angular_velocity(deepcopy(row['angular_velocity']))
        if 'joints' in row:
            obj.set_current_joint_positions(deepcopy(row['joints']))
            obj.set_joint_velocities(deepcopy(row['joint_velocities']))
            obj.initial_drive_targets = deepcopy(row['drive_targets'])
            obj.reset_drive_targets()
    if len(env.robot_manager.robot_key) != len(state['robots']):
        raise ValueError('Checkpoint robot identities changed during restart')
    for key, row in zip(env.robot_manager.robot_key, state['robots']):
        key.write_root_state_to_sim(row['root'].clone())
        key.write_joint_state_to_sim(row['joints'].clone(), row['velocities'].clone())
        key.set_joint_position_target(row['position_target'].clone())
        key.set_joint_velocity_target(row['velocity_target'].clone())
    if rng:
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch = sys.modules.get('torch')
        if torch is not None and state['torch_rng'] is not None:
            torch.set_rng_state(state['torch_rng'])
        if torch is not None and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
