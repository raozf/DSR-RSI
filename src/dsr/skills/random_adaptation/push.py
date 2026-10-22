"""Native joint Push-T policy with measured, bounded local response updates."""
from copy import deepcopy
import json
import numpy as np
from scipy.spatial.transform import Rotation

from rdj_push_submission.model import Model as ParentPushModel
from rdj_push_submission.planning import refresh_push_t_correction, _ee_pose_to_semantic
from .native_contract import FreshNativeMixin
from rdj_push_submission.geometry import _offset_local_x
from .push_response import PushResponseModel
from .held_transform import matrix, pose


SUPPORT_LOSS_Z_M = .010
SUPPORT_TILT_RAD = np.deg2rad(8)


def yaw(p):
    return float(np.arctan2(matrix(p)[1, 0], matrix(p)[0, 0]))


def angle_delta(a, b):
    return float(np.arctan2(np.sin(b - a), np.cos(b - a)))


def planar_delta(before, after, target_yaw, radius=.05):
    r = Rotation.from_euler('z', -target_yaw)
    xy = r.apply([*(np.asarray(after[:2]) - before[:2]), 0])[:2]
    return np.r_[xy, radius * angle_delta(yaw(before), yaw(after))]


class Model(FreshNativeMixin, ParentPushModel):
    def __init__(self, model_cfg):
        if model_cfg.get('task_name') != 'push_T_random':
            raise ValueError('random Push-T descriptor required')
        from .integrity import verify_package
        verify_package(model_cfg)
        super().__init__(model_cfg)

    def reset(self):
        super().reset()
        self.responses, self.samples, self.support_origins, self.stopped = {}, {}, {}, set()
        self._reset_fresh()

    def _refresh(self, fn, index, observation, *args):
        if index in self.stopped:
            return None
        if fn is not refresh_push_t_correction:
            self.responses.pop(index, None)
            self.samples.pop(index, None)
            self.support_origins.pop(index, None)
            return super()._refresh(fn, index, observation, *args)
        correction, object_pose = args
        context = self.contexts[index]
        if context is None:
            return super()._refresh(fn, index, observation, *args)
        arm = context['arm']
        edge = f"{arm}:{context['mode']}:{context.get('closing_sign', context.get('target_sign'))}"
        true_target = list(context['target_pose'])
        ee = _ee_pose_to_semantic(self.delta, self.bias, observation['state'][arm + '_ee_pose'])
        target_yaw = yaw(true_target)
        response = self.responses.setdefault(index, PushResponseModel())
        reason = 'first_contact_sample'
        virtual = true_target
        world_z = float(observation['state'][arm + '_ee_pose'][2])
        origin = self.support_origins.setdefault(index, world_z)
        world_z_drift = world_z - origin
        object_z_drift = float(object_pose[2]) - float(context['initial_t_z'])
        if world_z_drift > SUPPORT_LOSS_Z_M or object_z_drift > SUPPORT_LOSS_Z_M:
            self._stop_corrections(index, observation, correction, ee)
            print(json.dumps(dict(event='random_push_support_z_drift', env=index,
                                  ee_drift_m=world_z_drift, rgb_object_drift_m=object_z_drift,
                                  threshold_m=SUPPORT_LOSS_Z_M, native_success=None)), flush=True)
            return None
        if index in self.samples:
            previous = self.samples[index]
            if previous['edge'] == edge and abs(angle_delta(previous['target_yaw'], target_yaw)) < .04 and np.linalg.norm(np.asarray(previous['target_xy']) - true_target[:2]) < .004:
                # Infer rigid object-root motion from the observed attachment;
                # EEF XY alone misattributes rotation of an offset grasp.
                predicted = matrix(ee) @ np.linalg.inv(matrix(previous['ee'])) @ matrix(previous['object'])
                support_tilt = float(np.arccos(np.clip(predicted[2, 2], -1, 1)))
                if support_tilt > SUPPORT_TILT_RAD:
                    self._stop_corrections(index, observation, correction, ee)
                    print(json.dumps(dict(event='random_push_support_tilt', env=index,
                                          inferred_tilt_rad=support_tilt, native_success=None)), flush=True)
                    return None
                command = planar_delta(previous['object'], pose(predicted), target_yaw)
                actual = planar_delta(previous['object'], object_pose, target_yaw)
                reason = response.observe(edge=edge, command_delta=command, object_delta=actual,
                                          support_tilt_rad=support_tilt)
                if reason == 'updated':
                    desired = planar_delta(object_pose, true_target, target_yaw)
                    adapted = response.predict_command(desired, edge=edge, max_command=.018)
                    if adapted is not None:
                        adapted = np.asarray(adapted)
                        world = Rotation.from_euler('z', target_yaw).apply([*adapted[:2], 0])
                        # Radius-scaled yaw gives translation and rotation a
                        # common metric; bounded changes preserve support Z.
                        turn = np.clip(adapted[2] / .05, -.15, .15)
                        q = Rotation.from_euler('z', yaw(object_pose) + turn).as_quat()[[3, 0, 1, 2]]
                        virtual = [object_pose[0] + world[0], object_pose[1] + world[1], true_target[2], *q.tolist()]
                elif reason == 'change_contact_or_stop':
                    # Parent supports no qualified alternate edge at this
                    # stage. Withdraw through its existing release/home path.
                    self._stop_corrections(index, observation, correction, ee)
                    return None
            else:
                self.responses[index] = response = PushResponseModel()
                reason = 'contact_or_target_changed'
        self.samples[index] = dict(edge=edge, ee=list(ee), object=list(object_pose),
                                   target_yaw=target_yaw, target_xy=true_target[:2])
        self.contexts[index] = deepcopy(context)
        self.contexts[index]['target_pose'] = virtual
        try:
            result = super()._refresh(fn, index, observation, correction, object_pose)
        finally:
            self.contexts[index]['target_pose'] = true_target
        print(json.dumps(dict(event='random_push_response', env=index,
                              correction=correction, reason=reason, gain=response.gain,
                              true_target=true_target, virtual_target=virtual,
                              native_success=None)), flush=True)
        return result

    def _stop_corrections(self, index, observation, correction, semantic):
        context = self.contexts[index]
        arm = context['arm']
        plan = deepcopy(self.plans[index])
        seed = list(observation['state'][arm + '_arm_joint_state'])
        retreat = self.solve(arm, _offset_local_x(semantic, -.080), seed)
        prefix = 'target_' if context['mode'] == 'relay' else ''
        for stage in plan['stages']:
            name = stage['name']
            remaining = name.startswith(prefix + 'correct_') and int(name.rsplit('_', 1)[1]) >= correction
            if remaining or name == prefix + 'release':
                stage['target'][arm + '_arm_joint_state'] = list(seed)
            elif name == prefix + 'retreat':
                stage['target'][arm + '_arm_joint_state'] = list(retreat or seed)
        self.plans[index] = plan
        self.stopped.add(index)
        print(json.dumps(dict(event='random_push_no_response_stop', env=index,
                              retreat_ik_available=retreat is not None, native_success=None)), flush=True)
