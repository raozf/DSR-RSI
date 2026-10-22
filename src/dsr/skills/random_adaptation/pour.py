"""Native random-pour worker using the frozen RGB reader and EE executor."""
import json
import numpy as np
from scipy.spatial.transform import Rotation

from rdj_pour.perception_controller import StructuredModel as ParentPourModel
from rdj_pour.perception import registration, pose as mesh_pose
from rdj_pour.sam.sam3_http import segment_many_http

from .native_contract import FreshNativeMixin
from rdj_pour.controller import bounded_pose, pose_error
from rdj_pour.contract import CLOSED_GRIPPER, OFFICIAL_MAXIMUM_ACTIONS
from copy import deepcopy
from .held_transform import held_transform_update, matrix, rebind_future_stages


class Model(FreshNativeMixin, ParentPourModel):
    """One complete policy: parent plan plus one post-grasp RGB correction.

    Future targets are rebound without changing the parent program. Aborts
    consume fresh observations while retracing its existing upright waypoints.
    """

    def __init__(self, model_cfg):
        if model_cfg.get('task_name') != 'pour_liquid_into_cup_random':
            raise ValueError('random pour descriptor required')
        from .integrity import verify_package
        verify_package(model_cfg)
        super().__init__(model_cfg)
        self._geometry = {}
        self._rechecked = set()
        self._recovery = {}
        self._reset_fresh()

    def reset(self):
        super().reset()
        self._geometry = {}
        self._rechecked = set()
        self._recovery = {}
        self._reset_fresh()

    def initial(self, observation):
        plan = super().initial(observation)
        index = observation['env_idx']
        if plan.get('skill_choice') != 'infeasible':
            record = json.loads((self.out / f'env{index}_geometry.json').read_text())
            self._geometry[index] = record
        return plan

    def _memory_motion_envelopes(self, packet, context):
        for observation in packet:
            index = observation['env_idx']
            if index in self._rechecked or index not in self._geometry:
                continue
            state = self.core.state_for_env(index)
            if state is None or state['stage'] != 'above_mouth' or state['stage_actions'] != 0:
                continue
            self._rechecked.add(index)
            try:
                self._rebind_after_lift(index, observation, state)
            except (ValueError, KeyError, RuntimeError, np.linalg.LinAlgError) as error:
                self._start_recovery(index, observation, before_tilt=True, reason=str(error))
        return super()._memory_motion_envelopes(packet, context)

    def _upright_path(self, index, current):
        arm = self.plans[index]['arm']
        path = [list(s['target'][arm + '_ee_pose']) for s in self.plans[index]['stages']
                if s['name'].startswith('upright_')]
        if not path:
            return []
        # Do not start at full tilt when failure occurred partway into a pour.
        # Angular distance is relative to the actual planned upright pose.
        current_angle = pose_error(current, path[-1])[1]
        eligible = [i for i, p in enumerate(path) if pose_error(p, path[-1])[1] <= current_angle + 1e-6]
        if not eligible:
            return []
        start = min(eligible, key=lambda i: pose_error(current, path[i])[0]
                    + .1 * pose_error(current, path[i])[1])
        path = path[start:]
        # Joining the path and every remaining segment must stay local. This
        # gate cannot certify liquid state, grasp integrity or swept collisions.
        for a, b in zip([current, *path[:-1]], path):
            pos, angle = pose_error(a, b)
            if pos > .020 or angle > .12:
                return []
        return path

    def _start_recovery(self, index, observation, *, before_tilt, reason):
        arm = self.plans[index]['arm']
        current = observation['state'][arm + '_ee_pose']
        if before_tilt:
            lift = next(s for s in self.plans[index]['stages'] if s['name'] == 'lift')
            target = [*current[:3],
                      *lift['target'][arm + '_ee_pose'][3:]]
            path = [target] if pose_error(current, target)[1] <= .05 else []
        else:
            path = self._upright_path(index, current)
        self._recovery[index] = dict(arm=arm, path=path, cursor=0, reason=reason,
            hold=None if path else list(current), best_error=float('inf'), stagnant=0)
        print(json.dumps(dict(event='random_pour_upright_recovery', env=index,
                              reason=reason, before_tilt=before_tilt, waypoints=len(path),
                              status='retracing' if path else 'unqualified_path_hold', native_success=None)), flush=True)

    def _adapt_action(self, index, action):
        observation = self._latest_native[index]
        state = self.core.state_for_env(index)
        if index not in self._recovery and state and state['arm'] in ('left', 'right'):
            tilted = state['stage'].startswith(('tilt', 'upright')) or state['stage'] == 'dwell'
            path = self._upright_path(index, observation['state'][state['arm'] + '_ee_pose']) if tilted else []
            # One acknowledged action per original waypoint plus a small
            # tracking reserve, within the official total action budget.
            reserve = len(path) + 8
            if tilted and (not path or state['failed_hold'] or self._native_counts[index] + reserve >= OFFICIAL_MAXIMUM_ACTIONS):
                self._start_recovery(index, observation, before_tilt=False,
                                     reason='unqualified_path' if not path else 'tracking_failure_or_upright_budget_reserve')
        if index in self._recovery:
            recovery = self._recovery[index]
            result = {k: deepcopy(observation['state'][k]) for k in action}
            arm = recovery['arm']
            current = observation['state'][arm + '_ee_pose']
            path = recovery['path']
            if recovery['hold'] is None:
                target = path[recovery['cursor']]
                pos, angle = pose_error(current, target)
                if pos > .020 or angle > .12:
                    recovery['hold'] = list(current)
                    print(json.dumps(dict(event='random_pour_recovery_off_path', env=index,
                                          waypoint=recovery['cursor'], position_m=pos,
                                          angle_rad=angle, native_success=None)), flush=True)
                elif pos <= .001 and angle <= .008:
                    if recovery['cursor'] == len(path) - 1:
                        recovery['hold'] = list(target)
                    else:
                        recovery['cursor'] += 1
                        recovery['best_error'], recovery['stagnant'] = float('inf'), 0
                        target = path[recovery['cursor']]
                else:
                    error = pos + .1 * angle
                    if error < recovery['best_error'] - .0005:
                        recovery['best_error'], recovery['stagnant'] = error, 0
                    else:
                        recovery['stagnant'] += 1
                    if recovery['stagnant'] >= 20:
                        recovery['hold'] = list(current)
                        print(json.dumps(dict(event='random_pour_recovery_stalled', env=index,
                                              waypoint=recovery['cursor'], native_success=None)), flush=True)
                if recovery['hold'] is None:
                    result[arm + '_ee_pose'] = bounded_pose(current, target)
            if recovery['hold'] is not None:
                result[arm + '_ee_pose'] = list(recovery['hold'])
            result[arm + '_ee_joint_state'] = [CLOSED_GRIPPER]
            return result
        return action

    def _rebind_after_lift(self, index, observation, state):
        from rdj_pour.perception import SHAPES
        plan = self.plans[index]
        if plan.get('skill_choice') == 'infeasible':
            return
        arm = state['arm']
        record = self._geometry[index]
        initial_cup = np.asarray(record['fits']['cup']['pose'], float)
        nominal = np.asarray(plan['_nominal_attachment'], float)
        ee = observation['state'][arm + '_ee_pose']
        expected_bottle = matrix(ee) @ nominal
        rgb = np.asarray(observation['vision']['cam_head']['color'], dtype=np.uint8)[:, :, :3]
        groups = segment_many_http(rgb, [('bottle', .5), ('cup', .5)], timeout_s=600)
        fits = {}
        for label, group in zip(('bottle', 'cup'), groups, strict=True):
            candidates = []
            for mask in group.masks:
                if np.count_nonzero(mask) < 80:
                    continue
                keys = ('bottle',) if label == 'bottle' else tuple(k for k in SHAPES if k.startswith('cup'))
                for key in keys:
                    prior = expected_bottle if label == 'bottle' else initial_cup
                    candidates.append(registration(mask, key, prior=prior))
            if not candidates:
                raise ValueError(label + ' absent after lift')
            fit = min(candidates, key=lambda x: x['error_px'])
            if fit['error_px'] > 4 or fit['iou'] < .45:
                raise ValueError(label + ' RGB registration below threshold')
            fits[label] = fit
        if fits['cup']['key'] != record['fits']['cup']['key']:
            raise ValueError('cup identity changed')
        live = held_transform_update(ee_pose=ee,
                                      observed_object_pose=mesh_pose(fits['bottle']['pose']),
                                      nominal_attachment=nominal)
        cup_delta = fits['cup']['pose'] @ np.linalg.inv(initial_cup)
        cup_distance = float(np.linalg.norm(fits['cup']['pose'][:3, 3] - initial_cup[:3, 3]))
        cup_angle = float(Rotation.from_matrix(cup_delta[:3, :3]).magnitude())
        if cup_distance > .02 or cup_angle > .12:
            raise ValueError('cup motion exceeds local rebind envelope')
        start = next(k for k, s in enumerate(plan['stages']) if s['name'] == 'realign')
        if start <= state['stage_index']:
            raise ValueError('realign stage already consumed')
        corrected = rebind_future_stages(plan['stages'], arm=arm,
                                          first_future_index=start,
                                          nominal_attachment=nominal,
                                          live_attachment=live, cup_delta=cup_delta)
        plan['stages'] = corrected
        (self.out / f'env{index}_post_grasp_random.json').write_text(json.dumps({
            'bottle_error_px': fits['bottle']['error_px'],
            'cup_error_px': fits['cup']['error_px'],
            'cup_delta_m': cup_distance,
            'attachment_delta_m': float(np.linalg.norm(live[:3, 3] - nominal[:3, 3])),
            'rebound_from_stage': start,
            'native_success': None,
        }, indent=2) + '\n')
        print(json.dumps(dict(event='random_pour_rebound', env=index,
                              from_stage=start, cup_delta_m=cup_distance)), flush=True)
