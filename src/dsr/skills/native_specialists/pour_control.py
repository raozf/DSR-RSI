"""Incremental controls around the preserved PhysicalRSI language pour policy.

Only RGB/proprioception observations can advance this ledger. It deliberately
does not interpret upright bottle observations as a fluid success certificate.
Frame numbers are action observations, not invented simulation timestamps.
"""
from copy import deepcopy
from itertools import groupby
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .analytical_skills import pivot_end_effector, pour_order, reserve_budget


SPOUT = np.array([0., 0., .2181])  # Retained public bottle profile, not scene GT.
LIMIT = 800
JOINT_FIELDS = tuple(a+s for a in ('left', 'right')
                     for s in ('_arm_joint_state', '_ee_joint_state'))


def matrix(p):
    p = np.asarray(p, float)
    if p.shape != (7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:])-1) > .001:
        raise ValueError('finite unit-quaternion EEF pose required')
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
    m[:3, 3] = p[:3]
    return m


def pose(m):
    return [*m[:3, 3], *Rotation.from_matrix(m[:3, :3]).as_quat()[[3, 0, 1, 2]]]


def rebind_spout(ideal_ee, nominal_attachment, measured_attachment):
    """Preserve the parent's planned bottle/spout path after an observed slip."""
    bottle = matrix(ideal_ee) @ nominal_attachment
    point = bottle[:3, 3] + bottle[:3, :3] @ SPOUT
    proposal = pivot_end_effector(feature_world_m=point, feature_in_object_m=SPOUT,
        object_rotation=bottle[:3, :3], object_in_gripper_rotation=measured_attachment[:3, :3],
        object_in_gripper_translation_m=measured_attachment[:3, 3])
    m = np.eye(4); m[:3, :3] = proposal['rotation']; m[:3, 3] = proposal['position_m']
    return pose(m)


def compact_holds(actions):
    """Use a bounded subsequence of the parent path and retain fluid dwell.

    The parent's convergence gate still requires measured EEF arrival at each
    segment. Consecutive tilt/return waypoints must stay within 20 mm/0.12 rad.
    Every retained target was qualified by the parent. This is still only the
    parent's endpoint/path coverage, not a new collision certification.
    """
    result = []
    previous = None
    for name, group in groupby(actions, lambda x: x[0]):
        rows = list(group)
        if name.startswith(('tilt_', 'upright_')) and int(name.rsplit('_', 1)[1]) % 2 and not name.endswith('_59'):
            continue
        if name.startswith(('above_', 'tilt_', 'dwell', 'upright_')):
            if previous is not None:
                for arm in ('left', 'right'):
                    a, b = (matrix(r[arm+'_ee_pose']) for r in (previous, rows[0][1]))
                    if (np.linalg.norm(a[:3, 3]-b[:3, 3]) > .020
                            or Rotation.from_matrix(a[:3, :3].T@b[:3, :3]).magnitude() > .12):
                        raise ValueError('compacted spout path exceeds local step bounds')
            previous = rows[-1][1]
        keep = {'close': 8, 'release': 8, 'home': 3}.get(name, len(rows))
        result.extend(deepcopy(rows[:keep]))
    return result


def minimum_steps(actions):
    """Optimistic command lower bound, with mandatory fluid/gripper dwell."""
    return sum(len(list(group)) if name in {'close', 'release', 'home', 'dwell'} else 1
               for name, group in groupby(actions, lambda x: x[0]))


class PourLedger:
    def __init__(self):
        self.order = None
        self.rows = []
        self.last_frame = -1

    def bind(self, instruction, pair):
        order = pour_order(instruction)
        if self.order is not None and self.order != order:
            raise ValueError('language order changed within episode')
        if pair != len(self.rows) or pair > 2:
            raise ValueError('pair order or duplicate pair')
        if self.rows and not self.rows[-1]['complete']:
            raise ValueError('previous upright/release/home cycle unconfirmed')
        self.order = order
        self.rows.append(dict(pair=pair, colors=order[pair], initial_upright=False,
            observed_tilt=False, upright_frames=0, released_frames=0,
            home_frames=0, complete=False, native_fluid_success=None))

    def observe(self, *, frame, pair, angle_rad, released, home):
        if type(frame) is not int or frame <= self.last_frame or pair != len(self.rows)-1:
            raise ValueError('fresh same-pair observation required')
        if type(released) is not bool or type(home) is not bool:
            raise ValueError('explicit released and home observations required')
        self.last_frame = frame
        row = self.rows[pair]
        visible = angle_rad is not None
        if visible and (not math.isfinite(angle_rad) or not 0 <= angle_rad <= math.pi):
            raise ValueError('invalid observed bottle axis')
        # Margin around native 30 degrees; do not count a noisy threshold flicker.
        upright = visible and angle_rad < math.radians(24)
        tilted = visible and angle_rad > math.radians(36)
        if not row['initial_upright']:
            if upright and not released: row['initial_upright'] = True
            elif tilted: raise ValueError('initial upright state unobserved')
        elif tilted:
            if row['released_frames']: raise ValueError('released bottle tipped')
            row['observed_tilt'] = True
            row['upright_frames'] = 0
        if row['observed_tilt']:
            row['upright_frames'] = row['upright_frames']+1 if upright else 0
        row['released_frames'] = row['released_frames']+1 if released and upright and row['observed_tilt'] else 0
        row['home_frames'] = row['home_frames']+1 if home and row['released_frames'] >= 3 else 0
        row['complete'] = (row['initial_upright'] and row['observed_tilt']
                           and row['upright_frames'] >= 3 and row['released_frames'] >= 3
                           and row['home_frames'] >= 3)
        return deepcopy(row)


class LanguagePourMixin:
    """Use the parent's structured stages, shape reader, IK and setdown gates."""
    def reset(self):
        super().reset()
        self.pour_ledger = PourLedger()
        self._measurement = None
        self._upright_wait = 0
        self._terminal_wait = 0
        self._last_probe_t = -1
        self._released_world = self._released_pose = None
        self._initial_wait = 0
        self._abort_path = None
        self._abort_wait = 0

    def review_event(self, **event):
        with (self.run/'language_cycle.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(t=self.t, pair=self.pair_index,
                native_success=None) | event, allow_nan=False)+'\n')

    def reject(self, reason, evidence):
        # Preserve the already qualified return path before the parent's
        # rejection handler clears its action list. A failed task stays failed.
        if self.actions and self.cursor < len(self.actions) and self._abort_path is None:
            name = self.actions[self.cursor][0]
            if name.startswith(('tilt_', 'upright_', 'return_')) or name == 'dwell':
                self._begin_abort(reason)
        return super().reject(reason, evidence)

    def build_pair(self):
        try:
            self.pour_ledger.bind(self.obs['instruction'], self.pair_index)
        except ValueError as error:
            self.reject('language_or_cycle_contract', str(error)); return
        super().build_pair()
        if self.failure: return
        try:
            self.actions = compact_holds(self.actions)
        except ValueError as error:
            self.reject('unbounded_compact_path', str(error)); return
        self.ideal_pour_actions = deepcopy(self.actions)
        self._measurement = None
        self._upright_wait = self._terminal_wait = 0
        self._initial_wait = 0
        self._released_world = self._released_pose = None
        # Count the parent's exact trajectory before moving. This lower bound
        # cannot guarantee future convergence; the runtime checks it again.
        floor = minimum_steps(self.actions)
        try:
            reserve_budget(limit=LIMIT, used=self.t, proposed=floor,
                finish_reserve=(2-self.pair_index)*floor, observation_reserve=24)
        except ValueError as error:
            self.reject('three_pair_budget_infeasible', dict(reason=str(error),
                minimum_per_pair=floor, remaining_pairs=3-self.pair_index)); return
        self.review_event(event='bound_parent_plan', instruction_order=self.pour_ledger.order,
            nominal_commands=len(self.actions), minimum_commands=floor,
            preserved=['semantic_RGB', 'neck_grasp', 'full_chain_IK', 'finger_clearance',
                       'attachment_gates', 'setdown', 'qualified_waypoint_subsequence', 'fluid_dwell'],
            analytical_skills=['pour_order', 'reserve_budget'])

    def measure_bottle(self, name):
        from axis_registration import qualified_axis_fit
        from hue_identity import segment
        from model import project
        predicted = (self._released_world if self._released_world is not None
                     else matrix(self.actual()) @ self.active_attachment)
        center = project(predicted[:3, 3] + predicted[:3, :3] @ (SPOUT/2))
        radius = 1.5*np.linalg.norm(np.asarray(self.initial_bottle['box'][2:])-self.initial_bottle['box'][:2])
        _, mask = segment(self.rgb('cam_head'), self.wrist_appearance,
                          center, radius, 3*self.initial_bottle['area'])
        fit = qualified_axis_fit(mask, predicted) if mask is not None else dict(accepted=False, reason='unobserved')
        if fit['accepted'] and self._released_world is not None:
            self._released_world = np.asarray(fit['measured'])
        self._measurement = dict(t=self.t, stage=name, fit=fit)
        self.review_event(event='bottle_axis', **self._measurement)
        return np.asarray(fit['measured']) if fit['accepted'] else None

    def register_visible_body(self, name):
        # Overrides only the parent's RGB registration/rebind stage. All its
        # grasp, path qualification, attachment and setdown gates still run.
        from model import get_ik
        from geometry import finger_clearance
        measured = self.measure_bottle(name)
        if measured is None: return
        held = np.linalg.inv(matrix(self.actual())) @ measured
        seed = np.asarray(self.obs['state'][self.arm+'_arm_joint_state'])
        updated = deepcopy(self.actions); cache = {}; valid = True
        for i in range(self.cursor, len(updated)):
            stage, ideal = self.ideal_pour_actions[i]
            if stage.split('_')[0] not in {'tilt', 'dwell', 'upright'}: continue
            target = rebind_spout(ideal[self.arm+'_ee_pose'], self.observed_attachment, held)
            key = tuple(np.round(target, 6))
            if key not in cache:
                gap = finger_clearance(matrix(target))
                q = get_ik().solve(self.arm, seed, target) if gap >= .005 else None
                if q is None: valid = False; break
                cache[key] = q; seed = np.asarray(q)
            updated[i][1][self.arm+'_ee_pose'] = target
            updated[i][1][self.arm+'_arm_joint_state'] = cache[key]
        if valid:
            self.actions = updated; self.active_attachment = held
        self.review_event(event='spout_rebind', applied=valid, targets=len(cache),
                          analytical_skill='pivot_end_effector')

    def _home_measured(self):
        return all(np.linalg.norm(matrix(self.obs['state'][a+'_ee_pose'])[:3, 3]-matrix(self.home[a+'_ee_pose'])[:3, 3]) < .005
            and Rotation.from_matrix(matrix(self.home[a+'_ee_pose'])[:3, :3].T @ matrix(self.obs['state'][a+'_ee_pose'])[:3, :3]).magnitude() < .04
            for a in ('left', 'right'))

    def _memory_motion_visual_gates(self, packet, context):
        record = super()._memory_motion_visual_gates(packet, context)
        if self.failure: return record
        name = record['name']; family = name.split('_')[0]
        cycle = self.pour_ledger.rows[self.pair_index]
        required = (name == 'above_mouth' or family in {'return', 'place', 'release', 'home'}
                    or family == 'upright' and name.endswith(('_58', '_59')))
        periodic = family in {'tilt', 'dwell'} and self.t-self._last_probe_t >= 10
        probe = required or periodic
        if probe:
            self._last_probe_t = self.t
            if family == 'release' and self._released_world is None:
                self._released_world = matrix(self.actual()) @ self.active_attachment
            if not self._measurement or self._measurement['t'] != self.t:
                measured = self.measure_bottle(name)
            else:
                fit = self._measurement['fit']
                measured = np.asarray(fit['measured']) if fit['accepted'] else None
            angle = None if measured is None else float(np.arccos(np.clip(measured[2, 2], -1, 1)))
            released = family in {'release', 'home'} and all(
                float(self.obs['state'][a+'_ee_joint_state'][0]) >= .9 for a in ('left', 'right'))
            # The simulator gripper field can be command echo: release also
            # needs a stable RGB bottle across fresh frames below.
            if released and measured is not None:
                old = getattr(self, '_released_pose', None)
                released = old is not None and np.linalg.norm(old[:3, 3]-measured[:3, 3]) < .004
                self._released_pose = measured.copy()
            else: self._released_pose = None
            cycle = self.pour_ledger.observe(frame=self.t, pair=self.pair_index,
                angle_rad=angle, released=released, home=family == 'home' and self._home_measured())
            self.review_event(event='cycle_observation', ledger=cycle)
            if name == 'above_mouth' and not cycle['initial_upright']:
                record['advance'] = False; self._initial_wait += 1
                if self._initial_wait > 12:
                    self.reject('initial_upright_unconfirmed', cycle)
            if family == 'return' and cycle['upright_frames'] < 3:
                # Keep the last upright target until fresh evidence arrives.
                last = next((a for n, a in reversed(self.actions[:self.cursor]) if n.startswith('upright_')), None)
                if last is None: raise ValueError('upright path absent')
                record.update(a=deepcopy(last), advance=False)
                self._upright_wait += 1
                if self._upright_wait > 12:
                    self.reject('upright_cycle_unconfirmed', cycle)
            if family == 'home' and self.cursor+1 == len(self.actions) and not cycle['complete']:
                record['advance'] = False; self._terminal_wait += 1
                if self._terminal_wait > 12:
                    self.reject('released_home_cycle_unconfirmed', cycle)
        # Do not execute a later pair without observing its predecessor cycle.
        if self.failure:
            record.update(name='failure_hold', a=deepcopy(self.hold), advance=False)
        return record

    def _begin_abort(self, reason):
        """Retrace only an adjacent, already qualified upright path. No reset."""
        current = matrix(self.actual())
        rows = [(name, action) for name, action in self.actions if name.startswith('upright_')]
        unique = [(name, list(group)[0][1]) for name, group in groupby(rows, lambda x:x[0])]
        eligible = []
        for i, (_, action) in enumerate(unique):
            target = matrix(action[self.arm+'_ee_pose'])
            distance = np.linalg.norm(current[:3, 3]-target[:3, 3])
            angle = Rotation.from_matrix(current[:3, :3].T@target[:3, :3]).magnitude()
            if distance <= .020 and angle <= .12:
                eligible.append((distance+.1*angle, i))
        path = [] if not eligible else unique[min(eligible)[1]:]
        previous = current
        for _, action in path:
            target = matrix(action[self.arm+'_ee_pose'])
            if (np.linalg.norm(previous[:3, 3]-target[:3, 3]) > .020
                    or Rotation.from_matrix(previous[:3, :3].T@target[:3, :3]).magnitude() > .12):
                path = []; break
            previous = target
        self._abort_path = path
        self._abort_wait = 0
        self.review_event(event='bounded_upright_abort', reason=reason, waypoints=len(path),
                          reset_environment=False, attempted_task_restart=False)

    def _abort_action(self):
        action = {k:deepcopy(self.obs['state'][k]) for k in JOINT_FIELDS}
        if self._abort_path:
            name, target = self._abort_path[0]
            reached, _, _ = self.reached(target)
            if reached:
                self._abort_path.pop(0); self._abort_wait = 0
                if not self._abort_path:
                    self.review_event(event='abort_upright_target_reached', native_fluid_success=None)
                else: name, target = self._abort_path[0]
            else: self._abort_wait += 1
            if self._abort_wait > 8:
                self._abort_path = []; self.review_event(event='abort_stalled')
            if self._abort_path:
                action[self.arm+'_arm_joint_state'] = deepcopy(target[self.arm+'_arm_joint_state'])
                action[self.arm+'_ee_joint_state'] = [0.]
        self.t += 1
        return [action]

    def get_action(self):
        if self.t >= LIMIT:
            return [{k:deepcopy(self.obs['state'][k]) for k in JOINT_FIELDS}]
        if self._abort_path is not None: return self._abort_action()
        if self.actions and self.cursor < len(self.actions):
            name = self.actions[self.cursor][0]
            if name.startswith(('tilt_', 'upright_')) or name == 'dwell':
                # Preserve enough command budget for the existing upright path
                # and observation settling; never start a new episode.
                remaining = len({n for n, _ in self.actions if n.startswith('upright_')})+20
                if self.failure or self.t+remaining >= LIMIT:
                    self._begin_abort('failure_or_upright_budget_reserve')
                    return self._abort_action()
        return super().get_action()
