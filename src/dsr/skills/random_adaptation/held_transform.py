"""Rebind future bottle targets at acknowledged physical stage boundaries."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation


def matrix(pose):
    p = np.asarray(pose, float)
    if p.shape != (7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:]) - 1) > .001:
        raise ValueError('invalid wxyz pose')
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
    t[:3, 3] = p[:3]
    return t


def transform(t):
    t = np.asarray(t, float)
    if (t.shape != (4, 4) or not np.isfinite(t).all() or
            not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or
            np.linalg.det(t[:3, :3]) < .999):
        raise ValueError('invalid rigid transform')
    return t


def pose(t):
    t = transform(t)
    return [*t[:3, 3].tolist(), *Rotation.from_matrix(t[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()]


def held_transform_update(*, ee_pose, observed_object_pose, nominal_attachment,
                          max_translation_m=.045, max_angle_rad=.25):
    live = np.linalg.inv(matrix(ee_pose)) @ matrix(observed_object_pose)
    nominal = transform(nominal_attachment)
    distance = float(np.linalg.norm(live[:3, 3] - nominal[:3, 3]))
    angle = float(Rotation.from_matrix(nominal[:3, :3].T @ live[:3, :3]).magnitude())
    if distance > max_translation_m or angle > max_angle_rad:
        raise ValueError('held transform exceeds registration/grasp envelope')
    return live


def rebind_future_stages(stages, *, arm, first_future_index, nominal_attachment,
                         live_attachment, cup_delta=None, max_target_delta_m=.06):
    """Preserve desired bottle path; caller consumes pending actions first."""
    nominal, live = transform(nominal_attachment), transform(live_attachment)
    delta = np.eye(4) if cup_delta is None else transform(cup_delta)
    if not 0 <= first_future_index < len(stages) or arm not in ('left', 'right'):
        raise ValueError('invalid future boundary')
    result = deepcopy(stages)
    key = arm + '_ee_pose'
    for index in range(first_future_index, len(result)):
        original = matrix(stages[index]['target'][key])
        updated = delta @ original @ nominal @ np.linalg.inv(live)
        if np.linalg.norm(updated[:3, 3] - original[:3, 3]) > max_target_delta_m:
            raise ValueError('rebound path displacement exceeds envelope')
        result[index]['target'][key] = pose(updated)
    return result
