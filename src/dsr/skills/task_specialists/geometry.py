"""Observable feature geometry, with explicit uncertainty and no simulator API."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import linear_sum_assignment


class SkillError(ValueError):
    """A request lacks the evidence required by this particular skill."""


def scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise SkillError(name + ': finite scalar required')
    value = float(value)
    if not math.isfinite(value) or value < 0 or positive and value == 0:
        raise SkillError(name + ': invalid bound')
    return value


def array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SkillError(name + ': numeric geometry required') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise SkillError(name + ': invalid geometry')
    return result


def proper_rotation(value):
    r = array(value, (3, 3), 'rotation')
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-7, rtol=0) or abs(np.linalg.det(r)-1) > 1e-7:
        raise SkillError('rotation must be proper and orthonormal')
    return r


def unique_assignment(costs, *, max_cost, min_gap):
    """Globally unique one-to-one match, including its second-best assignment.

    Costs come from calibrated appearance/geometry features, not hidden IDs.
    Row-local confidence is insufficient when several objects share a target.
    Infinity denotes a forbidden pair. Returns row -> column, never aliases.
    """
    c = np.asarray(costs, dtype=float)
    ceiling = scalar(max_cost, 'max_cost')
    gap = scalar(min_gap, 'min_gap', positive=True)
    if c.ndim != 2 or not 0 < c.shape[0] <= c.shape[1] <= 64 or np.isnan(c).any() or (c < 0).any():
        raise SkillError('invalid assignment matrix')
    c = np.where(c <= ceiling, c, np.inf)

    def solve(matrix):
        try:
            rows, cols = linear_sum_assignment(matrix)
        except ValueError:
            return None
        total = float(matrix[rows, cols].sum())
        return (cols, total) if math.isfinite(total) else None

    best = solve(c)
    if best is None:
        raise SkillError('no feasible one-to-one assignment')
    cols, total = best
    alternate = math.inf
    # Every different complete assignment must omit at least one chosen edge.
    for row, col in enumerate(cols):
        forbidden = c.copy()
        forbidden[row, col] = np.inf
        candidate = solve(forbidden)
        if candidate is not None:
            alternate = min(alternate, candidate[1])
    if alternate-total < gap:
        raise SkillError('ambiguous global assignment; acquire another view')
    return dict(columns=cols.tolist(), cost=total,
                alternative_gap=None if math.isinf(alternate) else alternate-total)


def fit_functional_frame(source, destination, *, max_residual_m):
    """Rigid fit of >=3 corresponding *semantic* landmarks (not PCA axes).

    Caller must establish landmark identity and use the same metric frame.
    Residual measures consistency, not absolute pose accuracy or uncertainty.
    """
    a, b = np.asarray(source, float), np.asarray(destination, float)
    cap = scalar(max_residual_m, 'max_residual_m')
    if a.ndim != 2 or a.shape[1:] != (3,) or len(a) < 3 or a.shape != b.shape:
        raise SkillError('three corresponding landmarks required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise SkillError('nonfinite landmarks')
    ac, bc = a-a.mean(axis=0), b-b.mean(axis=0)
    for cloud in (ac, bc):
        sv = np.linalg.svd(cloud, compute_uv=False)
        if sv[0] < 1e-6 or sv[1]/sv[0] < 1e-3:
            raise SkillError('collinear or ill-conditioned semantic landmarks')
    u, _, vt = np.linalg.svd(ac.T @ bc)
    fix = np.eye(3)
    fix[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ fix @ u.T
    t = b.mean(axis=0)-r @ a.mean(axis=0)
    residual = float(np.max(np.linalg.norm(a @ r.T+t-b, axis=1)))
    if residual > cap:
        raise SkillError('inconsistent semantic correspondence')
    return dict(rotation=r.tolist(), translation_m=t.tolist(), max_residual_m=residual)


def precision_envelope(*, translation_errors_m, angular_errors_rad, lever_arm_m, tolerance_m):
    """Conservative sum of calibrated bounds; no independence assumption.

    Includes error from the grasp offset and semantic feature registration.
    Correlated errors cannot be reduced by averaging three copies of a frame.
    """
    if not translation_errors_m or not angular_errors_rad:
        raise SkillError('explicit translation and rotation error budgets required')
    linear = sum(scalar(v, 'translation_error') for v in translation_errors_m)
    angle = sum(scalar(v, 'angular_error') for v in angular_errors_rad)
    lever = scalar(lever_arm_m, 'lever_arm_m')
    tolerance = scalar(tolerance_m, 'tolerance_m', positive=True)
    bound = linear + 2*lever*math.sin(min(angle, math.pi)/2)
    if bound >= tolerance:
        raise SkillError('precision unobservable at requested tolerance')
    return dict(error_bound_m=bound, remaining_margin_m=tolerance-bound,
                calibration_required=True)


def support_margin(footprint_xy, support_polygon_xy, *, uncertainty_m, clearance_m):
    """Sufficient footprint-inside-convex-support test, independent of yaw.

    Inputs are points projected into the observed support plane, in metres.
    This is not a stability proof: COM, compliance and contact remain external.
    """
    f, p = np.asarray(footprint_xy, float), np.asarray(support_polygon_xy, float)
    margin = scalar(uncertainty_m, 'uncertainty_m')+scalar(clearance_m, 'clearance_m')
    if any(x.ndim != 2 or x.shape[1:] != (2,) or len(x) < 3 or not np.isfinite(x).all() for x in (f, p)):
        raise SkillError('finite planar polygons required')
    area = float(np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1)))/2
    if abs(area) < 1e-12:
        raise SkillError('degenerate support')
    if area < 0:
        p = p[::-1]
    distances = []
    for start, end in zip(p, np.roll(p, -1, axis=0)):
        e = end-start
        length = float(np.linalg.norm(e))
        if length < 1e-9:
            raise SkillError('duplicate support vertices')
        side = lambda points: (e[0]*(points[:, 1]-start[1])-e[1]*(points[:, 0]-start[0]))/length
        if np.min(side(p)) < -1e-9:
            raise SkillError('nonconvex or unordered support polygon')
        distances.extend(side(f).tolist())
    available = min(distances)
    if available < margin:
        raise SkillError('insufficient support containment margin')
    return dict(min_edge_distance_m=available, remaining_margin_m=available-margin)


def hinge_increment(*, hinge_m, axis, edge_m, remaining_angle_rad, max_angle_rad,
                    max_contact_step_m):
    """One short rotation about the current visually estimated hinge.

    Sign and closure direction are supplied by a task-specific RGB estimator.
    Reobserve after executing this proposal. It is neither a full open-loop
    trajectory nor a force, collision or latch guarantee.
    """
    h, a, e = array(hinge_m, (3,), 'hinge'), array(axis, (3,), 'axis'), array(edge_m, (3,), 'edge')
    n = np.linalg.norm(a)
    if abs(n-1) > 1e-6:
        raise SkillError('unit observed hinge axis required')
    if isinstance(remaining_angle_rad, bool) or not math.isfinite(remaining_angle_rad):
        raise SkillError('finite signed remaining angle required')
    cap = scalar(max_angle_rad, 'max_angle_rad', positive=True)
    distance = scalar(max_contact_step_m, 'max_contact_step_m', positive=True)
    if cap > .2:
        raise SkillError('increment too large for feedback contact')
    radial = e-h-a*np.dot(e-h, a)
    radius = float(np.linalg.norm(radial))
    if radius < .005:
        raise SkillError('free edge too close to hinge axis')
    angle_cap = min(cap, 2*math.asin(min(1., distance/(2*radius))))
    angle = float(np.clip(remaining_angle_rad, -angle_cap, angle_cap))
    v = e-h
    rotated = v*math.cos(angle)+np.cross(a, v)*math.sin(angle)+a*np.dot(a, v)*(1-math.cos(angle))
    return dict(contact_m=(h+rotated).tolist(), increment_rad=angle,
                reobserve_required=True, requires_native_path_check=True)


def pivot_end_effector(*, feature_world_m, feature_in_object_m, object_rotation,
                       object_in_gripper_rotation, object_in_gripper_translation_m):
    """Keep a measured spout, mallet tip or handle feature at its target.

    Rotating about the EEF origin would move the functional point. These
    transforms must be visually calibrated; this does not establish motion
    feasibility, liquid volume or actual contact.
    """
    target = array(feature_world_m, (3,), 'feature_world')
    feature = array(feature_in_object_m, (3,), 'feature_in_object')
    obj_r = proper_rotation(object_rotation)
    held_r = proper_rotation(object_in_gripper_rotation)
    held_t = array(object_in_gripper_translation_m, (3,), 'object_in_gripper_translation')
    ee_r = obj_r @ held_r.T
    obj_t = target-obj_r @ feature
    ee_t = obj_t-ee_r @ held_t
    return dict(position_m=ee_t.tolist(), rotation=ee_r.tolist(), requires_native_path_check=True)
