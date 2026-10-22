"""All planning is in a common table/world frame, in millimetres and degrees.

Dobot RPY convention: Rz(rz) @ Ry(ry) @ Rx(rx). Confirm this on the installed
controller and record it in calibration.json before using a real configuration.
"""
import math
import numpy as np
from .model import PreparationError


def vector(value, n, name):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"{name}: expected {n} finite numbers") from exc
    if a.shape != (n,) or not np.isfinite(a).all():
        raise PreparationError(f"{name}: expected {n} finite numbers")
    return a


def matrix(value, name):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"{name}: expected a rigid 4x4 matrix") from exc
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise PreparationError(f"{name}: expected a rigid 4x4 matrix")
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-6):
        raise PreparationError(f"{name}: invalid homogeneous bottom row")
    r = a[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise PreparationError(f"{name}: rotation must be orthonormal and right handed")
    return a


def rpy_matrix(deg):
    x, y, z = np.deg2rad(vector(deg, 3, "RPY"))
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                     [-sy, cy*sx, cy*cx]])


def matrix_rpy(r):
    y = math.asin(float(np.clip(-r[2, 0], -1, 1)))
    if abs(math.cos(y)) < 1e-7:
        x, z = math.atan2(-r[1, 2], r[1, 1]), 0.0
    else:
        x, z = math.atan2(r[2, 1], r[2, 2]), math.atan2(r[1, 0], r[0, 0])
    return np.rad2deg([x, y, z])


def transform_pose(transform, pose):
    p = vector(pose, 6, "TCP pose")
    return np.r_[transform[:3, :3] @ p[:3] + transform[:3, 3],
                 matrix_rpy(transform[:3, :3] @ rpy_matrix(p[3:]))]


def rotation_distance(a, b):
    r = rpy_matrix(a).T @ rpy_matrix(b)
    return math.degrees(math.acos(float(np.clip((np.trace(r)-1)/2, -1, 1))))


def intersects_box(start, end, low, high):
    """Exact segment/AABB test, including endpoints (no sparse sampling holes)."""
    t0, t1 = 0.0, 1.0
    for s, d, lo, hi in zip(start, np.asarray(end)-start, low, high):
        if abs(d) < 1e-12:
            if s < lo or s > hi:
                return False
        else:
            near, far = sorted(((lo-s)/d, (hi-s)/d))
            t0, t1 = max(t0, near), min(t1, far)
            if t0 > t1:
                return False
    return True


def segment_distance(point, start, end):
    delta = np.asarray(end)-start
    t = float(np.clip(np.dot(np.asarray(point)-start, delta) / max(np.dot(delta, delta), 1e-12), 0, 1))
    return float(np.linalg.norm(np.asarray(point)-(np.asarray(start)+t*delta)))
