"""Estimate world_T_base from measured point correspondences, in millimetres."""
import numpy as np
from .model import PreparationError


def fit_rigid(base_points, world_points):
    a, b = np.asarray(base_points, dtype=float), np.asarray(world_points, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 3 or len(a) < 4 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise PreparationError("Provide >=4 corresponding finite base/world XYZ points in mm")
    ac, bc = a-a.mean(axis=0), b-b.mean(axis=0)
    if np.linalg.matrix_rank(ac, tol=1e-5) < 2 or np.linalg.matrix_rank(bc, tol=1e-5) < 2:
        raise PreparationError("Calibration points must not be collinear")
    u, _, vt = np.linalg.svd(ac.T @ bc)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ correction @ u.T
    t = b.mean(axis=0)-r @ a.mean(axis=0)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = r, t
    errors = np.linalg.norm((r @ a.T).T+t-b, axis=1)
    if errors.max() > 3:
        raise PreparationError(f"Calibration residual {errors.max():.3f} mm exceeds 3 mm; check correspondences/units")
    return {"world_T_base": transform.tolist(), "rms_mm": float(np.sqrt(np.mean(errors**2))),
            "max_residual_mm": float(errors.max()), "points": len(a), "tcp_calibration_included": False}
