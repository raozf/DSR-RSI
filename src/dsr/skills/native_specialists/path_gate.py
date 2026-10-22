"""Observed-volume path veto; full-arm qualification stays explicit."""
import numpy as np


def scene_path_gate(points, *, envelope_radius_m, obstacles, unknown=False,
                    full_arm_qualified=False):
    """Segment/expanded-AABB veto for EEF or held-object motion.

    This does not certify a whole-arm trajectory. ProhibitedArea is never used
    here: that field controls layout generation, not runtime motion.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all() or not np.isfinite(envelope_radius_m) or envelope_radius_m <= 0:
        raise ValueError('invalid path envelope')
    for obstacle in obstacles:
        lo, hi = np.asarray(obstacle['lower'], float), np.asarray(obstacle['upper'], float)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([*lo, *hi]).all() or np.any(lo >= hi):
            raise ValueError('invalid observed obstacle')
        lo, hi = lo - envelope_radius_m, hi + envelope_radius_m
        for a, b in zip(p[:-1], p[1:]):
            lower, upper = 0., 1.
            for axis in range(3):
                d = b[axis] - a[axis]
                if abs(d) < 1e-12:
                    if not lo[axis] <= a[axis] <= hi[axis]:
                        upper = -1.
                        break
                else:
                    t0, t1 = sorted(((lo[axis] - a[axis]) / d, (hi[axis] - a[axis]) / d))
                    lower, upper = max(lower, t0), min(upper, t1)
            if lower <= upper:
                return dict(allowed=False, qualified=False, reason='observed_volume_intersection')
    return dict(allowed=not unknown, qualified=not unknown and full_arm_qualified,
                reason='unknown_geometry' if unknown else 'clear_observed_volume' if full_arm_qualified else 'full_arm_qualification_missing')
