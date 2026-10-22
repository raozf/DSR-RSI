"""Nesting intervals from registered public shell sections, in metres."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BowlProfileInterval:
    z: np.ndarray
    inner_lower: np.ndarray
    outer_upper: np.ndarray
    floor_z: float
    bottom_z: float
    rim_z: float
    model_sha256: str

    def __post_init__(self):
        z, inner, outer = (np.asarray(x, float).copy() for x in
                           (self.z, self.inner_lower, self.outer_upper))
        if (z.ndim != 1 or len(z) < 6 or inner.shape != z.shape or outer.shape != z.shape
                or not np.isfinite([*z, *inner, *outer, self.floor_z, self.bottom_z, self.rim_z]).all()
                or np.any(np.diff(z) <= 0) or np.any(inner < 0) or np.any(outer <= 0)
                or np.any(inner > outer) or not self.bottom_z < self.floor_z < self.rim_z
                or z[0] < self.bottom_z or z[-1] > self.rim_z or not self.model_sha256):
            raise ValueError('invalid public bowl profile')
        for name, value in (('z', z), ('inner_lower', inner), ('outer_upper', outer)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def estimate_profile_interval(**fields):
    """Validate the mesh-derived profile at its loader boundary."""
    return BowlProfileInterval(**fields)


def nesting_offset(support, held, *, lateral_error_m, vertical_error_m,
                   minimum_separation_m=.006, resolution_m=.001):
    """Smallest feasible root offset, with conservative radial clearance.

    This is a geometric placement proxy, not a force or success certificate.
    Same-type bowls still need their complete outer/inner sections checked.
    """
    if not np.isfinite([lateral_error_m, vertical_error_m, resolution_m]).all() or min(lateral_error_m, vertical_error_m) < 0 or resolution_m <= 0:
        raise ValueError('invalid profile uncertainty')
    lo = max(minimum_separation_m, support.floor_z - held.bottom_z + vertical_error_m)
    hi = support.rim_z - held.bottom_z - max(.005, vertical_error_m)
    inside = support.z >= support.floor_z
    if inside.sum() < 4 or hi < lo:
        return None
    sz, sr = support.z[inside], support.inner_lower[inside]
    hz = np.linspace(held.z[0], held.z[-1], max(100, len(held.z) * 3))
    hr = np.interp(hz, held.z, held.outer_upper)
    feasible = []
    for offset in np.arange(lo, hi + resolution_m / 2, resolution_m):
        world_z = hz + offset
        overlap = world_z <= support.rim_z - vertical_error_m
        if not overlap.any():
            continue
        query = world_z[overlap]
        if query.min() < sz[0] or query.max() > sz[-1]:
            continue
        lows = np.maximum(sz[0], query - vertical_error_m)
        highs = np.minimum(sz[-1], query + vertical_error_m)
        allowed = np.minimum(np.interp(lows, sz, sr), np.interp(highs, sz, sr))
        if np.all(hr[overlap] + lateral_error_m <= allowed):
            feasible.append(float(offset))
    if not feasible:
        return None
    return dict(offset_m=feasible[0], lower_m=feasible[0], upper_m=feasible[-1],
                radial_error_m=float(lateral_error_m), vertical_error_m=float(vertical_error_m))
