"""Physical-time prediction from causal RGB tracks, never implicit wall time."""
import numpy as np
from .geometry import SkillError, scalar, array
from .memory import stamp


def fit_track(samples, *, object_id, max_gap_s, measurement_error_m):
    if not isinstance(object_id, str) or not object_id or len(samples) < 3:
        raise SkillError('three same-object timed observations required')
    gap = scalar(max_gap_s, 'max_gap_s', positive=True)
    error = scalar(measurement_error_m, 'measurement_error_m')
    last = None
    times, points = [], []
    for sample in samples:
        mark = stamp(sample['stamp'], last)
        if sample.get('object_id') != object_id or last and mark['time_s']-last['time_s'] > gap:
            raise SkillError('track identity changed or observation gap')
        last = mark
        times.append(mark['time_s'])
        points.append(array(sample['position_m'], (3,), 'track_position'))
    t = np.asarray(times)-times[-1]
    span = times[-1]-times[0]
    if span < 1e-3:
        raise SkillError('insufficient temporal baseline')
    design = np.column_stack((t, np.ones(len(t))))
    coef, _, _, _ = np.linalg.lstsq(design, np.asarray(points), rcond=None)
    residual = float(np.max(np.linalg.norm(design @ coef-points, axis=1)))
    if residual > error:
        raise SkillError('track inconsistent with constant-velocity uncertainty bound')
    # Bounded measurement error, not a statistically calibrated covariance.
    velocity_error = float(np.sum(abs(t-t.mean())) / np.sum((t-t.mean())**2))*error
    offset_error = error*(1+abs(t.mean())*float(np.sum(abs(t-t.mean())) / np.sum((t-t.mean())**2)))
    return dict(object_id=object_id, stamp=last, position_m=coef[1].tolist(),
                velocity_m_s=coef[0].tolist(), fit_residual_m=residual,
                position_error_m=offset_error, velocity_error_m_s=velocity_error,
                fit_span_s=span)


def predict_intercept(track, *, now_s, now_clock, contact_delay_s, max_age_s, max_horizon_s,
                      acceleration_bound_m_s2, timing_error_s, tolerance_m, workspace_min_m, workspace_max_m):
    """Delay means physical time until contact in the declared clock domain.

    Synchronous RoboDojo pauses physics while waiting on a model RPC. Its wall
    inference duration must not be added as simulated conveyor displacement.
    A native adapter still needs to provide actual frame simulation timestamps
    and measured action duration; collect_freq alone may not cover long plans.
    """
    if now_clock not in ('simulation', 'real_sensor') or now_clock != track['stamp']['clock']:
        raise SkillError('intercept clock domain mismatch')
    now = scalar(now_s, 'now_s')
    delay = scalar(contact_delay_s, 'contact_delay_s', positive=True)
    age_cap = scalar(max_age_s, 'max_age_s', positive=True)
    horizon = scalar(max_horizon_s, 'max_horizon_s', positive=True)
    acc = scalar(acceleration_bound_m_s2, 'acceleration_bound_m_s2')
    jitter = scalar(timing_error_s, 'timing_error_s')
    tolerance = scalar(tolerance_m, 'tolerance_m', positive=True)
    age = now-track['stamp']['time_s']
    if not 0 <= age <= age_cap or age+delay+jitter > horizon:
        raise SkillError('stale or excessive intercept horizon')
    p = array(track['position_m'], (3,), 'track_position')
    v = array(track['velocity_m_s'], (3,), 'track_velocity')
    lower, upper = array(workspace_min_m, (3,), 'workspace_min'), array(workspace_max_m, (3,), 'workspace_max')
    if not np.all(lower < upper):
        raise SkillError('empty intercept workspace')
    dt = age+delay
    predicted = p+v*dt
    # Account for model deviation over both the fit span and extrapolation;
    # zero measurement residual alone does not prove constant future motion.
    bound = (scalar(track['position_error_m'], 'position_error_m')
             +scalar(track['velocity_error_m_s'], 'velocity_error_m_s')*(dt+jitter)
             +float(np.linalg.norm(v))*jitter
             +acc*(track['fit_span_s']+dt+jitter)**2)
    if bound >= tolerance or np.any(predicted-bound < lower) or np.any(predicted+bound > upper):
        raise SkillError('interception uncertain or outside reachable workspace')
    return dict(object_id=track['object_id'], contact_time_s=now+delay,
                target_m=predicted.tolist(), error_bound_m=bound,
                requires_native_path_check=True, reobserve_before_contact=True)
