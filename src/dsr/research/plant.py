"""Deterministic numerical plant and independent trajectory verifier.

This is a 1D damped point mass, not a robot simulator or a RoboDojo task.
Controllers observe position, velocity and goal; mass and drag remain in the
trusted simulator. Native trajectory scores are computed outside the candidate.
"""

from __future__ import annotations

import math
import random

DT = 0.02
STEPS = 150
FORCE_LIMIT = 4.0


def make_case(seed: int, task: str) -> dict:
    if task not in {"reach", "brake"}:
        raise ValueError("Unknown numerical condition")
    rng = random.Random(seed)
    return {
        "seed": seed,
        "task": task,
        "position": rng.uniform(-1.3, -0.7) if task == "reach" else rng.uniform(0.6, 1.2),
        "velocity": rng.uniform(-0.1, 0.1) if task == "reach" else rng.uniform(1.0, 1.8),
        "goal": rng.uniform(0.7, 1.3) if task == "reach" else rng.uniform(-0.1, 0.1),
        "mass": rng.uniform(0.8, 1.4),
        "drag": rng.uniform(0.1, 0.5),
        "dt": DT,
        "steps": STEPS,
    }


def verify_trajectory(case: dict, trace: list) -> dict:
    """Replay actual candidate actions and independently derive the outcome."""
    if len(trace) != case["steps"]:
        raise ValueError("Incomplete numerical rollout")
    x, velocity = case["position"], case["velocity"]
    errors, effort = [], 0.0
    for row in trace:
        if set(row) != {"position", "velocity", "action"}:
            raise ValueError("Unexpected trajectory fields")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in row.values()):
            raise ValueError("Non-finite trajectory")
        if abs(row["position"] - x) > 1e-9 or abs(row["velocity"] - velocity) > 1e-9:
            raise ValueError("Candidate trace disagrees with independent dynamics")
        action = row["action"]
        if abs(action) > FORCE_LIMIT:
            raise ValueError("Action exceeds physical contract")
        errors.append(abs(x - case["goal"]))
        effort += action * action * case["dt"]
        velocity += ((action - case["drag"] * velocity) / case["mass"]) * case["dt"]
        x += velocity * case["dt"]
    final_error = abs(x - case["goal"])
    average_error = sum(errors) / len(errors)
    score = 1.0 / (1.0 + average_error + final_error + 0.15 * abs(velocity))
    return {
        "score": score,
        "success": final_error < 0.12 and abs(velocity) < 0.2,
        "terminal_position": x,
        "terminal_velocity": velocity,
        "terminal_error": final_error,
        "mean_absolute_error": average_error,
        "effort_integral": effort,
        "steps": len(trace),
    }


# Trusted worker suffix. Only the generated action() function is mutable.
# The actor gets its observation dictionary, never hidden dynamics or a score.
RUNNER_SOURCE = r'''
import json
import math

cases = json.load(open('/input.json'))['cases']
rollouts = []
for case in cases:
    x, velocity = case['position'], case['velocity']
    trace = []
    for tick in range(case['steps']):
        observation = {'position': x, 'velocity': velocity, 'goal': case['goal']}
        force = action(observation)
        if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) or abs(force) > 4.0:
            raise ValueError('invalid action')
        trace.append({'position': x, 'velocity': velocity, 'action': force})
        velocity += ((force - case['drag'] * velocity) / case['mass']) * case['dt']
        x += velocity * case['dt']
    rollouts.append({'seed': case['seed'], 'task': case['task'], 'trace': trace})
json.dump({'rollouts': rollouts}, open('/output/result.json', 'w'), allow_nan=False)
'''
