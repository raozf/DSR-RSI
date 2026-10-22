import numpy as np
from .geometry import intersects_box, rotation_distance, segment_distance, vector
from .model import PreparationError


def check_segment(config, side, start, goal, other):
    start, goal, other = (vector(v, 6, "TCP pose") for v in [start, goal, other])
    s = config.safety
    pad = s["tcp_padding_mm"]
    bounds = config.frames[side]["workspace_world_mm"]
    lo, hi = np.asarray(bounds["min"])+pad, np.asarray(bounds["max"])-pad
    if np.any(np.minimum(start[:3], goal[:3]) < lo) or np.any(np.maximum(start[:3], goal[:3]) > hi):
        raise PreparationError(f"{side}: TCP segment exceeds calibrated workspace (including padding)")
    if np.linalg.norm(goal[:3]-start[:3]) > s["max_step_mm"]+1e-6:
        raise PreparationError(f"{side}: TCP step exceeds max_step_mm")
    if rotation_distance(start[3:], goal[3:]) > s["max_rotation_step_deg"]+1e-6:
        raise PreparationError(f"{side}: rotation step exceeds max_rotation_step_deg")
    if segment_distance(other[:3], start[:3], goal[:3]) < s["min_tcp_separation_mm"]:
        raise PreparationError(f"{side}: segment too close to the other arm's TCP")
    for box in s.get("forbidden_boxes_world_mm", []):
        if intersects_box(start[:3], goal[:3], np.asarray(box["min"])-pad, np.asarray(box["max"])+pad):
            raise PreparationError(f"{side}: segment intersects forbidden box {box.get('name', '')}")


def validate_plan(config, plan):
    poses = {side: config.park(side).copy() for side in config.arms}
    for side in poses:
        check_segment(config, side, poses[side], poses[side], poses["right" if side == "left" else "left"])
    if not plan.steps:
        raise PreparationError("Empty task plan")
    stages = set()
    for step in plan.steps:
        if step.kind == "checkpoint":
            if step.stage in stages:
                raise PreparationError("Duplicate checkpoint: "+step.stage)
            stages.add(step.stage)
            continue
        if step.arm not in poses:
            raise PreparationError("Action must name left or right arm")
        if step.kind == "move":
            other = "right" if step.arm == "left" else "left"
            check_segment(config, step.arm, poses[step.arm], step.pose_world_mm_deg, poses[other])
            poses[step.arm] = vector(step.pose_world_mm_deg, 6, "target")
        elif step.kind == "gripper":
            if step.opening is None or not np.isfinite(step.opening) or not 0 <= step.opening <= 1:
                raise PreparationError("Gripper opening must be in [0, 1]")
        else:
            raise PreparationError("Unknown step kind: "+step.kind)
    for side in poses:
        if not np.allclose(poses[side], config.park(side), atol=1e-6):
            raise PreparationError(side+": task must end at its calibrated park TCP")


def check_robot_state(config, states, parked_side=None):
    if set(states) != {"left", "right"}:
        raise PreparationError("Fresh feedback from both left and right arms is required")
    for side, state in states.items():
        q = vector(state["joints_deg"], 6, side+" joints")
        f = config.frames[side]
        if np.any(q < f["joint_min_deg"]) or np.any(q > f["joint_max_deg"]):
            raise PreparationError(side+": measured joint state exceeds configured bounds")
        if state["mode"] not in {5, 7} or state["error"] or state["drag"] or state["paused"]:
            raise PreparationError(side+": invalid robot mode, alarm, drag or pause state")
        # Endpoint/workspace and forbidden-volume checks also apply to live observations.
        other = "right" if side == "left" else "left"
        check_segment(config, side, state["pose_world_mm_deg"], state["pose_world_mm_deg"], states[other]["pose_world_mm_deg"])
    if parked_side:
        p = vector(states[parked_side]["pose_world_mm_deg"], 6, "parked pose")
        park = config.park(parked_side)
        q = np.asarray(states[parked_side]["joints_deg"])
        qpark = np.asarray(config.frames[parked_side]["park_joints_deg"])
        if states[parked_side]["mode"] != 5 or np.max(np.abs(q-qpark)) > config.safety["park_joint_tolerance_deg"] or np.linalg.norm(p[:3]-park[:3]) > config.safety["park_drift_tolerance_mm"] or rotation_distance(p[3:], park[3:]) > config.safety["rotation_tolerance_deg"]:
            raise PreparationError(parked_side+": inactive arm moved away from park")
