"""Measured JSON is an explicit input; no simulation camera calibration is reused."""
from datetime import datetime, timezone
import numpy as np
from .config import read_json
from .geometry import vector
from .model import PreparationError


def age_seconds(stamp):
    try:
        t = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if t.tzinfo is None:
            raise ValueError("missing timezone")
        age = (datetime.now(timezone.utc)-t).total_seconds()
    except (TypeError, AttributeError, ValueError) as exc:
        raise PreparationError("captured_at must be an ISO timestamp with timezone") from exc
    if age < -2:
        raise PreparationError("Observation timestamp is in the future; check the acquisition clock")
    return age


def load_scene(path, config, task=None, hardware=False):
    scene = read_json(path)
    if scene.get("schema") != "dsr.xtrainer-scene/v1":
        raise PreparationError("Unsupported scene schema")
    if task and scene.get("task") != task:
        raise PreparationError("Scene task does not match requested task")
    if scene.get("calibration_id") != config.calibration["calibration_id"]:
        raise PreparationError("Scene and robot calibration_id differ")
    if scene.get("frame") != "world_mm":
        raise PreparationError("Scene must use the calibrated common world_mm frame")
    if scene.get("source", {}).get("kind") not in {"manual", "vision"}:
        raise PreparationError("Scene source.kind must be manual or vision")
    if hardware:
        if scene.get("example_only") is not False:
            raise PreparationError("Example scenes cannot be executed on hardware")
        if age_seconds(scene.get("captured_at")) > config.safety["scene_max_age_s"]:
            raise PreparationError("Scene is stale; acquire a new measured scene")
    objects = scene.get("objects")
    if not isinstance(objects, dict):
        raise PreparationError("scene.objects must be an object keyed by object identity")
    for name, obj in objects.items():
        vector(obj.get("position_world_mm"), 3, name+".position_world_mm")
    return scene


def alignment_metrics(objects):
    cubes = [objects["cube"+str(i)] for i in range(3)]
    points = np.asarray([vector(x["position_world_mm"], 3, "cube") for x in cubes])
    # This port fixes the straightedge along table X, so compare Y edge positions.
    lower_edges = [p[1]-float(c["size_mm"])/2 for c, p in zip(cubes, points)]
    spread = float(max(lower_edges)-min(lower_edges))
    yaw_error = max(abs((float(c["yaw_deg"])+45) % 90-45) for c in cubes)
    return {"edge_spread_mm": spread, "max_square_yaw_error_deg": yaw_error,
            "aligned": spread <= 8 and yaw_error <= 8}


def check_observation(expected, record, initial_scene):
    """Check external checkpoint evidence, never just trust an 'approved' flag."""
    kind = expected["kind"]
    if kind == "grasp":
        if record.get("grasped_object") != expected["object"]:
            raise PreparationError("Checkpoint did not confirm the requested grasp")
    elif kind == "cover_state":
        if sorted(record.get("covered_labels", [])) != sorted(expected["covered_labels"]):
            raise PreparationError("Observed covered labels do not match the remembered task state")
        if record.get("all_cups_opening_down") is not True:
            raise PreparationError("All cups must remain opening down")
        positions = record.get("block_positions_world_mm", {})
        for label in ["red", "green", "blue"]:
            now = vector(positions.get(label), 3, "checkpoint block "+label)
            before = vector(initial_scene["objects"][label]["position_world_mm"], 3, label)
            if np.linalg.norm(now-before) >= 50:
                raise PreparationError(label+": block displacement reached 50 mm")
    elif kind == "alignment":
        metrics = alignment_metrics(record.get("objects", {}))
        if not metrics["aligned"]:
            raise PreparationError("Measured blocks are not aligned within 8 mm / 8 degrees")
        lifts = record.get("max_block_lift_mm", {})
        for label in ["cube0", "cube1", "cube2"]:
            if not isinstance(lifts.get(label), (float, int)) or not 0 <= lifts[label] <= 7:
                raise PreparationError("Record max_block_lift_mm for all cubes (<= 7 mm)")
    else:
        raise PreparationError("Unknown checkpoint predicate: "+kind)
