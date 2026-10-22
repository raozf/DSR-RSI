"""Reusable feedback checks for RGB-estimated geometry, never task rewards.

These import-free functions can be packaged into native policy(robot, memory)
candidates. Passing a check means observed progress, not benchmark success.
"""


def finite_vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("Geometry has the wrong dimension")
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not -1e100 < x < 1e100 for x in value):
        raise ValueError("Geometry must contain finite numbers")
    return list(value)


def observed_object(scene, name, previous):
    """Require fresh, named RGB geometry; disappearance is uncertainty."""
    if scene.get("frame") != "world" or scene.get("unit") != "m":
        raise ValueError("World-frame RGB estimates in metres are required")
    identity = scene["observation"]
    if not isinstance(identity["sequence"], int) or isinstance(identity["sequence"], bool) or identity["sequence"] < 1:
        raise ValueError("Invalid observation sequence")
    if previous is not None and (identity["channel_id"] != previous["channel_id"]
                                or identity["sequence"] <= previous["sequence"]):
        raise ValueError("Fresh observations from the same episode are required")
    if name not in scene["objects"]:
        raise ValueError("Target unresolved; absence cannot verify a grasp or placement")
    estimate = scene["objects"][name]
    pose = finite_vector(estimate["pose"], 7)
    dimensions = finite_vector(estimate["dims"], 3)
    if min(dimensions) <= 0 or abs(sum(x*x for x in pose[3:]) - 1) > 1e-3:
        raise ValueError("Invalid observed cuboid")
    return {"pose": pose, "dims": dimensions, "observation": dict(identity)}


def lift_progress(initial, current, expected_lift_m, tolerance_m):
    """Check the object actually rises without lateral escape or inversion."""
    displacement = [current["pose"][i] - initial["pose"][i] for i in range(3)]
    if displacement[2] < expected_lift_m - tolerance_m:
        raise ValueError("Object did not follow the lift or slipped")
    if displacement[2] > expected_lift_m + tolerance_m:
        raise ValueError("Observed object identity or height is inconsistent")
    if sum(x*x for x in displacement[:2]) ** 0.5 > tolerance_m:
        raise ValueError("Object moved laterally during vertical lift")
    similarity = abs(sum(a*b for a, b in zip(initial["pose"][3:], current["pose"][3:])))
    if similarity < 0.9659258262890683:
        raise ValueError("Object rotated more than 30 degrees during lift")
    return displacement[2]


def stable_geometry(samples, tolerance_m):
    """Reject slip after initial acquisition and repeated cached frames."""
    if not isinstance(tolerance_m, (int, float)) or isinstance(tolerance_m, bool) or not 0 < tolerance_m <= 0.3:
        raise ValueError("A bounded positive stability tolerance is required")
    if len(samples) < 3:
        raise ValueError("At least three fresh observations are required for stability")
    for sample in samples:
        pose = finite_vector(sample["pose"], 7)
        if abs(sum(x*x for x in pose[3:]) - 1) > 1e-3:
            raise ValueError("Stability requires normalized observed quaternions")
        identity = sample["observation"]
        if (not isinstance(identity["channel_id"], str) or not identity["channel_id"] or
                not isinstance(identity["sequence"], int) or isinstance(identity["sequence"], bool) or
                identity["sequence"] < 1):
            raise ValueError("Invalid stability observation identity")
    previous = samples[0]
    for sample in samples[1:]:
        before, after = previous["observation"], sample["observation"]
        if before["channel_id"] != after["channel_id"] or after["sequence"] <= before["sequence"]:
            raise ValueError("Stability requires fresh observations from one episode")
        previous = sample
    for axis in range(3):
        coordinates = [sample["pose"][axis] for sample in samples]
        if max(coordinates) - min(coordinates) > tolerance_m:
            raise ValueError("Object is unstable during the hold")
    reference = samples[0]["pose"][3:]
    if any(abs(sum(a*b for a, b in zip(reference, sample["pose"][3:]))) < 0.9914448613738104
           for sample in samples[1:]):
        raise ValueError("Object orientation is unstable during the hold")
    return {"observations": len(samples), "first": samples[0]["observation"],
            "last": samples[-1]["observation"], "scope": "RGB stability, not native success"}


def manipulation_parameters(memory):
    """Require supplied robot calibration; never guess a task layout or pose."""
    configuration = memory["manipulation"]
    for name in ("clearance_m", "lift_m", "geometry_tolerance_m", "stability_tolerance_m"):
        number = configuration[name]
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not 0 < number <= 0.3:
            raise ValueError("Invalid manipulation bound: " + name)
    if configuration["lift_m"] < 0.06 or configuration["geometry_tolerance_m"] >= configuration["lift_m"] / 3:
        raise ValueError("Lift must be distinguishable from perception uncertainty")
    for name in ("open_grip", "closed_grip"):
        number = configuration[name]
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not 0 <= number <= 1:
            raise ValueError("Normalized gripper commands are required")
    if configuration["open_grip"] <= configuration["closed_grip"]:
        raise ValueError("Open gripper command must exceed closed command")
    for arm in ("left", "right"):
        pose = finite_vector(configuration["home_poses"][arm], 7)
        quaternion = finite_vector(configuration["grasp_quaternions"][arm], 4)
        finite_vector(configuration["ee_offsets_m"][arm], 3)
        if abs(sum(x*x for x in quaternion) - 1) > 1e-4 or abs(sum(x*x for x in pose[3:]) - 1) > 1e-4:
            raise ValueError("Robot calibration quaternions must be normalized")
    return configuration


def grasp_target(estimate, arm, configuration, rim):
    """Convert an observed object centre to a calibrated grasp-frame target."""
    offset = configuration["ee_offsets_m"][arm]
    target = [estimate["pose"][i] + offset[i] for i in range(3)]
    if rim:
        # Near-side rim contact follows the observed bowl radius, not its layout.
        target[0] += (-1 if arm == "left" else 1) * estimate["dims"][0] * 0.45
        target[2] += estimate["dims"][2] * 0.35
    return target


def placement_inside(container, item, margin_m):
    """Require visible containment; image disappearance never clears a ledger."""
    for axis in (0, 1):
        clearance = (container["dims"][axis] - item["dims"][axis]) / 2 - margin_m
        if clearance <= 0 or abs(item["pose"][axis] - container["pose"][axis]) > clearance:
            raise ValueError("Object not visibly inside the container")
    bottom = container["pose"][2] - container["dims"][2] / 2
    top = container["pose"][2] + container["dims"][2] / 2
    if not bottom <= item["pose"][2] < top:
        raise ValueError("Object height is inconsistent with containment")
    return True


def relative_alignment(actual, target, tolerance_m, position_axes, require_orientation):
    """Require a fresh object pose to be close to an observed target pose.

    The target is another RGB estimate from the same episode.  This helper is
    deliberately geometry-only: it cannot inspect the evaluator or infer a
    hidden goal.  Callers should observe both poses immediately before using
    the result and preserve the returned error in their evidence record.
    """
    if not isinstance(tolerance_m, (int, float)) or isinstance(tolerance_m, bool) or not 0 < tolerance_m <= 0.3:
        raise ValueError("A positive alignment tolerance is required")
    actual_pose = finite_vector(actual["pose"], 7)
    target_pose = finite_vector(target["pose"], 7)
    if any(abs(sum(x*x for x in pose[3:]) - 1) > 1e-3 for pose in (actual_pose, target_pose)):
        raise ValueError("Alignment requires normalized observed quaternions")
    if not isinstance(require_orientation, bool):
        raise ValueError("Orientation requirement must be explicit")
    actual_id, target_id = actual["observation"], target["observation"]
    if actual_id != target_id:
        raise ValueError("Relative alignment requires poses from the same observation")
    if (not isinstance(position_axes, (list, tuple)) or not position_axes or
            any(not isinstance(axis, int) or isinstance(axis, bool) or axis not in (0, 1, 2)
                for axis in position_axes) or
            len(set(position_axes)) != len(position_axes)):
        raise ValueError("Position axes must be distinct Cartesian indices")
    error = sum((actual_pose[i] - target_pose[i]) ** 2 for i in position_axes) ** 0.5
    if error > tolerance_m:
        raise ValueError("Object is not aligned with the observed target")
    # The scalar quaternion distance is sign invariant, as q and -q encode
    # the same orientation.
    orientation = abs(sum(a * b for a, b in zip(actual_pose[3:], target_pose[3:])))
    if require_orientation and orientation < 0.9659258262890683:
        raise ValueError("Object orientation is not aligned with the observed target")
    return {"position_error_m": error, "orientation_similarity": orientation,
            "position_axes": list(position_axes),
            "scope": "RGB relative alignment, not native success"}


def identity_ledger(names):
    """Create an explicit mutable object ledger for a multi-object policy."""
    if not isinstance(names, (list, tuple)) or not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Object identity ledger requires non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("Object identity ledger contains duplicate names")
    return {name: {"status": "unseen", "attempts": 0} for name in names}


def released_stability(samples, tolerance_m):
    """Check that a released object remains visible and settled."""
    if not isinstance(tolerance_m, (int, float)) or isinstance(tolerance_m, bool) or not 0 < tolerance_m <= 0.3:
        raise ValueError("A bounded positive stability tolerance is required")
    if len(samples) < 3:
        raise ValueError("At least three fresh observations are required for stability")
    if samples[-1]["pose"][2] < samples[0]["pose"][2] - tolerance_m:
        raise ValueError("Object descended after release")
    result = stable_geometry(samples, tolerance_m)
    result["scope"] = "RGB post-release stability, not native success"
    result["vertical_drift_m"] = samples[-1]["pose"][2] - samples[0]["pose"][2]
    return result
