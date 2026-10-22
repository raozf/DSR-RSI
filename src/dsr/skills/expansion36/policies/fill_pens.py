"""One arm stabilizes the cup; measured slots protect inserted pens."""
import numpy as np
from scipy.spatial.transform import Rotation
from ..control import groups, feature, quaternion, vector, rotation, upright, PolicyFailure


def insertion_slots(holder, pens):
    """Four disjoint pen columns inside a conservative observed aperture."""
    upright(holder, tolerance=7)
    radius = float(feature(holder, "opening_radius_m"))
    radii = []
    for pen in pens:
        axis = vector(feature(pen, "tip_axis_local"), 3)
        if abs(np.linalg.norm(axis) - 1) > .001:
            raise PolicyFailure("pen_tip_axis")
        extent = np.abs(np.eye(3) - np.outer(axis, axis)) @ vector(pen["size"], 3)
        radii.append(float(np.linalg.norm(extent) / 2))
    pen_radius = max(radii)
    spacing = pen_radius + .003
    if not np.isfinite(radius) or radius - pen_radius - .004 < np.sqrt(2) * spacing:
        raise PolicyFailure("pen_aperture_capacity")
    measured_depth = float(feature(holder, "opening_depth_m"))
    depth = min(.05, measured_depth - .006)
    if not np.isfinite(measured_depth) or depth < .04:
        raise PolicyFailure("pen_insertion_depth")
    return [np.array([x * spacing, y * spacing, -depth])
            for x, y in ((-1, -1), (-1, 1), (1, -1), (1, 1))]


def verify_inserted(robot, holder, inserted):
    r = rotation(holder["quaternion"])
    mouth = vector(feature(holder, "opening"), 3)
    for identity, offset in inserted:
        pen = robot.object(identity)
        tip = vector(pen["position"], 3) + rotation(pen["quaternion"]).apply(feature(pen, "tip_local"))
        local = r.inv().apply(tip - mouth)
        if np.linalg.norm(local[:2] - offset[:2]) > .008 or abs(local[2] - offset[2]) > .015:
            raise PolicyFailure("inserted_pen_displaced", identity)


def policy(robot, memory):
    scene = yield from robot.observe()
    holder, pens = groups(scene, "pen_holder", 1)[0], groups(scene, "pen", 4)
    slots = insertion_slots(holder, pens)
    initial, initial_q = list(holder["position"]), list(holder["quaternion"])
    holder_arm = yield from robot.pick(holder["id"], rim=True)
    yield from robot.carry(holder_arm, initial, initial_q, slow=True)
    inserted = []
    for original, offset in zip(pens, slots):
        yield from robot.observe()
        current_holder = robot.object(holder["id"])
        upright(current_holder, tolerance=7)
        verify_inserted(robot, current_holder, inserted)
        mouth = vector(feature(current_holder, "opening"), 3)
        holder_r = rotation(current_holder["quaternion"])
        pen = robot.object(original["id"])
        tip_axis = vector(feature(pen, "tip_axis_local"), 3)
        object_q = quaternion(Rotation.align_vectors([-holder_r.apply([0, 0, 1])], [tip_axis])[0])
        arm = yield from robot.pick(pen["id"])
        target = mouth + holder_r.apply(offset) - rotation(object_q).apply(feature(pen, "tip_local"))
        yield from robot.observe()
        actual_holder = robot.object(holder["id"])
        actual = vector(feature(actual_holder, "opening"), 3)
        if np.linalg.norm(actual - mouth) > .006 or (holder_r.inv() * rotation(actual_holder["quaternion"])).magnitude() > np.deg2rad(3):
            raise PolicyFailure("holder_drift")
        verify_inserted(robot, actual_holder, inserted)
        yield from robot.place(arm, target, object_q, tolerance=.012)
        inserted.append((pen["id"], offset))
    yield from robot.observe()
    verify_inserted(robot, robot.object(holder["id"]), inserted)
    yield from robot.release_at(holder_arm, initial, initial_q, tolerance=.012)
    yield from robot.observe()
    verify_inserted(robot, robot.object(holder["id"]), inserted)
    return (yield from robot.home_arms())
