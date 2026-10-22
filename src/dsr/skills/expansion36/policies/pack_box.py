"""Four front-facing objects, oriented box and nonoverlapping floor slots."""
from ..control import groups, feature, packing_slots, PolicyFailure, quaternion, rotation
from scipy.spatial.transform import Rotation


def policy(robot, memory):
    scene = yield from robot.observe()
    box = groups(scene, "box", 1)[0]
    # The box's local Y axis must face +/- world X.
    box_q = quaternion(Rotation.from_euler("z", 90, degrees=True))
    yield from robot.transfer(box["id"], box["position"], box_q, tolerance=.025)
    scene = yield from robot.observe()
    box = robot.object(box["id"])
    items = groups(scene, "pack_item", 4)
    oriented = []
    desired_q = {}
    for item in items:
        front = feature(item, "front_axis_local")
        q = quaternion(Rotation.align_vectors([[-1, 0, 0], [0, 0, 1]], [front, [0, 0, 1]])[0])
        # Conservative world-aligned footprint for the requested orientation.
        copy = dict(item)
        copy["size"] = (abs(rotation(box_q).as_matrix().T @ rotation(q).as_matrix()) @ item["size"]).tolist()
        oriented.append(copy); desired_q[item["id"]] = q
    slots = packing_slots(box, oriented)
    for item in oriented:
        yield from robot.transfer(item["id"], slots[item["id"]], desired_q[item["id"]], tolerance=.02)
    return (yield from robot.home_arms())
