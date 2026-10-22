"""Observation-driven primitives copied into the isolated bowl controller.

Generators yield commands to the descriptor worker, which acknowledges them
only after physical execution. The same port supports deterministic fault
injection in CPU tests. It never receives environment objects or reward APIs.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.spatial.transform import Rotation


class PolicyFailure(ValueError):
    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(reason + (": " + detail if detail else ""))


def vector(value, length):
    a = np.asarray(value, dtype=float)
    if a.shape != (length,) or not np.isfinite(a).all():
        raise PolicyFailure("invalid_geometry")
    return a


def rotation(q):
    q = vector(q, 4)
    if abs(np.linalg.norm(q) - 1) > 1e-3:
        raise PolicyFailure("invalid_orientation")
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def quaternion(r):
    return r.as_quat()[[3, 0, 1, 2]].tolist()


def yaw(q, angle):
    return quaternion(Rotation.from_euler("z", angle) * rotation(q))


def groups(scene, kind, count=None):
    items = [x for x in scene["objects"].values() if x["kind"] == kind]
    if not items or count is not None and len(items) != count:
        raise PolicyFailure("object_count", f"{kind}: {len(items)}, expected {count}")
    return deepcopy(sorted(items, key=lambda x: x["id"]))


def feature(obj, name):
    value = obj.get("features", {}).get(name)
    if value is None:
        raise PolicyFailure("unresolved_feature", obj["id"] + "/" + name)
    return value


def surface(obj):
    # Bounding-box top is used only for horizontal support surfaces. Callers
    # requiring a hole, handle, inner floor or semantic axis request a feature.
    return float(obj["position"][2] + obj["size"][2] / 2)


def upright(obj, tolerance=12):
    normal = rotation(obj["quaternion"]).apply([0, 0, 1])
    if normal[2] < math.cos(math.radians(tolerance)):
        raise PolicyFailure("tilted_support", obj["id"])


def packing_slots(container, items, margin=.008):
    """Shelf-pack observed footprints in an observed *inner* rectangle.

    No arbitrary resizing or placement outside the visible opening is allowed.
    Input items are already oriented to their required final pose.
    """
    upright(container, tolerance=7)
    inner = vector(feature(container, "inner_size"), 2)
    floor = vector(feature(container, "inner_floor"), 3)
    if min(inner) <= 2 * margin:
        raise PolicyFailure("opening_too_small")
    ordered = sorted(items, key=lambda x: (-x["size"][1], -x["size"][0], x["id"]))
    x = y = margin
    row_height = 0.
    destinations = {}
    for item in ordered:
        rim = container.get("features", {}).get("rim_height")
        if rim is not None and floor[2] + item["size"][2] > rim + .008:
            raise PolicyFailure("item_exceeds_container_depth", item["id"])
        w, h = item["size"][:2]
        if w + 2 * margin > inner[0] or h + 2 * margin > inner[1]:
            raise PolicyFailure("item_does_not_fit", item["id"])
        if x + w + margin > inner[0]:
            x, y, row_height = margin, y + row_height + margin, 0.
        if y + h + margin > inner[1]:
            raise PolicyFailure("packing_capacity")
        local = [x + w / 2 - inner[0] / 2, y + h / 2 - inner[1] / 2, 0]
        point = floor + rotation(container["quaternion"]).apply(local)
        point[2] += item["size"][2] / 2
        destinations[item["id"]] = point.tolist()
        x, row_height = x + w + margin, max(row_height, h)
    return destinations


class Robot:
    """Bounded closed-loop actions; every completion is a visual claim only."""
    def __init__(self, memory, instruction, home):
        self.memory = deepcopy(memory)
        self.instruction = instruction
        self.home = deepcopy(home)
        self.scene = None
        self.sequence = -1
        self.channel = None
        self.holding = {}
        self.hypotheses = {}
        self.closed_arms = set()
        self.native_steps = 0
        self.completed = []
        self.stage = "initialization"
        self.commands = 0
        self.ee = deepcopy(home)

    def command(self, op, **kwargs):
        self.commands += 1
        if self.commands > self.memory.get("max_commands", 700):
            raise PolicyFailure("command_budget")
        response = yield {"op": op, "stage": self.stage, **kwargs}
        return response

    def observe(self, required=None):
        scene = yield from self.command("observe", **({'required':required} if required else {}))
        if scene.get("source") != "rgb" or scene.get("frame") != "world" or scene.get("unit") != "m":
            raise PolicyFailure("observation_contract")
        seq, channel = scene.get("sequence"), scene.get("channel")
        if type(seq) is not int or seq <= self.sequence or not isinstance(channel, str) or not channel:
            raise PolicyFailure("stale_observation")
        if self.channel is not None and channel != self.channel:
            raise PolicyFailure("episode_changed")
        for identity, obj in scene["objects"].items():
            if identity != obj.get("id") or not obj.get("kind"):
                raise PolicyFailure("object_identity")
            vector(obj["position"], 3)
            size = vector(obj["size"], 3)
            if min(size) <= 0 or not .30 <= obj.get("confidence", 0) <= 1:
                raise PolicyFailure("uncertain_geometry", identity)
            rotation(obj["quaternion"])
        self.sequence, self.channel, self.scene = seq, channel, scene
        if "robot_state" in scene:
            self.ee = {arm:list(scene["robot_state"][arm+"_ee_pose"]) for arm in ("left","right")}
            for arm, held in self.holding.items():
                held['ee_position'] = np.asarray(self.ee[arm][:3],float)
                held['ee_q'] = list(self.ee[arm][3:])
        return scene

    def object(self, identity):
        if self.scene is None or identity not in self.scene["objects"]:
            raise PolicyFailure("object_not_visible", identity)
        return deepcopy(self.scene["objects"][identity])

    def choose_arm(self, item, destination=None):
        candidates = [a for a in ("left", "right") if a not in self.holding]
        x = item["position"][0]
        preferred = "left" if x < 0 else "right"
        if preferred in candidates:
            return preferred
        if candidates:
            return candidates[0]
        raise PolicyFailure("both_arms_occupied")

    def move(self, arm, position, q, slow=False):
        vector(position, 3)
        rotation(q)
        result = yield from self.command("move", arm=arm, position=list(position),
                                       quaternion=list(q), slow=slow)
        self.ee[arm] = [*position, *q]
        return result

    def grip(self, arm, opened):
        result = yield from self.command("grip", arm=arm, value=1. if opened else 0.)
        if opened:self.closed_arms.discard(arm)
        else:self.closed_arms.add(arm)
        return result

    def wait(self, steps=3):
        yield from self.command("wait", steps=steps)

    def stable(self, identity, target=None, tolerance=.018, q=None):
        samples = []
        for _ in range(3):
            yield from self.wait(2)
            yield from self.observe()
            item = self.object(identity)
            if target is not None and np.linalg.norm(vector(item["position"], 3) - vector(target, 3)) > tolerance:
                raise PolicyFailure("placement_error", identity)
            if q is not None and self.orientation_error(item, q) > math.radians(15):
                raise PolicyFailure("orientation_error", identity)
            samples.append(vector(item["position"], 3))
        if np.max(np.ptp(samples, axis=0)) > self.memory.get("stability_m", .012):
            raise PolicyFailure("unstable_object", identity)
        return item

    @staticmethod
    def orientation_error(item, q):
        if item.get("features", {}).get("yaw_symmetric"):
            a, b = rotation(q).apply([0,0,1]), rotation(item["quaternion"]).apply([0,0,1])
            return math.acos(float(np.clip(np.dot(a,b),-1,1)))
        return (rotation(q).inv() * rotation(item["quaternion"])).magnitude()

    def grasp_orientation(self, item):
        return list(self.memory["grasp_quaternion"])

    def contact_pose(self, point, q):
        return vector(point,3)-rotation(q).apply(vector(self.memory['contact_offset_local'],3))

    def pick(self, identity, arm=None, rim=False):
        self.stage = "acquire:" + identity
        yield from self.observe()
        item = self.object(identity)
        arm = arm or self.choose_arm(item)
        if arm in self.holding:
            raise PolicyFailure("occupied_gripper", arm)
        point = vector(item.get("features", {}).get("grasp_point", item["position"]), 3)
        if rim:
            point = vector(feature(item, "rim_grasp"), 3)
        q = self.grasp_orientation(item)
        contact = self.contact_pose(point,q)
        above = contact.copy(); above[2] += self.memory["clearance_m"]
        yield from self.grip(arm, True)
        yield from self.move(arm, above, q)
        yield from self.move(arm, contact, q, slow=True)
        yield from self.grip(arm, False)
        yield from self.observe()
        closed = self.object(identity)
        actual = self.ee[arm]
        self.hypotheses[arm] = dict(id=identity,
            offset=rotation(actual[3:]).inv().apply(vector(closed["position"],3)-actual[:3]),
            rotation=rotation(actual[3:]).inv()*rotation(closed["quaternion"]))
        for fraction in (.5, 1.):
            lifted = contact.copy(); lifted[2] += self.memory["lift_m"] * fraction
            yield from self.move(arm, lifted, q, slow=True)
            yield from self.observe()
            current = self.object(identity)
            expected = vector(item["position"], 3) + [0, 0, self.memory["lift_m"] * fraction]
            if np.linalg.norm(vector(current["position"], 3) - expected) > self.memory["grasp_tolerance_m"]:
                raise PolicyFailure("missed_or_slipping_grasp", identity)
        current = yield from self.stable(identity, expected, self.memory["grasp_tolerance_m"])
        # Actual observed object-to-EEF transform, not an assumed zero offset.
        actual = self.ee[arm]
        lifted, q = np.array(actual[:3]), actual[3:]
        local_offset = rotation(q).inv().apply(vector(current["position"], 3) - lifted)
        relative_q = rotation(q).inv() * rotation(current["quaternion"])
        self.hypotheses.pop(arm,None)
        self.holding[arm] = {"id": identity, "offset": local_offset,
                             "rotation": relative_q, "ee_position": lifted, "ee_q": q}
        return arm

    def carry(self, arm, object_position, object_q=None, slow=False):
        held = self.holding[arm]
        object_q = object_q if object_q is not None else quaternion(rotation(held["ee_q"]) * held["rotation"])
        ee_r = rotation(object_q) * held["rotation"].inv()
        target = vector(object_position, 3) - ee_r.apply(held["offset"])
        q = quaternion(ee_r)
        yield from self.move(arm, target, q, slow=slow)
        held.update(ee_position=target, ee_q=q)

    def place(self, arm, destination, object_q=None, tolerance=.018, verify=True):
        held = self.holding[arm]
        identity = held["id"]
        self.stage = "place:" + identity
        object_q = object_q or self.object(identity)["quaternion"]
        destination = vector(destination, 3)
        clearance = destination.copy()
        clearance[2] = max(destination[2] + self.memory["clearance_m"],
                           held["ee_position"][2] + self.memory["clearance_m"] / 2)
        # Lift before translating: a diagonal shortcut can sweep through rims.
        current = vector(held["ee_position"], 3) + rotation(held["ee_q"]).apply(held["offset"])
        current[2] = clearance[2]
        yield from self.carry(arm, current, slow=True)
        yield from self.carry(arm, clearance, object_q)
        yield from self.carry(arm, destination, object_q, slow=True)
        yield from self.release_at(arm, destination, object_q, tolerance, verify)

    def release_at(self, arm, destination, object_q=None, tolerance=.018, verify=True):
        """Release on a task-specific contact path without a second approach."""
        held = self.holding[arm]
        identity = held["id"]
        yield from self.grip(arm, True)
        # The open command has been acknowledged. Stop predicting that this
        # object follows the gripper during withdrawal; verification below
        # must observe the released object at its support, not at the EEF.
        del self.holding[arm]
        retreat = vector(held["ee_position"],3); retreat[2] += self.memory["clearance_m"]
        yield from self.move(arm, retreat, held["ee_q"], slow=True)
        # Vacate the placement column before another arm uses that support.
        parking = [*self.home[arm][:2], retreat[2]]
        yield from self.move(arm, parking, held["ee_q"])
        if verify:
            yield from self.stable(identity, destination, tolerance, object_q)
        self.completed.append(identity)

    def transfer(self, identity, destination, q=None, rim=False, tolerance=.018):
        arm = yield from self.pick(identity, rim=rim)
        yield from self.place(arm, destination, q, tolerance)

    def home_arms(self):
        if self.holding:
            raise PolicyFailure("return_home_while_holding")
        self.stage = "return_home"
        for arm in ("left", "right"):
            pose = self.home[arm]
            current = list(self.ee[arm])
            high = max(current[2], pose[2], 1.12)
            yield from self.move(arm, [*current[:2], high], current[3:])
            yield from self.move(arm, [*pose[:2], high], pose[3:])
            yield from self.move(arm, pose[:3], pose[3:])
            yield from self.grip(arm, True)
        return {"status": "controller_finished", "objects": self.completed,
                "native_success": None}

    def press(self, identity, release=True):
        self.stage = "press:" + identity
        yield from self.observe()
        button = self.object(identity)
        top = vector(feature(button, "contact"), 3)
        direction = vector(feature(button, "press_axis"), 3)
        if abs(np.linalg.norm(direction) - 1) > 1e-3:
            raise PolicyFailure("press_axis")
        travel = feature(button, "travel_m")
        if not .003 <= travel <= .08:
            raise PolicyFailure("press_travel")
        arm = self.choose_arm(button)
        q = self.memory["grasp_quaternion"]
        offset = self.contact_pose([0,0,0],q)
        yield from self.grip(arm, False)
        yield from self.move(arm, top - direction * .04 + offset, q)
        yield from self.move(arm, top + direction * travel + offset, q, slow=True)
        yield from self.observe()
        pressed = vector(feature(self.object(identity), "contact"), 3)
        if np.dot(pressed - top, direction) < travel * .5:
            raise PolicyFailure("press_not_observed", identity)
        yield from self.move(arm, top - direction * .06 + offset, q)
        if release:
            yield from self.wait(4)
            yield from self.observe()
            if np.linalg.norm(vector(feature(self.object(identity), "contact"), 3) - top) > .008:
                raise PolicyFailure("release_not_observed", identity)

    def close_lid(self, identity):
        self.stage = "close_lid:" + identity
        yield from self.observe()
        lid = self.object(identity)
        hinge = vector(feature(lid, "hinge"), 3)
        axis = vector(feature(lid, "hinge_axis"), 3)
        edge = vector(feature(lid, "free_edge"), 3)
        angle = float(feature(lid, "closing_angle_rad"))
        if abs(np.linalg.norm(axis) - 1) > .001 or not .1 <= abs(angle) <= math.pi:
            raise PolicyFailure("hinge_geometry")
        arm = self.choose_arm(lid)
        q = self.memory["grasp_quaternion"]
        yield from self.grip(arm, False)
        above = edge + [0, 0, .08]
        yield from self.move(arm, self.contact_pose(above,q), q)
        for a in np.linspace(0, angle, 16):
            point = hinge + Rotation.from_rotvec(axis * a).apply(edge - hinge)
            yield from self.move(arm, self.contact_pose(point,q), q, slow=True)
        yield from self.move(arm, self.contact_pose(point+[0,0,.10],q), q)
        yield from self.observe()
        if abs(float(feature(self.object(identity), "closing_angle_rad"))) > .12:
            raise PolicyFailure("lid_not_closed")
