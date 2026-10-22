"""Parse explicit left/middle/right category destinations from instruction."""
import re
from ..control import groups, feature, packing_slots, PolicyFailure


def destinations(instruction):
    pairs = re.findall(r"(?:Put\s+)?([\w -]+?)\s+objects into the\s+(left|middle|right)\s+basket", instruction, re.I)
    mapping = {}
    for category, side in pairs:
        category = re.sub(r"^(?:and|put)\s+", "", category.strip(), flags=re.I).strip().lower()
        if side.lower() in mapping:
            raise PolicyFailure("ambiguous_instruction")
        mapping[side.lower()] = category
    if set(mapping) != {"left", "middle", "right"} or len(set(mapping.values())) != 3:
        raise PolicyFailure("unsupported_instruction")
    return mapping


def policy(robot, memory):
    mapping = destinations(robot.instruction)
    scene = yield from robot.observe()
    items = groups(scene, "item")
    baskets = sorted(groups(scene, "basket", 3), key=lambda x: x["position"][0])
    assigned = set()
    for side, basket in zip(("left", "middle", "right"), baskets):
        members = [x for x in items if feature(x, "category").lower() == mapping[side]]
        if not members:
            raise PolicyFailure("missing_instructed_category", mapping[side])
        slots = packing_slots(basket, members)
        for item in members:
            yield from robot.transfer(item["id"], slots[item["id"]], basket["quaternion"], tolerance=.025)
            assigned.add(item["id"])
    if len(assigned) != len(items):
        raise PolicyFailure("unassigned_object")
    return (yield from robot.home_arms())
