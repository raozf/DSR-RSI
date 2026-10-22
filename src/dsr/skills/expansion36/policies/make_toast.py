"""Insert exactly two of four slices, preserve two on shelf, depress lever."""
from ..control import groups, feature, PolicyFailure


def policy(robot, memory):
    scene = yield from robot.observe()
    breads = groups(scene, "bread", 4)
    toaster, shelf = groups(scene, "toaster", 1)[0], groups(scene, "bread_shelf", 1)[0]
    slots = feature(toaster, "slots")
    if len(slots) != 2:
        raise PolicyFailure("toast_slot_count")
    chosen = sorted(breads, key=lambda x: sum((x["position"][i] - toaster["position"][i])**2 for i in (0,1)))[:2]
    for bread, slot in zip(chosen, slots):
        target = list(slot["point"]); target[2] += memory["toast_exposed_m"]
        yield from robot.transfer(bread["id"], target, slot["quaternion"], tolerance=.015)
    # Do not move the remaining two unless visibly outside their shelf.
    for item in breads:
        if item not in chosen:
            yield from robot.observe()
            current = robot.object(item["id"])
            if any(abs(current["position"][i]-shelf["position"][i]) > shelf["size"][i]/2 for i in (0,1)):
                raise PolicyFailure("remaining_bread_off_shelf")
    lever = groups(robot.scene, "lever", 1)[0]
    yield from robot.press(lever["id"], release=False)
    return (yield from robot.home_arms())
