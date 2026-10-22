"""Exactly three transfers and three observed press/release cycles."""
import numpy as np
from ..control import groups, surface, PolicyFailure


def policy(robot, memory):
    scene = yield from robot.observe()
    a, b = groups(scene, "block", 2)
    pads, buttons = groups(scene, "pad", 3), groups(scene, "button", 1)
    def nearest(item):
        ranked = sorted(pads, key=lambda p: np.linalg.norm(np.array(item["position"][:2]) - p["position"][:2]))
        if np.linalg.norm(np.array(item["position"][:2]) - ranked[0]["position"][:2]) > .045:
            raise PolicyFailure("initial_pad_ambiguous")
        return ranked[0]
    pa, pb = nearest(a), nearest(b)
    if pa["id"] == pb["id"]:
        raise PolicyFailure("initial_pad_overlap")
    empty = next(p for p in pads if p["id"] not in {pa["id"], pb["id"]})
    for item, target in [(a, empty), (b, pa), (a, pb)]:
        yield from robot.observe()
        current, pad = robot.object(item["id"]), robot.object(target["id"])
        point = [*pad["position"][:2], surface(pad) + current["size"][2] / 2]
        yield from robot.transfer(item["id"], point, tolerance=.018)
        # Never retry an ambiguous press: the task rejects a fourth event.
        yield from robot.press(buttons[0]["id"], release=True)
    return (yield from robot.home_arms())
