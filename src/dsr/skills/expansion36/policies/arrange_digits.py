"""Stable digit identities, descending order and explicit orientation repair."""
from ..control import groups, feature, surface, PolicyFailure


def policy(robot, memory):
    scene = yield from robot.observe()
    digits = groups(scene, "digit")
    pads = sorted(groups(scene, "pad"), key=lambda x: x["position"][0])
    if len(digits) != len(pads) or len(digits) not in (4, 5):
        raise PolicyFailure("digit_pad_count")
    readings = [(feature(x, "digit"), x["id"]) for x in digits]
    if any(type(n) is not int or not 0 <= n <= 9 for n, _ in readings):
        raise PolicyFailure("unreadable_digit")
    # Duplicate digits are allowed: stable identities break numeric ties.
    order = sorted(readings, key=lambda x: (-x[0], x[1]))
    occupied = set()
    for (_, identity), pad in zip(order, pads):
        yield from robot.observe()
        item, target = robot.object(identity), robot.object(pad["id"])
        destination = [*target["position"][:2], surface(target) + item["size"][2] / 2]
        # Upright glyph in the table plane; not arbitrary box-PCA sign.
        q = feature(item, "upright_quaternion")
        yield from robot.transfer(identity, destination, q, tolerance=.015)
        occupied.add(pad["id"])
    if len(occupied) != len(digits):
        raise PolicyFailure("duplicate_pad")
    return (yield from robot.home_arms())
