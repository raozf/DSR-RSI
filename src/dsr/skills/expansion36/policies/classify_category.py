"""Capacity-aware category grouping with permanent basket assignments."""
from ..control import groups, feature, packing_slots, PolicyFailure


def policy(robot, memory):
    scene = yield from robot.observe()
    items = groups(scene, "item")
    baskets = sorted(groups(scene, "basket", 3), key=lambda x: x["position"][0])
    categories = sorted({feature(item, "category") for item in items})
    if len(categories) != 3:
        raise PolicyFailure("ambiguous_categories")
    # Categories without named destinations are assigned once. Never re-sort
    # the category list after removal of one object from the visible table.
    for category, basket in zip(categories, baskets):
        members = [x for x in items if feature(x, "category") == category]
        slots = packing_slots(basket, members)
        for item in members:
            yield from robot.transfer(item["id"], slots[item["id"]], basket["quaternion"], tolerance=.025)
    return (yield from robot.home_arms())
