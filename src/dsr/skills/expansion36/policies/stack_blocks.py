"""Three-block tower with support-centred placement and no base-ID shortcut."""
from ..control import groups, surface, upright


def policy(robot, memory):
    scene = yield from robot.observe()
    blocks = groups(scene, "block", 3)
    ordered = sorted(blocks, key=lambda x: (-x["size"][0] * x["size"][1], x["id"]))
    support_id = ordered[0]["id"]
    for item in ordered[1:]:
        yield from robot.observe()
        support, current = robot.object(support_id), robot.object(item["id"])
        upright(support)
        point = [*support["position"][:2], surface(support) + current["size"][2] / 2]
        yield from robot.transfer(current["id"], point, support["quaternion"], tolerance=.012)
        support_id = current["id"]
    return (yield from robot.home_arms())
