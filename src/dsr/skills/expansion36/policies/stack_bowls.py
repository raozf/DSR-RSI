"""Rim grasp and measured bowl nesting, checking the supporting bowl each time."""
from ..control import groups, feature, upright


def policy(robot, memory):
    scene = yield from robot.observe()
    bowls = sorted(groups(scene, "bowl", 3), key=lambda x: (-feature(x, "rim_diameter"), x["id"]))
    support_id = bowls[0]["id"]
    for item in bowls[1:]:
        yield from robot.observe()
        support = robot.object(support_id)
        upright(support, tolerance=6 if support_id == bowls[0]["id"] else 20)
        current = robot.object(item["id"])
        # Calibrated nesting ratio is a development parameter, not a layout.
        height = current["size"][2] * memory["nesting_fraction"]
        destination = [*support["position"][:2], support["position"][2] + height]
        yield from robot.transfer(current["id"], destination, support["quaternion"], rim=True, tolerance=.02)
        support_id = current["id"]
    return (yield from robot.home_arms())
