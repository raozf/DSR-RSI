"""Two support pairs, two bridge boards, crown and roof: eight-piece tower."""
import numpy as np
from ..control import groups, feature, surface, upright, PolicyFailure


def policy(robot, memory):
    scene = yield from robot.observe()
    supports = groups(scene, "tower_support", 4)
    boards = sorted(groups(scene, "tower_board", 2), key=lambda x: x["position"][1], reverse=True)
    crown, roof = groups(scene, "tower_crown", 1)[0], groups(scene, "tower_roof", 1)[0]
    # One support from each initial side is required at each level. Preserve
    # those observed identities before any piece moves.
    lateral = sorted(supports, key=lambda x:x["position"][0])
    ordered = [lateral[0], lateral[2], lateral[1], lateral[3]]
    # Do not assemble on a board still lying on the table. Select a visibly
    # empty footprint; all coordinates come from the current scene.
    pieces = supports + boards + [crown, roof]
    half = np.array(boards[0]["size"][:2])/2 + .02
    options = [np.array([x,y]) for y in np.linspace(-.27,.04,9) for x in np.linspace(-.18,.18,9)]
    clear = [p for p in options if all(np.any(np.abs(p-np.array(o["position"][:2])) > half+np.array(o["size"][:2])/2) for o in pieces)]
    if not clear:
        raise PolicyFailure("no_clear_assembly_footprint")
    centre = min(clear,key=lambda p:sum(np.linalg.norm(p-np.array(o["position"][:2])) for o in pieces))
    span = boards[0]["size"][0] * .32
    if not .025 <= span <= .18:
        raise PolicyFailure("bridge_geometry")
    base_z = float(np.median([x["position"][2] - x["size"][2] / 2 for x in supports]))
    for layer in range(2):
        pair = ordered[layer * 2:layer * 2 + 2]
        tops = []
        for sign, item in zip((-1, 1), pair):
            point = [centre[0] + sign * span, centre[1], base_z + item["size"][2] / 2]
            yield from robot.transfer(item["id"], point, [1, 0, 0, 0], tolerance=.012)
            tops.append(base_z + item["size"][2])
        if abs(tops[0] - tops[1]) > .008:
            raise PolicyFailure("unequal_support_heights")
        board = boards[layer]
        point = [*centre, max(tops) + board["size"][2] / 2]
        yield from robot.transfer(board["id"], point, [1, 0, 0, 0], tolerance=.012)
        yield from robot.observe()
        placed = robot.object(board["id"])
        upright(placed, tolerance=5)
        base_z = surface(placed)
    for item in (crown, roof):
        q = feature(item, "assembly_quaternion")
        height = feature(item, "assembly_height")
        point = [*centre, base_z + height / 2]
        yield from robot.transfer(item["id"], point, q, tolerance=.012)
        base_z += height
    return (yield from robot.home_arms())
