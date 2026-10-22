"""Four visible unoccupied wells, gentle placement, then articulated lid."""
from ..control import groups, feature, PolicyFailure


def policy(robot, memory):
    scene = yield from robot.observe()
    eggs = groups(scene, "egg", 4)
    holder, lid = groups(scene, "egg_holder", 1)[0], groups(scene, "lid", 1)[0]
    wells = feature(holder, "wells")
    if len(wells) < len(eggs) or len({tuple(point) for point in wells}) != len(wells):
        raise PolicyFailure("unresolved_wells")
    available = list(range(len(wells)))
    for egg in sorted(eggs, key=lambda x: x["position"][0]):
        yield from robot.observe()
        current = robot.object(egg['id'])
        # Stable well indices are carried in the holder's rigid frame, so a
        # displaced holder must not use world targets from the first frame.
        observed_wells = feature(robot.object(holder['id']), 'wells')
        if len(observed_wells) != len(wells):
            raise PolicyFailure('well_identity_changed')
        index = min(available, key=lambda k: sum((observed_wells[k][i] - current['position'][i])**2 for i in (0,1)))
        available.remove(index)
        well = observed_wells[index]
        point = [well[0], well[1], well[2] + current["size"][2] / 2]
        yield from robot.transfer(egg["id"], point, tolerance=.012)
    yield from robot.close_lid(lid["id"])
    return (yield from robot.home_arms())
