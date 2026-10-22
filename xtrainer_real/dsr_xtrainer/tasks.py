"""New real-hardware ports of two DSR task semantics; native controllers stay intact."""
import numpy as np
from .config import bounded
from .geometry import vector
from .model import Plan, PreparationError
from .skills import Skills
from .safety import validate_plan


def align_blocks(config, scene):
    objects = scene["objects"]
    cubes = [objects["cube"+str(i)] for i in range(3)]
    edge = objects["straightedge"]
    if edge.get("arm") not in {"left", "right"}:
        raise PreparationError("straightedge.arm must be left or right")
    side = edge["arm"]
    contact = vector(edge.get("contact_tcp_world_mm_deg"), 6, "straightedge contact TCP")
    center = vector(edge["position_world_mm"], 3, "straightedge edge midpoint")
    length = bounded(edge.get("length_mm"), 30, 600, "straightedge.length_mm")
    if edge.get("axis_world") != [1, 0, 0] or edge.get("push_axis_world") != [0, 1, 0]:
        raise PreparationError("This qualified fixture uses an X-aligned ruler pushed along +Y")
    faces = []
    for cube in cubes:
        p = vector(cube["position_world_mm"], 3, "cube position")
        size = bounded(cube.get("size_mm"), 5, 100, "cube.size_mm")
        yaw = bounded(cube.get("yaw_deg"), -360, 360, "cube.yaw_deg")
        if abs((yaw+45) % 90-45) > 8:
            raise PreparationError("Pre-align cube axes within 8 degrees; a planar ruler sweep does not correct arbitrary rotations")
        if abs(p[0]-center[0])+size/2+5 > length/2:
            raise PreparationError("Straightedge does not span all three blocks with 5 mm clearance")
        faces.append(p[1]-size/2)
    if center[1] > min(faces)-1:
        raise PreparationError("Straightedge edge must initially be behind all blocks along +Y")
    distance = max(faces)-center[1]+2  # common lower edge, plus 2 mm contact margin
    if not 1 < distance <= bounded(config.data["tasks"].get("max_ruler_sweep_mm"), 1, 100, "max_ruler_sweep_mm"):
        raise PreparationError("Required ruler sweep exceeds calibrated max_ruler_sweep_mm")
    s = Skills(config)
    end = contact.copy()
    end[1] += distance
    s.grip(side, 0.0, "ruler.close_pusher")
    s.checkpoint("ruler.fixture_check", {"kind": "grasp", "object": "straightedge"},
                 f"Confirm the measured straightedge is securely held by the {side} gripper at park before approaching the blocks.")
    s.move(side, s.above(contact), "ruler.approach")
    s.move(side, contact, "ruler.contact")
    s.move(side, end, "ruler.sweep")
    s.move(side, s.above(end), "ruler.retract")
    s.park(side, "ruler.return")
    s.checkpoint("alignment.final", {"kind": "alignment"},
                 "Confirm 3 blocks form a straight row (<=8 mm / 8 deg), no block lifted >7 mm; both arms at park.")
    return Plan("align_blocks", s.steps, {"sweep_mm": float(distance), "arm": side,
                "initial_objects": objects, "fixture": "X ruler, +Y push, no grasp or lift of blocks"})


def cover_blocks(config, scene):
    objects = scene["objects"]
    labels = ["red", "green", "blue"]
    positions = {label: vector(objects[label]["position_world_mm"], 3, label) for label in labels}
    order = sorted(labels, key=lambda label: positions[label][0])
    if min(np.diff(sorted(p[0] for p in positions.values()))) < 20:
        raise PreparationError("Blocks must have a clear left-to-right X order (>=20 mm)")
    cups = [objects["cup"+str(i)] for i in range(3)]
    assignment, covered, s = {}, [], Skills(config)
    for label, cup in zip(order, cups):
        side = cup.get("arm")
        if side not in {"left", "right"} or cup.get("opening_down") is not True:
            raise PreparationError("Each cup needs an arm and must initially be inverted (opening_down=true)")
        cube = objects[label]
        size = bounded(cube.get("size_mm"), 5, 100, label+".size_mm")
        diameter = bounded(cup.get("inner_diameter_mm"), 5, 300, "cup.inner_diameter_mm")
        height = bounded(cup.get("inner_height_mm"), 5, 300, "cup.inner_height_mm")
        if diameter <= size*np.sqrt(2)+8 or height <= size+5:
            raise PreparationError("Cup must clear the block diagonal by 8 mm and height by 5 mm")
        source = vector(cup.get("grasp_tcp_world_mm_deg"), 6, "cup grasp TCP")
        target = source.copy()
        target[:2] += positions[label][:2]-vector(cup["position_world_mm"], 3, "cup")[:2]
        assignment[label] = {"cup": "cup"+str(len(assignment)), "arm": side,
                             "source_tcp": source.tolist(), "cover_tcp": target.tolist()}
    # Remember color -> cup -> arm BEFORE occlusion. Never re-sort after covering.
    for i, label in enumerate(order):
        a = assignment[label]
        stage = f"cover.{i+1}.{label}"
        s.transfer(a["arm"], a["source_tcp"], a["cover_tcp"], a["cup"], stage)
        covered.append(label)
        s.checkpoint(stage+".state", {"kind": "cover_state", "covered_labels": covered.copy()},
                     f"Confirm only {', '.join(covered)} are covered; all cups remain inverted; blocks moved <50 mm.")
    for i, label in enumerate(labels):
        a = assignment[label]
        stage = f"uncover.{i+1}.{label}"
        s.transfer(a["arm"], a["cover_tcp"], a["source_tcp"], a["cup"], stage)
        covered.remove(label)
        s.checkpoint(stage+".state", {"kind": "cover_state", "covered_labels": covered.copy()},
                     f"Confirm {label} has been uncovered in red/green/blue order; remaining covered: {covered}.")
    return Plan("cover_blocks", s.steps, {"cover_order": order, "uncover_order": labels,
                "assignment": assignment, "initial_objects": objects})


def build_plan(config, scene, task):
    try:
        plan = {"align_blocks": align_blocks, "cover_blocks": cover_blocks}[task](config, scene)
    except (KeyError, TypeError, IndexError) as exc:
        raise PreparationError(f"Incomplete {task} scene/config: missing or invalid {exc}") from exc
    validate_plan(config, plan)
    return plan
