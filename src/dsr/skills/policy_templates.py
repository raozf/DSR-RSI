"""Source fragments compiled into import-free native Code Policy candidates.

Shared fragments deliberately emit direct calls to the frozen primitive API.
The generated source, rather than this templating code, is hashed and executed.
"""

PREAMBLE = '''
    configuration = manipulation_parameters(memory)
    scene = robot.scene()
    completed = []
'''

GUARDED_PICK = '''
        initial = observed_object(scene, name, None)
        arm = "left" if initial["pose"][0] < 0 else "right"
        quaternion = configuration["grasp_quaternions"][arm]
        contact = grasp_target(initial, arm, configuration, RIM)
        above = list(contact)
        above[2] += configuration["clearance_m"]
        robot.gripper(arm=arm, command=configuration["open_grip"])
        robot.approach(arm=arm, position_m=above, quaternion_wxyz=quaternion)
        robot.approach(arm=arm, position_m=contact, quaternion_wxyz=quaternion)
        robot.gripper(arm=arm, command=configuration["closed_grip"])
        before = initial["observation"]
        for fraction in (0.5, 1.0):
            lifted = list(contact)
            lifted[2] += configuration["lift_m"] * fraction
            robot.approach(arm=arm, position_m=lifted, quaternion_wxyz=quaternion)
            scene = robot.scene()
            current = observed_object(scene, name, before)
            lift_progress(initial, current, configuration["lift_m"] * fraction,
                          configuration["geometry_tolerance_m"])
            before = current["observation"]
        samples = [current]
        for sample_index in range(2):
            robot.gripper(arm=arm, command=configuration["closed_grip"])
            scene = robot.scene()
            current = observed_object(scene, name, before)
            lift_progress(initial, current, configuration["lift_m"],
                          configuration["geometry_tolerance_m"])
            samples.append(current)
            before = current["observation"]
        stable_geometry(samples, configuration["stability_tolerance_m"])
'''

PLACE = '''
        # Preserve the measured object-to-gripper offset throughout transfer.
        destination = [desired[i] + contact[i] - initial["pose"][i] for i in range(3)]
        clearance = list(destination)
        clearance[2] = max(destination[2] + configuration["clearance_m"], lifted[2])
        robot.approach(arm=arm, position_m=clearance, quaternion_wxyz=quaternion)
        robot.approach(arm=arm, position_m=destination, quaternion_wxyz=quaternion)
        robot.gripper(arm=arm, command=configuration["open_grip"])
        robot.approach(arm=arm, position_m=clearance, quaternion_wxyz=quaternion)
        samples = []
        for sample_index in range(3):
            robot.gripper(arm=arm, command=configuration["open_grip"])
            scene = robot.scene()
            current = observed_object(scene, name, before)
            samples.append(current)
            before = current["observation"]
        released_stability(samples, configuration["stability_tolerance_m"])
'''

HOME = '''
    for arm in ("left", "right"):
        home = configuration["home_poses"][arm]
        robot.approach(arm=arm, position_m=home[:3], quaternion_wxyz=home[3:])
        robot.gripper(arm=arm, command=configuration["open_grip"])
    return {"status": "visually_completed", "completed": completed,
            "scope": "Controller feedback only; native task checker decides success"}
'''


def policy_body(name: str) -> str:
    if name == "general-pickup-stable":
        return PREAMBLE + '''
    if configuration["lift_m"] < 0.12:
        raise ValueError("General pickup requires clearance beyond 10 cm")
    for name in [memory["target"]]:
''' + GUARDED_PICK.replace("RIM", "False") + '''
        completed.append(name)
    return {"status": "stable_lift_observed", "completed": completed,
            "scope": "Controller feedback only; native task checker decides success"}
'''
    if name == "stack-bowls-feedback":
        return PREAMBLE + '''
    names = memory["targets"]
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("Stacking requires three distinct observed bowl identities")
    ledger = identity_ledger(names)
    estimates = {name: observed_object(scene, name, None) for name in names}
    # Widest visible bowl provides the base; no fixed world layout is supplied.
    base_name = max(names, key=lambda item: estimates[item]["dims"][0] * estimates[item]["dims"][1])
    ledger[base_name]["status"] = "support"
    base = estimates[base_name]
    quaternion = base["pose"][3:]
    if 1 - 2 * (quaternion[1]**2 + quaternion[2]**2) < 0.992546151641322:
        raise ValueError("Base bowl is not visibly upright within seven degrees")
    nesting_fraction = memory["nesting_fraction"]
    if not isinstance(nesting_fraction, (int, float)) or isinstance(nesting_fraction, bool) or not 0.1 <= nesting_fraction <= 0.9:
        raise ValueError("An explicit development-calibrated nesting fraction is required")
    for name in [item for item in names if item != base_name]:
        if ledger[name]["status"] != "unseen":
            raise ValueError("Object identity was scheduled twice")
        ledger[name]["attempts"] += 1
''' + GUARDED_PICK.replace("RIM", "True") + '''
        support = observed_object(scene, base_name, None)
        desired = [support["pose"][0], support["pose"][1],
                   support["pose"][2] + initial["dims"][2] * nesting_fraction]
''' + PLACE + '''
        support_after = observed_object(scene, base_name, None)
        relative_alignment(current, support_after, 0.03, position_axes=(0, 1), require_orientation=False)
        if current["pose"][2] <= support_after["pose"][2]:
            raise ValueError("Bowl was not visibly placed above its support")
        quaternion_after = current["pose"][3:]
        if 1 - 2 * (quaternion_after[1]**2 + quaternion_after[2]**2) < 0.7071067811865476:
            raise ValueError("Placed bowl is tilted by more than 45 degrees")
        completed.append(name)
        ledger[name]["status"] = "completed"
        base_name = name
''' + HOME.replace('return {"status": "visually_completed", "completed": completed,', 'return {"status": "visually_completed", "completed": completed, "ledger": ledger,')
    if name == "stack-blocks-feedback":
        return PREAMBLE + '''
    names = memory["targets"]
    base_name = memory["base"]
    if len(names) != 3 or len(set(names)) != 3 or base_name not in names:
        raise ValueError("Stacking requires three distinct blocks and an explicit base identity")
    ledger = identity_ledger(names)
    for name in names:
        observed_object(scene, name, None)
    placement_tolerance = memory["placement_tolerance_m"]
    if not isinstance(placement_tolerance, (int, float)) or isinstance(placement_tolerance, bool) or not 0 < placement_tolerance <= 0.015:
        raise ValueError("Block placement needs an explicit precision tolerance")
    ledger[base_name]["status"] = "support"
    for name in [item for item in names if item != base_name]:
        if ledger[name]["status"] != "unseen":
            raise ValueError("Object identity was scheduled twice")
        ledger[name]["attempts"] += 1
''' + GUARDED_PICK.replace("RIM", "False") + '''
        support = observed_object(scene, base_name, None)
        desired = [support["pose"][0], support["pose"][1],
                   support["pose"][2] + (support["dims"][2] + initial["dims"][2]) / 2]
''' + PLACE + '''
        support_after = observed_object(scene, base_name, None)
        relative_alignment(current, support_after, placement_tolerance, position_axes=(0, 1), require_orientation=False)
        expected_height = (support_after["dims"][2] + current["dims"][2]) / 2
        if abs(current["pose"][2] - support_after["pose"][2] - expected_height) > placement_tolerance:
            raise ValueError("Block has not settled at the observed support height")
        completed.append(name)
        ledger[name]["status"] = "completed"
        base_name = name
''' + HOME.replace('return {"status": "visually_completed", "completed": completed,', 'return {"status": "visually_completed", "completed": completed, "ledger": ledger,')
    if name == "bottles-to-dustbin-feedback":
        return PREAMBLE + '''
    names = memory["targets"]
    if len(names) != 4 or len(set(names)) != 4 or memory["container"] in names:
        raise ValueError("Disposal requires four distinct bottles and one container")
    ledger = identity_ledger(names)
    margin = memory["placement_margin_m"]
    if not isinstance(margin, (int, float)) or isinstance(margin, bool) or not 0 < margin <= 0.05:
        raise ValueError("Positive placement clearance is required")
    for name in names:
        if ledger[name]["status"] != "unseen":
            raise ValueError("Object identity was scheduled twice")
        ledger[name]["attempts"] += 1
''' + GUARDED_PICK.replace("RIM", "False") + '''
        container = observed_object(scene, memory["container"], None)
        if any(container["dims"][axis] - initial["dims"][axis] <= 2 * margin for axis in (0, 1)):
            raise ValueError("Observed opening has insufficient placement clearance")
        # Place near the observed container bottom; release is not a success test.
        desired = [container["pose"][0], container["pose"][1],
                   container["pose"][2] - container["dims"][2] / 2 + initial["dims"][2] / 2 + margin]
''' + PLACE + '''
        container_after = observed_object(scene, memory["container"], None)
        placement_inside(container_after, current, margin)
        completed.append(name)
        ledger[name]["status"] = "completed"
''' + HOME.replace('return {"status": "visually_completed", "completed": completed,', 'return {"status": "visually_completed", "completed": completed, "ledger": ledger,')
    raise ValueError("Unknown candidate Code Policy: " + name)
