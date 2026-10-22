"""Eight ordered key strokes using an observed mallet-tip transform."""
import numpy as np
from ..control import groups, feature, rotation, vector, PolicyFailure


def policy(robot, memory):
    scene = yield from robot.observe()
    mallet = groups(scene, "mallet", 1)[0]
    instrument = groups(scene, "xylophone", 1)[0]
    if feature(instrument, "key_order_source") != "visible_low_note_endpoint":
        raise PolicyFailure("xylophone_order_unresolved")
    keys = feature(instrument, "keys")
    if len(keys) != 8:
        raise PolicyFailure("key_count")
    tip = vector(feature(mallet, "tip_local"), 3)
    arm = yield from robot.pick(mallet["id"])
    q = robot.object(mallet["id"])["quaternion"]
    for i, point in enumerate(keys):
        robot.stage = "strike_key:" + str(i)
        centre = vector(point, 3) - rotation(q).apply(tip)
        # Source task specifies a 1--3.6 cm tip-height interval, not a force
        # sensor. Report only observed geometry, never invented contact.
        strike = centre + [0, 0, .02]
        raised = strike + [0, 0, .05]
        yield from robot.carry(arm, raised, q)
        yield from robot.carry(arm, strike, q, slow=True)
        yield from robot.observe()
        current = robot.object(mallet["id"])
        actual = vector(current["position"],3) + rotation(current["quaternion"]).apply(feature(current,"tip_local"))
        if np.linalg.norm(actual - (vector(point,3)+[0,0,.02])) > .012:
            raise PolicyFailure("mallet_tip_misalignment")
        if i < 7:
            yield from robot.carry(arm, raised, q, slow=True)
    return {"status":"controller_finished", "strikes":8, "native_success":None}
