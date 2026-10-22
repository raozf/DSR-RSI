"""Resume a registered placement from code-owned targets and measured ownership.

The planner can select this recovery but cannot supply its geometry. An uncertain
grasp remains uncertain; only an existing verified attachment or a visibly stable
release with measured open grippers permits continuation.
"""
from copy import deepcopy
import numpy as np

from .control import PolicyFailure, rotation, quaternion


PENDING = 'strategy.pending_placement'


def anchor(item, path):
    value = item['features']
    for key in path:
        value = value[key]
    return np.asarray(value, float)


def remember_placement(robot, identity, target, q, tolerance, support, path):
    previous = robot.progress.get(PENDING)
    if previous and previous['identity'] != identity:
        raise PolicyFailure('pending_placement_requires_resolution')
    reference = robot.object(support)
    robot.progress[PENDING] = dict(identity=identity, destination=list(target),
        quaternion=list(q) if q is not None else None, tolerance=tolerance,
        support=support, support_quaternion=reference['quaternion'], anchor_path=list(path),
        anchor=anchor(reference,path).tolist(), scene_sequence=robot.sequence)


def forget_placement(robot, identity):
    if robot.progress.get(PENDING,{}).get('identity') == identity:
        del robot.progress[PENDING]


def resume_placement(robot, memory, value):
    checkpoint = deepcopy(robot.progress.get(PENDING))
    if not checkpoint:
        raise PolicyFailure('no_registered_pending_placement')
    if robot.hypotheses:
        raise PolicyFailure('pending_placement_attachment_uncertain')
    identity = checkpoint['identity']
    if len(robot.holding)>1 or any(h['id']!=identity for h in robot.holding.values()):
        raise PolicyFailure('pending_placement_ownership_mismatch')
    yield from robot.observe()
    support = robot.object(checkpoint['support'])
    if (np.linalg.norm(anchor(support,checkpoint['anchor_path'])-checkpoint['anchor'])>.006
            or robot.orientation_error(support,checkpoint['support_quaternion'])>np.deg2rad(8)):
        raise PolicyFailure('pending_placement_support_moved')
    target, q = checkpoint['destination'], checkpoint['quaternion']
    if robot.holding:
        arm, held = next(iter(robot.holding.items()))
        measured = robot.scene['robot_state'][arm+'_ee_joint_state'][0]
        if measured<.95:
            item = robot.object(identity);pose = robot.ee[arm]
            expected = np.asarray(pose[:3])+rotation(pose[3:]).apply(held['offset'])
            expected_q = quaternion(rotation(pose[3:])*held['rotation'])
            if (np.linalg.norm(np.asarray(item['position'])-expected)>memory['grasp_tolerance_m']
                    or robot.orientation_error(item,expected_q)>np.deg2rad(15)):
                raise PolicyFailure('pending_placement_attachment_unconfirmed')
            yield from robot.place(arm,target,q,checkpoint['tolerance'])
        else:
            # Opening may have physically executed before its acknowledgment
            # was interrupted. Prove a stable release before retiring ownership.
            if robot.closed_arms:
                raise PolicyFailure('pending_placement_ownership_mismatch')
            yield from robot.stable(identity,target,checkpoint['tolerance'],q)
            del robot.holding[arm]
    else:
        if robot.closed_arms:
            raise PolicyFailure('pending_placement_attachment_uncertain')
        yield from robot.stable(identity,target,checkpoint['tolerance'],q)
    if identity not in robot.completed:
        robot.completed.append(identity)
    forget_placement(robot,identity)
    # This is a recovered primitive, not a verified stage receipt or native win.
    return dict(recovered_placement=identity,native_success=None)
