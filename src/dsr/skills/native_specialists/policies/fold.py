"""Deformable landmarks with sleeves and hem; no rigid-shirt transport."""
import numpy as np
from ..control import groups,feature,PolicyFailure


def policy(robot,memory):
    cloth=groups((yield from robot.observe()),'garment',1)[0];identity=cloth['id']
    # Exit both origin trigger regions before any fold, then return only once.
    robot.stage='leave_origin_before_fold'
    for arm in ('left','right'):
        p=np.array(robot.ee[arm][:3]);p[1]+=.07
        yield from robot.move(arm,p,robot.ee[arm][3:])
    for source,target in [('left_sleeve','right_chest'),('right_sleeve','left_chest'),('left_hem','left_shoulder'),('right_hem','right_shoulder')]:
        yield from robot.fold_point(identity,source,target)
    yield from robot.observe();f=feature(robot.object(identity),'landmarks')
    for source,target in [('left_sleeve','right_chest'),('right_sleeve','left_chest'),('left_hem','left_shoulder'),('right_hem','right_shoulder')]:
        if np.linalg.norm(np.asarray(f[source])[:2]-np.asarray(f[target])[:2])>.065:raise PolicyFailure('fold_final_landmarks')
    a=np.asarray(f['right_hem'])[:2]-np.asarray(f['left_hem'])[:2]
    b=np.asarray(f['right_shoulder'])[:2]-np.asarray(f['left_shoulder'])[:2]
    if min(np.linalg.norm(a),np.linalg.norm(b))<.03 or abs(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b))<np.cos(np.deg2rad(40)):
        raise PolicyFailure('fold_line_alignment')
    return (yield from robot.home_arms())
