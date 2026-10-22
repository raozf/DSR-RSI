"""Conditional box orientation and per-category semantic fronts."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from ..control import groups,feature,rotation,quaternion,PolicyFailure,upright
from .common import align,floor_slots,footprint_inside,verify_direction


def policy(robot,memory):
    scene=yield from robot.observe();box=groups(scene,'box',1)[0];upright(box,7)
    r=rotation(feature(box,'interior_quaternion'))
    # The long internal axis is aligned to the task's +/- world X box axis.
    size=np.array(feature(box,'inner_size'));long=int(np.argmax(size));axis=r.as_matrix()[:,long]
    if abs(axis[0])<np.cos(np.deg2rad(30)):
        angle=-np.arctan2(axis[1],axis[0]);q=quaternion(Rotation.from_euler('z',angle)*rotation(box['quaternion']))
        yield from robot.transfer(box['id'],box['position'],q,tolerance=.012)
        yield from robot.observe();box=robot.object(box['id'])
    items=[groups(robot.scene,kind,1)[0] for kind in ('car','brush','hammer','shoe')]
    oriented=[];desired={};br=rotation(feature(box,'interior_quaternion'))
    for item in items:
        # Semantic front is measured separately for each object, not PCA sign.
        q=align(feature(item,'front_axis_local'),feature(item,'up_axis_local'),[1,0,0]);desired[item['id']]=q
        copy=deepcopy(item);copy['size']=(abs(br.as_matrix().T@rotation(q).as_matrix())@item['size']).tolist();oriented.append(copy)
    slots=floor_slots(box,oriented)
    box_initial=np.array(box['position'])
    for item in items:
        yield from robot.transfer(item['id'],slots[item['id']],desired[item['id']],tolerance=.012)
        yield from robot.observe();box=robot.object(box['id']);now=robot.object(item['id'])
        if np.linalg.norm(np.array(box['position'])-box_initial)>.008:raise PolicyFailure('box_moved_during_packing')
        if not footprint_inside(now,box):raise PolicyFailure('packed_footprint_unconfirmed')
        verify_direction(now,feature(now,'front_axis_local'),[1,0,0],35)
    for item in items:yield from robot.stable(item['id'])
    return (yield from robot.home_arms())
