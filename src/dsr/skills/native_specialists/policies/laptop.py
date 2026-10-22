"""Protect released headset while closing and inserting laptop."""
import numpy as np
from ..control import groups,feature,rotation,PolicyFailure
from .common import align
from ..resumption import remember_placement,forget_placement

PERCEPTION_PROFILES = ('independent_parts',)


def place_headset(robot,memory):
    scene=yield from robot.observe(required={
        'headset':dict(count=1,features=['bridge_axis_local','bridge_local','grasp']),
        'stand':dict(count=1,features=['cradle_axis','cradle'])})
    headset=groups(scene,'headset',1)[0];stand=groups(scene,'stand',1)[0]
    q=align(feature(headset,'bridge_axis_local'),[0,0,1],feature(stand,'cradle_axis'))
    yield from robot.place_point(headset['id'],feature(headset,'bridge_local'),feature(stand,'cradle'),q,.015)
    yield from robot.observe();robot.protect(headset['id'],.008)
    return dict(headset=headset['id'],stand=stand['id'])


def check_headset(robot,context):
    headset=robot.object(context['headset']);stand=robot.object(context['stand'])
    bridge=np.array(headset['position'])+rotation(headset['quaternion']).apply(feature(headset,'bridge_local'))
    if np.linalg.norm(bridge-feature(stand,'cradle'))>.020:
        raise PolicyFailure('headset_not_in_cradle')
    groups(robot.scene,'laptop',1);groups(robot.scene,'laptop_rack',1)


def finish_laptop(robot,memory,context):
    laptop=groups(robot.scene,'laptop',1)[0]
    if abs(feature(laptop,'closing_angle_rad'))>.12:yield from robot.close_lid(laptop['id'])
    yield from robot.observe();laptop=robot.object(laptop['id']);rack=groups(robot.scene,'laptop_rack',1)[0]
    if abs(feature(laptop,'closing_angle_rad'))>.12:raise PolicyFailure('laptop_hinge_not_closed')
    axis=feature(laptop,'storage_axis_local');q=align(feature(laptop,'hinge_axis_local'),axis,feature(rack,'slot_axis'))
    across=np.array(feature(rack,'slot_across'))
    thickness=float(np.sum(abs(across@rotation(q).as_matrix())*np.asarray(laptop['size'])))
    if thickness+.006>feature(rack,'slot_width_m'):raise PolicyFailure('laptop_slot_clearance')
    if 'laptop.finish' in robot.progress:
        destination=np.array(feature(rack,'opening'))-rotation(q).apply(feature(laptop,'bottom_local'))
        remember_placement(robot,laptop['id'],destination,q,.012,rack['id'],['opening'])
    arm=yield from robot.pick(laptop['id']);yield from robot.observe()
    laptop=robot.object(laptop['id']);point=np.array(feature(rack,'opening'))
    destination=point-rotation(q).apply(feature(laptop,'bottom_local'))
    if 'laptop.finish' in robot.progress:
        robot.progress['laptop.finish']=dict(identity=laptop['id'],rack=rack['id'],
            opening=point.tolist(),destination=destination.tolist(),quaternion=list(q))
        remember_placement(robot,laptop['id'],destination,q,.012,rack['id'],['opening'])
    yield from robot.carry(arm,destination+[0,0,.09],q)
    for z in (.06,.03,0.):
        yield from robot.carry(arm,destination+[0,0,z],q,slow=True)
        yield from robot.observe()  # Includes protected headset check.
    yield from robot.release_at(arm,destination,q,.012)
    yield from robot.observe();now=robot.object(laptop['id'])
    if abs(feature(now,'closing_angle_rad'))>.12:raise PolicyFailure('laptop_reopened')
    if 'laptop.finish' in robot.progress:forget_placement(robot,laptop['id'])
    return (yield from robot.home_arms())


def policy(robot,memory):
    context=yield from place_headset(robot,memory)
    return (yield from finish_laptop(robot,memory,context))


def resume_place_headset(robot,memory):
    robot.stage='place_headset:perception'
    scene=yield from robot.observe(required={
        'headset':dict(count=1,features=['bridge_axis_local','bridge_local','grasp']),
        'stand':dict(count=1,features=['cradle_axis','cradle'])})
    headset=groups(scene,'headset',1)[0];stand=groups(scene,'stand',1)[0]
    key='laptop.place_headset'
    if key not in robot.progress:
        robot.progress[key]=dict(headset=headset['id'],stand=stand['id'])
    context=robot.progress[key]
    if context!=dict(headset=headset['id'],stand=stand['id']):
        raise PolicyFailure('headset_or_stand_identity_changed')
    if headset['id'] not in robot.protected:
        q=align(feature(headset,'bridge_axis_local'),[0,0,1],feature(stand,'cradle_axis'))
        target=np.array(feature(stand,'cradle'))-rotation(q).apply(feature(headset,'bridge_local'))
        if np.linalg.norm(np.array(headset['position'])-target)<.015 and robot.orientation_error(headset,q)<np.deg2rad(15):
            yield from robot.stable(headset['id'],target,.015,q)
        else:
            remember_placement(robot,headset['id'],target,q,.015,stand['id'],['cradle'])
            yield from robot.place_point(headset['id'],feature(headset,'bridge_local'),feature(stand,'cradle'),q,.015)
        yield from robot.observe();robot.protect(headset['id'],.008)
        forget_placement(robot,headset['id'])
    return context


def resume_finish_laptop(robot,memory,context):
    key='laptop.finish'
    checkpoint=robot.progress.setdefault(key,{})
    if checkpoint:
        yield from robot.observe()
        now=robot.object(checkpoint['identity']);rack=robot.object(checkpoint['rack'])
        if np.linalg.norm(np.array(feature(rack,'opening'))-checkpoint['opening'])>.006:
            raise PolicyFailure('laptop_rack_moved')
        if (abs(feature(now,'closing_angle_rad'))<=.12
                and np.linalg.norm(np.array(now['position'])-checkpoint['destination'])<.012
                and robot.orientation_error(now,checkpoint['quaternion'])<np.deg2rad(15)):
            yield from robot.stable(now['id'],checkpoint['destination'],.012,checkpoint['quaternion'])
            if now['id'] not in robot.completed:robot.completed.append(now['id'])
            forget_placement(robot,now['id'])
            return (yield from robot.home_arms())
        if now['id'] in robot.completed:
            raise PolicyFailure('completed_laptop_placement_changed')
    return (yield from finish_laptop(robot,memory,context))


STRATEGY_STAGES = (
    dict(name='place_headset',description='Place and verify headset on the observed stand, preserving released progress.',
         run=resume_place_headset,check=check_headset,protected=lambda c:[c['headset']],reserve=330,
         output='laptop.headset_placed/v1',resume='Retain headset/stand identities and protected release; verify before skipping. Empty ownership required.'),
    dict(name='finish_laptop',description='Close laptop and insert it into the observed rack while protecting headset.',
         run=resume_finish_laptop,check_input=check_headset,input='laptop.headset_placed/v1',output='laptop.controller_finished/v1',
         terminal=True,resume='Reobserve hinge and rack; protect headset. Empty ownership required; never replay the headset stage.'),
)
