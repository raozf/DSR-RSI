"""Observed four-well assignment and a separate lid completion stage."""
import numpy as np
from ..control import groups,feature,PolicyFailure
from ..resumption import remember_placement,forget_placement


def fill_wells(robot,memory):
    robot.carton_lid_required=False
    robot.stage='fill_wells:perception'
    for attempt in range(3):
        try:
            scene=yield from robot.observe()
            eggs=groups(scene,'egg',4);holder=groups(scene,'egg_holder',1)[0]
            wells=feature(holder,'wells')
            if len(wells)!=4:raise PolicyFailure('four_visible_wells_required')
            break
        except PolicyFailure:
            if attempt==2:raise
            yield from robot.wait(2)
    available=list(range(len(wells)));assignments={}
    for egg in sorted(eggs,key=lambda e:e['position'][0]):
        yield from robot.observe()
        now=robot.object(egg['id']);current=feature(robot.object(holder['id']),'wells')
        if len(current)!=len(wells):raise PolicyFailure('well_identity_changed')
        index=min(available,key=lambda k:np.linalg.norm(np.array(current[k])[:2]-np.array(now['position'])[:2]))
        available.remove(index);assignments[egg['id']]=index
        target=np.array(current[index])+[0,0,now['size'][2]/2]
        yield from robot.transfer(egg['id'],target,tolerance=.012)
        robot.protect(egg['id'],.008)
    return dict(egg_ids=[e['id'] for e in eggs],assignments=assignments,holder=holder['id'])


def check_wells(robot,context):
    wells=feature(robot.object(context['holder']),'wells')
    for identity,index in context['assignments'].items():
        item=robot.object(identity);delta=np.array(item['position'])-wells[index]
        if np.linalg.norm(delta[:2])>.016 or not 0<=delta[2]-item['size'][2]/2<.022:
            raise PolicyFailure('egg_not_seated_in_assigned_well',identity)


def finish_lid(robot,memory,context):
    robot.carton_lid_required=True
    robot.stage='finish_lid:perception'
    for attempt in range(3):
        try:
            scene=yield from robot.observe()
            lid=groups(scene,'lid',1)[0]
            for key in ('hinge','hinge_axis','free_edge','closing_angle_rad'):feature(lid,key)
            context['lid']=lid['id']
            break
        except PolicyFailure:
            if attempt==2:raise
            yield from robot.wait(2)
    # A closed opaque lid can hide eggs. Keep their verified volume as a path
    # exclusion region; absence is never reported as a new occupancy observation.
    robot.expected_occlusion={identity:context['lid'] for identity in context['egg_ids']}
    lid=robot.object(context['lid'])
    if abs(feature(lid,'closing_angle_rad'))>.12:
        yield from robot.close_lid(lid['id'])
    yield from robot.observe()
    if abs(feature(robot.object(lid['id']),'closing_angle_rad'))>.12:
        raise PolicyFailure('egg_lid_not_visibly_closed')
    return (yield from robot.home_arms())


def policy(robot,memory):
    context=yield from fill_wells(robot,memory)
    yield from robot.observe();check_wells(robot,context)
    return (yield from finish_lid(robot,memory,context))


def resume_fill_wells(robot,memory):
    """Retain assignments; adopt only freshly verified, released placements."""
    robot.carton_lid_required=False
    robot.stage='fill_wells:perception'
    scene=yield from robot.observe()
    eggs=groups(scene,'egg',4);holder=groups(scene,'egg_holder',1)[0]
    wells=feature(holder,'wells')
    if len(wells)!=4:raise PolicyFailure('four_visible_wells_required')
    key='eggs.fill_wells'
    if key not in robot.progress:
        available=list(range(4));assignments={}
        for egg in sorted(eggs,key=lambda e:e['position'][0]):
            index=min(available,key=lambda k:np.linalg.norm(np.array(wells[k])[:2]-egg['position'][:2]))
            available.remove(index);assignments[egg['id']]=index
        robot.progress[key]=dict(egg_ids=[e['id'] for e in eggs],assignments=assignments,holder=holder['id'])
    context=robot.progress[key]
    if set(context['egg_ids'])!={e['id'] for e in eggs} or context['holder']!=holder['id']:
        raise PolicyFailure('well_identity_changed')
    for identity,index in context['assignments'].items():
        yield from robot.observe()
        current=feature(robot.object(context['holder']),'wells')
        if len(current)!=4:raise PolicyFailure('well_identity_changed')
        egg=robot.object(identity);target=np.array(current[index])+[0,0,egg['size'][2]/2]
        if identity not in robot.protected:
            if np.linalg.norm(np.array(egg['position'])-target)<.012:
                yield from robot.stable(identity,target,.012)
            else:
                remember_placement(robot,identity,target,None,.012,context['holder'],['wells',index])
                yield from robot.transfer(identity,target,tolerance=.012)
            robot.protect(identity,.008)
            forget_placement(robot,identity)
        check_wells(robot,dict(holder=context['holder'],assignments={identity:index}))
    return context


def check_retained_wells(robot,context):
    # A verified receipt can survive explicit closed-lid occlusion. It does
    # not become a fresh occupancy measurement or native success evidence.
    hidden=set(context['egg_ids']) & robot.visibility_unknown
    for identity in hidden:
        if identity not in robot.protected or identity not in robot.expected_occlusion:
            raise PolicyFailure('unverified_occluded_egg',identity)
    visible={identity:index for identity,index in context['assignments'].items() if identity not in hidden}
    if visible:check_wells(robot,dict(holder=context['holder'],assignments=visible))


STRATEGY_STAGES = (
    dict(name='fill_wells',description='Seat eggs using retained well assignments; verify stable released subgoal.',
         run=resume_fill_wells,check=check_wells,protected=lambda c:c['egg_ids'],reserve=230,
         output='eggs.seated/v1',resume='Retain assignments and protected eggs; reobserve and skip verified placements. Empty ownership required.'),
    dict(name='finish_lid',description='Close the observed lid and return home, retaining verified egg placements.',
         run=finish_lid,check_input=check_retained_wells,input='eggs.seated/v1',output='eggs.controller_finished/v1',
         terminal=True,resume='Reobserve hinge angle before contacts; preserve seated-egg receipt and exclusions. Withdraw fist contact before retry.'),
)
