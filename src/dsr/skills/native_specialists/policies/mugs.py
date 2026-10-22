"""Fit cup handle holes to observed support segments, not three guessed hooks."""
import numpy as np
from ..control import groups,feature,rotation,PolicyFailure
from .common import align


def support_slots(supports,mugs):
    slots=[]
    width=max(max(m['size'][:2]) for m in mugs)+.012
    for s in supports:
        length=s['length_m'];n=max(1,int(length/width))
        for offset in np.linspace(-max(0,length-width)/2,max(0,length-width)/2,n):
            slots.append(dict(point=(np.array(s['point'])+np.array(s['axis'])*offset).tolist(),axis=s['axis']))
    if len(slots)<len(mugs):raise PolicyFailure('insufficient_observed_support_capacity')
    return sorted(slots,key=lambda s:s['point'][0])


def policy(robot,memory):
    scene=yield from robot.observe();mugs=groups(scene,'mug',3);rack=groups(scene,'rack',1)[0]
    slots=support_slots(feature(rack,'supports'),mugs)
    for mug,slot in zip(sorted(mugs,key=lambda o:o['position'][0]),slots):
        if feature(mug,'aperture_m')<.012:raise PolicyFailure('handle_aperture_too_small')
        arm=yield from robot.pick(mug['id']);yield from robot.observe();mug_now=robot.object(mug['id'])
        q=align(feature(mug_now,'handle_axis_local'),feature(mug_now,'up_axis_local'),slot['axis'])
        destination=np.array(slot['point'])-rotation(q).apply(feature(mug_now,'handle_local'))
        direction=np.array(slot['axis']);approach=destination+direction*.05+[0,0,.025]
        yield from robot.carry(arm,approach+[0,0,.06],q)
        yield from robot.carry(arm,approach,q,slow=True)
        yield from robot.carry(arm,destination+[0,0,.025],q,slow=True)
        yield from robot.carry(arm,destination,q,slow=True)
        yield from robot.release_at(arm,destination,q,tolerance=.015)
        yield from robot.observe();now=robot.object(mug['id'])
        handle=np.array(now['position'])+rotation(now['quaternion']).apply(feature(now,'handle_local'))
        if handle[2]<.905 or now['position'][2]-mug['position'][2]<.047 or np.linalg.norm(handle-slot['point'])>.025:
            raise PolicyFailure('released_handle_support_unconfirmed')
        robot.protect(mug['id'],.005)
    return (yield from robot.home_arms())
