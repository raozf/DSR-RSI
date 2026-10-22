"""Keep the observed bearing layer; remeasure support before each descent."""
import numpy as np
from ..control import groups,feature,PolicyFailure,upright


def policy(robot,memory):
    blocks=groups((yield from robot.observe()),'block',3)
    ordered=sorted(blocks,key=lambda o:(-np.prod(o['size'][:2]),o['id']))
    base=ordered[0]['id'];support=base;placed=[base]
    for item in ordered[1:]:
        yield from robot.observe();bearing=robot.object(support);upright(bearing,7)
        initial={i:np.array(robot.object(i)['position']) for i in placed}
        arm=yield from robot.pick(item['id'])
        yield from robot.observe();bearing=robot.object(support);current=robot.object(item['id'])
        normal=np.array(feature(bearing,'support_normal'))
        if normal[2]<np.cos(np.deg2rad(7)):raise PolicyFailure('tilted_bearing_face')
        target=np.array(feature(bearing,'support_point'))+[0,0,current['size'][2]/2]
        if np.any(np.asarray(current['size'][:2])>np.asarray(bearing['size'][:2])+.002):raise PolicyFailure('insufficient_support_footprint')
        for identity,point in initial.items():
            if np.linalg.norm(np.asarray(robot.object(identity)['position'])-point)>.005:raise PolicyFailure('bearing_layer_moved')
        yield from robot.place(arm,target,bearing['quaternion'],tolerance=.008)
        yield from robot.observe();now=robot.object(item['id']);upright(now,7)
        if np.linalg.norm(np.array(now['position'][:2])-robot.object(support)['position'][:2])>.010:raise PolicyFailure('stack_xy_margin')
        for identity,point in initial.items():
            if np.linalg.norm(np.asarray(robot.object(identity)['position'])-point)>.005:raise PolicyFailure('released_bearing_layer_moved')
        placed.append(item['id']);support=item['id']
    for identity in placed:yield from robot.stable(identity)
    return (yield from robot.home_arms())
