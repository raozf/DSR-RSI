"""Physical handover, grounded dustpan and visible residual feedback."""
import numpy as np
from ..control import groups,feature,rotation,PolicyFailure
from .common import align


def sweep_orientation(item, direction):
    """Keep the brush span across travel and the handle above the bristles."""
    direction=np.asarray(direction,float);direction[2]=0
    if np.linalg.norm(direction)<1e-6:raise PolicyFailure('sweep_direction_unresolved')
    direction/=np.linalg.norm(direction)
    return align(feature(item,'brush_axis_local'),feature(item,'brush_up_local'),
                 np.cross([0,0,1],direction))


def contained(item,pan):
    f=pan['features'];d=np.array(f['inward']);p=np.array(item['position'])-f['mouth'];across=np.cross([0,0,1],d)
    length=np.dot(np.array(f['floor'])-f['mouth'],d)*2
    normal=np.asarray(feature(pan,'normal'),float)
    if abs(np.linalg.norm(normal)-1)>.001:raise PolicyFailure('pan_normal_unresolved')
    extent=np.sum(abs(normal@rotation(item['quaternion']).as_matrix())*np.asarray(item['size'])/2)
    bottom=np.dot(np.asarray(item['position'])-f['floor'],normal)-extent
    grounded=-.015<=bottom<=.020
    return bool(.01<=np.dot(p,d)<=length-.005 and
                abs(np.dot(p,across))+max(item['size'][:2])/2<f['width_m']/2-.005 and grounded)


def policy(robot,memory):
    scene=yield from robot.observe();broom=groups(scene,'broom',1)[0];pan=groups(scene,'pan',1)[0];targets=groups(scene,'target')
    if not 3<=len(targets)<=6:raise PolicyFailure('sweep_initial_count')
    pan0=np.array(pan['position']);ids=[t['id'] for t in targets];broom0=np.array(broom['position']);return_q=list(broom['quaternion'])
    arm=yield from robot.pick(broom['id'],'left');yield from robot.observe();broom=robot.object(broom['id'])
    receiver=np.array(broom['position'])+rotation(broom['quaternion']).apply(feature(broom,'handover_local'))
    arm=yield from robot.handover(broom['id'],'right',receiver)
    for attempt in range(10):
        yield from robot.observe();pan=robot.object(pan['id']);f=pan['features']
        if pan['position'][2]-pan0[2]>.007 or np.dot(feature(pan,'normal'),[0,0,1])<np.cos(np.deg2rad(4)):
            raise PolicyFailure('dustpan_lift_or_tilt')
        residual=[robot.object(i) for i in ids if not contained(robot.object(i),pan)]
        if not residual:break
        target=min(residual,key=lambda x:np.linalg.norm(np.array(x['position'])-f['mouth']))
        direction=np.array(f['floor'])-target['position'];direction[2]=0;direction/=max(np.linalg.norm(direction),1e-6)
        tip_start=np.array(target['position'])-direction*(max(target['size'][:2])/2+.035);tip_start[2]=max(.767,float(np.array(f['mouth'])[2]))
        brush=robot.object(broom['id']);q=sweep_orientation(brush,direction);offset=rotation(q).apply(feature(brush,'tip_local'))
        yield from robot.carry(arm,tip_start+[0,0,.07]-offset,q)
        yield from robot.carry(arm,tip_start-offset,q,slow=True)
        end=np.array(f['floor']);end[2]=tip_start[2]
        for t in np.linspace(0,1,7)[1:]:
            point=tip_start*(1-t)+end*t
            yield from robot.carry(arm,point-offset,q,slow=True)
            yield from robot.observe()
            if robot.object(pan['id'])['position'][2]-pan0[2]>.007:raise PolicyFailure('dustpan_lift_during_sweep')
        yield from robot.carry(arm,end+[0,0,.07]-offset,q)
    else:raise PolicyFailure('sweep_residual_budget')
    # Return broom to a visible table point to the right of the pan.
    target=broom0.copy();target[0]=max(target[0],pan['position'][0]+.13)
    yield from robot.place(arm,target,return_q,tolerance=.02)
    yield from robot.observe();pan=robot.object(pan['id'])
    if not all(contained(robot.object(i),pan) for i in ids):raise PolicyFailure('sweep_released_residual')
    return (yield from robot.home_arms())
