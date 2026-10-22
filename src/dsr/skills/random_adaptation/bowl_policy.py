"""One complete random-bowl policy: profile geometry, feedback and recovery."""
import math
import numpy as np

from .bowl_profile import estimate_profile_interval, nesting_offset
from .support.control import PolicyFailure, groups, feature, upright, rotation, quaternion, vector
from .path_gate import scene_path_gate


def memory():
    return dict(clearance_m=.10,lift_m=.10,grasp_tolerance_m=.018,stability_m=.006,
                ee_offset=[0,0,.025],grasp_quaternion=[.5,-.5,.5,.5],max_commands=500,
                max_steps=800,max_translation_step_m=.014,max_rotation_step_rad=.10,
                descent_step_m=.008,lateral_error_m=.004,vertical_error_m=.002,
                support_drift_m=.006,pose_tolerance_m=.006,home_reserve_steps=100,
                calibration_scope='ARX X5 public calibration; native qualification pending')


def profile(item):
    return estimate_profile_interval(**feature(item,'profile'))


def support_check(robot,base_id,support_id,reference,limits):
    base=robot.object(base_id);support=robot.object(support_id)
    upright(base,tolerance=6);upright(support,tolerance=10)
    for item in (base,support):
        if np.linalg.norm(vector(item['position'],3)-reference[item['id']])>limits['support_drift_m']:
            raise PolicyFailure('support_moved',item['id'])
    return support


def descent(robot,arm,base_id,support_id,destination,q,reference,limits):
    held=robot.holding[arm]
    current=held['ee_position']+rotation(held['ee_q']).apply(held['offset'])
    count=max(1,math.ceil((current[2]-destination[2])/limits['descent_step_m']))
    for z in np.linspace(current[2],destination[2],count+1)[1:]:
        yield from robot.observe()
        support_check(robot,base_id,support_id,reference,limits)
        observed=robot.object(held['id'])
        expected=held['ee_position']+rotation(held['ee_q']).apply(held['offset'])
        if np.linalg.norm(vector(observed['position'],3)-expected)>limits['grasp_tolerance_m']:
            raise PolicyFailure('held_bowl_slipping')
        if robot.native_steps>=limits['max_steps']-limits['home_reserve_steps']:
            raise PolicyFailure('home_budget_reserve')
        yield from robot.carry(arm,[*destination[:2],float(z)],q,slow=True)


def recover(robot,reason,limits):
    """Retreat upright while holding; never drop into an uncertain support."""
    robot.stage='recovery:'+str(reason)
    for arm,held in list(robot.holding.items()):
        current=held['ee_position']+rotation(held['ee_q']).apply(held['offset'])
        q=quaternion(rotation(held['ee_q'])*held['rotation'])
        current[2]+=.08
        yield from robot.carry(arm,current,q,slow=True)
    for arm in robot.closed_arms-set(robot.holding):
        actual=robot.ee[arm]
        yield from robot.move(arm,[*actual[:2],actual[2]+.08],actual[3:],slow=True)
    robot.hypotheses.clear()
    status=('controller_recovered_hold' if robot.holding else
            'controller_retreated_closed_gripper' if robot.closed_arms else 'controller_stopped_unheld')
    return dict(status=status,reason=str(reason),native_success=None)


def policy(robot,limits):
    try:
        scene=yield from robot.observe()
        bowls=groups(scene,'bowl',3)
        if len({feature(x,'profile_group') for x in bowls})!=1:raise PolicyFailure('mixed_bowl_profiles')
        base=min(bowls,key=lambda x:(abs(x['position'][0]),x['position'][1],x['id']))
        base_id=support_id=base['id']
        for item in sorted((x for x in bowls if x['id']!=base_id),key=lambda x:x['id']):
            yield from robot.observe()
            support=robot.object(support_id);current=robot.object(item['id'])
            upright(robot.object(base_id),tolerance=6);upright(support,tolerance=10)
            interval=nesting_offset(profile(support),profile(current),
                                    lateral_error_m=limits['lateral_error_m'],
                                    vertical_error_m=limits['vertical_error_m'])
            if interval is None:raise PolicyFailure('no_qualified_nesting_interval')
            # Floor clearance is a lower bound. Native z separation must remain
            # >5mm; the profile solver uses at least 6mm.
            destination=vector(support['position'],3)+rotation(support['quaternion']).apply([0,0,interval['offset_m']])
            q=list(support['quaternion'])
            reference={identity:vector(robot.object(identity)['position'],3).copy() for identity in {base_id,support_id}}
            arm=yield from robot.pick(item['id'],rim=True)
            held=robot.holding[arm]
            start=held['ee_position']+rotation(held['ee_q']).apply(held['offset'])
            obstacles=[]
            for other in groups(robot.scene,'bowl',3):
                if other['id']==item['id']:continue
                centre=np.asarray(other['position']);half=np.asarray(other['size'])/2
                obstacles.append(dict(lower=(centre-half).tolist(),upper=(centre+half).tolist()))
            # The route tracks the held bowl root, so expand observed bowls by
            # the bowl's circumscribed radius plus RGB registration margin.
            envelope=float(np.linalg.norm(np.asarray(item['size'][:2])/2))+.008
            high=max(start[2],destination[2]+limits['clearance_m'],
                     max((np.asarray(x['upper'])[2] for x in obstacles),default=destination[2])
                     +envelope+.015)
            route=scene_path_gate([start,[*start[:2],high],[*destination[:2],high]],
                                  envelope_radius_m=envelope,obstacles=obstacles)
            if not route['allowed']:raise PolicyFailure('observed_path_intersection')
            yield from robot.carry(arm,[*start[:2],high],q,slow=True)
            yield from robot.carry(arm,[*destination[:2],high],q)
            yield from descent(robot,arm,base_id,support_id,destination,q,reference,limits)
            # Check the support and held root one last time before opening.
            yield from robot.observe()
            support_check(robot,base_id,support_id,reference,limits)
            if np.linalg.norm(vector(robot.object(item['id'])['position'],3)-destination)>limits['pose_tolerance_m']:
                raise PolicyFailure('nesting_pose_error')
            yield from robot.release_at(arm,destination,q,limits['pose_tolerance_m'])
            support_id=item['id']
        yield from robot.observe()
        ordered=sorted(groups(robot.scene,'bowl',3),key=lambda x:x['position'][2])
        for item in ordered:upright(item,tolerance=12)
        upright(ordered[0],tolerance=6)
        if any(np.linalg.norm(np.array(a['position'][:2])-b['position'][:2])>.015 or b['position'][2]-a['position'][2]<.005 for a,b in zip(ordered,ordered[1:])):
            raise PolicyFailure('released_stack_proxy_failed')
        return (yield from robot.home_arms())
    except PolicyFailure as error:
        return (yield from recover(robot,error.reason,limits))
