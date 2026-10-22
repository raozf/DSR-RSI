"""Task functional-point actions on fresh RGB, with observed attachment."""
from copy import deepcopy
import math
import numpy as np
from .control import Robot, PolicyFailure, feature, vector, rotation, quaternion
from .path_gate import scene_path_gate


class SemanticRobot(Robot):
    def __init__(self, *args):
        super().__init__(*args)
        self.protected = {}
        self.expected_occlusion = {}
        self.visibility_unknown = set()
        self.progress = {}
        self.recovery_history = []
        self.cleared_views = set()

    def recover(self, error):
        """Keep task/ownership memory while obtaining another usable view."""
        record=dict(reason=error.reason,detail=str(error),stage=self.stage,
                    holding={a:h['id'] for a,h in self.holding.items()},
                    completed=list(self.completed),attempt=len(self.recovery_history)+1)
        self.recovery_history.append(record)
        yield from self.command('recovery',**record)
        if self.native_steps>=self.memory['max_steps']:
            raise PolicyFailure('physical_step_budget')
        # Only unoccupied, open arms may vacate a view. An uncertain grasp is
        # retained until observed; it is never discarded to make replanning easy.
        for arm in ('left','right'):
            if (arm in self.cleared_views or arm in self.holding or arm in self.hypotheses
                    or arm in self.closed_arms):continue
            self.cleared_views.add(arm)
            measured=np.asarray(self.ee[arm][:3]);home=np.asarray(self.home[arm][:3])
            height=min(1.38,max(measured[2]+.06,home[2],1.10))
            try:
                if measured[2]<height-.005:
                    yield from self._qualified_move(arm,[*measured[:2],height],self.ee[arm][3:],True)
                if np.linalg.norm(measured[:2]-home[:2])>.015:
                    yield from self._qualified_move(arm,[*home[:2],height],self.ee[arm][3:],True)
            except PolicyFailure as rejected:
                record['view_motion_rejected']=rejected.reason
            break
        yield from self.wait(2)

    def protect(self, identity, tolerance=.008):
        self.protected[identity] = (deepcopy(self.object(identity)), tolerance)

    def observe(self, required=None):
        scene = yield from super().observe(required)
        for identity, (old, tolerance) in self.protected.items():
            if identity not in scene['objects'] and identity in self.expected_occlusion:
                occluder=self.object(self.expected_occlusion[identity])
                box=occluder.get('box');before=old.get('box')
                # Explicit, fresh closed-lid evidence plus image containment.
                # Keep the old exclusion volume, and label occupancy unknown.
                if box is not None and before is not None and abs(feature(occluder,'closing_angle_rad'))<=.12:
                    pixel=(np.asarray(before[:2])+before[2:])/2
                    if np.all(pixel>=box[:2]) and np.all(pixel<=box[2:]) and occluder['confidence']>=.6:
                        self.visibility_unknown.add(identity)
                        continue
            now = self.object(identity)  # Missing/occluded is not completion.
            self.visibility_unknown.discard(identity)
            if np.linalg.norm(np.asarray(now['position'])-old['position']) > tolerance:
                raise PolicyFailure('protected_object_moved', identity)
            if self.orientation_error(now,old['quaternion']) > np.deg2rad(15):
                raise PolicyFailure('protected_object_rotated', identity)
        return scene

    def carry(self, arm, object_position, object_q=None, slow=False):
        """Verify each segment against measured EEF and fresh object geometry."""
        yield from super().carry(arm,object_position,object_q,slow)
        yield from self.observe()
        held=self.holding[arm];item=self.object(held['id']);actual=self.ee[arm]
        expected=np.asarray(actual[:3])+rotation(actual[3:]).apply(held['offset'])
        expected_q=quaternion(rotation(actual[3:])*held['rotation'])
        if np.linalg.norm(np.asarray(item['position'])-expected)>self.memory['grasp_tolerance_m']:
            raise PolicyFailure('attachment_translation_unconfirmed',held['id'])
        if self.orientation_error(item,expected_q)>np.deg2rad(15):
            raise PolicyFailure('attachment_rotation_unconfirmed',held['id'])

    def grasp_orientation(self, item):
        # Called by the parent after its single acquisition observation.
        # Contact and orientation must refer to that same measured object.
        grasp = feature(item, 'grasp')
        if not 0.005 <= grasp['width_m'] <= .072:
            raise PolicyFailure('unqualified_jaw_width', item['id'])
        rotation(grasp['quaternion'])
        return list(grasp['quaternion'])

    def close_lid(self, identity):
        """Recompute each hinge contact from observed progress, with one budget."""
        from .analytical_skills import hinge_increment
        self.stage='close_lid:'+identity
        yield from self.observe()
        lid=self.object(identity);arm=self.choose_arm(lid);q=self.memory['grasp_quaternion']
        edge=np.asarray(feature(lid,'free_edge'))
        yield from self.grip(arm,False)
        yield from self.move(arm,self.contact_pose(edge+[0,0,.06],q),q)
        previous=None;stalled=0
        for _ in range(48):
            yield from self.observe();lid=self.object(identity)
            angle=float(feature(lid,'closing_angle_rad'))
            if abs(angle)<=.10:break
            if previous is not None:
                stalled=stalled+1 if abs(angle)>=previous-.004 else 0
                if stalled>=3:raise PolicyFailure('hinge_no_observed_progress')
            previous=abs(angle)
            proposal=hinge_increment(hinge_m=feature(lid,'hinge'),axis=feature(lid,'hinge_axis'),
                edge_m=feature(lid,'free_edge'),remaining_angle_rad=angle,max_angle_rad=.15,max_contact_step_m=.018)
            yield from self.move(arm,self.contact_pose(proposal['contact_m'],q),q,slow=True)
        else:raise PolicyFailure('hinge_contact_budget')
        current=np.asarray(self.ee[arm][:3]);yield from self.move(arm,current+[0,0,.08],q)
        yield from self.grip(arm,True)
        for _ in range(3):
            yield from self.wait(2);yield from self.observe()
            if abs(float(feature(self.object(identity),'closing_angle_rad')))>.12:
                raise PolicyFailure('lid_reopened_after_withdrawal')

    def press_latch(self, identity):
        """Short observed lever contacts; release evidence belongs to caller."""
        self.stage='press_latch:'+identity
        yield from self.observe()
        lever=self.object(identity);start=vector(feature(lever,'contact'),3)
        bottom=vector(feature(lever,'bottom'),3);travel=np.linalg.norm(bottom-start)
        if travel<=.006:return
        if travel>.10:raise PolicyFailure('lever_travel_unresolved')
        direction=(bottom-start)/travel;arm=self.choose_arm(lever);q=self.memory['grasp_quaternion']
        yield from self.grip(arm,False)
        yield from self.move(arm,self.contact_pose(start-direction*.04,q),q)
        yield from self.move(arm,self.contact_pose(start,q),q,slow=True)
        previous=None;stalled=0
        for _ in range(20):
            yield from self.observe();lever=self.object(identity)
            contact=vector(feature(lever,'contact'),3);current_bottom=vector(feature(lever,'bottom'),3)
            if np.linalg.norm(current_bottom-bottom)>.006:raise PolicyFailure('lever_track_moved')
            delta=current_bottom-contact;remaining=float(np.dot(delta,direction))
            if np.linalg.norm(delta-direction*remaining)>.008:raise PolicyFailure('lever_off_track')
            if np.linalg.norm(delta)<=.006:break
            if remaining<0 or remaining>travel+.006:raise PolicyFailure('lever_contact_outside_track')
            if previous is not None:
                stalled=stalled+1 if remaining>=previous-.002 else 0
                if stalled>=3:raise PolicyFailure('lever_no_observed_progress')
            previous=remaining
            target=contact+direction*min(.006,remaining)
            yield from self.move(arm,self.contact_pose(target,q),q,slow=True)
        else:raise PolicyFailure('lever_contact_budget')
        yield from self.move(arm,self.contact_pose(start-direction*.06,q),q)
        yield from self.grip(arm,True)

    def _qualified_move(self, arm, position, q, slow=False):
        if self.stage != 'return_home' and self.native_steps >= self.memory['max_steps']-self.memory['home_reserve_steps']:
            raise PolicyFailure('terminal_budget_reserve')
        # Protect completed work against the moving held object's envelope.
        obstacles=[]
        for identity,(obj,_) in self.protected.items():
            half=abs(rotation(obj['quaternion']).as_matrix())@np.asarray(obj['size'])/2
            centre=np.asarray(obj['position'])
            obstacles.append(dict(lower=(centre-half).tolist(),upper=(centre+half).tolist()))
        radius=.025
        if arm in self.holding:
            held=self.holding[arm]
            obj=self.object(held['id'])
            radius=max(radius,float(np.linalg.norm(obj['size'])/2+np.linalg.norm(held['offset'])))
        gate=scene_path_gate([self.ee[arm][:3],position],envelope_radius_m=radius,obstacles=obstacles)
        if not gate['allowed']:
            raise PolicyFailure('protected_path_intersection')
        return (yield from super().move(arm,position,q,slow))

    def move(self, arm, position, q, slow=False):
        try:
            return (yield from self._qualified_move(arm,position,q,slow))
        except PolicyFailure as error:
            if (not self.memory.get('recover_motion',False) or error.reason not in
                    {'eef_target_unreachable','native_joint_path_rejected','native_joint_solution_missing',
                     'protected_path_intersection','cross_arm_clearance'}):raise
            record=dict(reason=error.reason,stage=self.stage,arm=arm,
                        strategy='lift_translate_rotate_descend',target=list(position))
            self.recovery_history.append(record)
            yield from self.command('recovery',**record)
            # The failed goal was never acknowledged. Replan from the current
            # measured EEF; every waypoint still uses both path and native IK gates.
            current=list(self.ee[arm]);target=vector(position,3)
            for extra in (0., .04):
                current=list(self.ee[arm])
                height=max(current[2],target[2])+self.memory['clearance_m']+extra
                if height>=1.45:break
                try:
                    yield from self._qualified_move(arm,[*current[:2],height],current[3:],True)
                    yield from self._qualified_move(arm,[*target[:2],height],current[3:],True)
                    yield from self._qualified_move(arm,[*target[:2],height],q,True)
                    result=yield from self._qualified_move(arm,target,q,True)
                except PolicyFailure as rejected:
                    yield from self.command('recovery',reason=rejected.reason,arm=arm,
                        strategy='alternative_clearance',extra_clearance_m=extra,outcome='rejected')
                else:
                    yield from self.command('recovery',reason=error.reason,arm=arm,
                        strategy='lift_translate_rotate_descend',outcome='finished')
                    return result
            raise PolicyFailure('motion_recovery_exhausted',str(error)) from error

    def place_point(self, identity, local_point, target, q, tolerance=.012):
        destination=vector(target,3)-rotation(q).apply(vector(local_point,3))
        arm=yield from self.pick(identity)
        yield from self.place(arm,destination,q,tolerance)
        return arm

    def handover(self, identity, receiver, receiver_point):
        """A physical two-gripper transfer; never a controller handoff."""
        donor=next((a for a,h in self.holding.items() if h['id']==identity),None)
        if donor is None or donor==receiver or receiver in self.holding:
            raise PolicyFailure('handover_ownership')
        item=self.object(identity)
        q=feature(item,'grasp')['quaternion']
        contact=self.contact_pose(receiver_point,q)
        if np.linalg.norm(contact-self.ee[donor][:3])<.085:
            raise PolicyFailure('handover_gripper_overlap')
        yield from self.grip(receiver,True)
        yield from self.move(receiver,contact+[0,0,.07],q)
        yield from self.move(receiver,contact,q,slow=True)
        yield from self.grip(receiver,False)
        yield from self.observe()
        item=self.object(identity);actual=self.ee[receiver]
        attachment=dict(id=identity,offset=rotation(actual[3:]).inv().apply(np.asarray(item['position'])-actual[:3]),
            rotation=rotation(actual[3:]).inv()*rotation(item['quaternion']),
            ee_position=np.array(actual[:3]),ee_q=actual[3:])
        yield from self.grip(donor,True)
        del self.holding[donor]
        self.holding[receiver]=attachment
        yield from self.move(donor,np.asarray(self.ee[donor][:3])+[0,0,.08],self.ee[donor][3:])
        start=np.array(item['position'])
        for lift in (.015,.030):
            yield from self.carry(receiver,start+[0,0,lift],slow=True)
            now=self.object(identity)
            if np.linalg.norm(np.asarray(now['position'])-(start+[0,0,lift]))>.010:
                raise PolicyFailure('handover_attachment_unconfirmed',identity)
        yield from self.stable(identity,start+[0,0,.030],.012)
        return receiver

    def fold_point(self, identity, source_name, target_name):
        """Cloth landmark feedback; never propagate a rigid body transform."""
        yield from self.observe()
        cloth=self.object(identity);landmarks=feature(cloth,'landmarks')
        source=vector(landmarks[source_name],3);target=vector(landmarks[target_name],3)
        arm=self.choose_arm(dict(position=source));q=self.memory['grasp_quaternion']
        offset=self.contact_pose([0,0,0],q)
        self.stage='fold:'+source_name
        yield from self.grip(arm,True)
        yield from self.move(arm,source+[0,0,.06]+offset,q)
        yield from self.move(arm,source+offset,q,True)
        yield from self.grip(arm,False)
        for fraction in (.25,.5,.75,1.):
            point=source*(1-fraction)+target*fraction+[0,0,.08*math.sin(math.pi*fraction)]
            yield from self.move(arm,point+offset,q,True)
            yield from self.observe()
            measured=vector(feature(self.object(identity),'landmarks')[source_name],3)
            if np.linalg.norm(measured-point)>.05:
                raise PolicyFailure('cloth_landmark_not_following',source_name)
        yield from self.grip(arm,True)
        yield from self.move(arm,target+[0,0,.07]+offset,q)
        yield from self.observe()
        points=feature(self.object(identity),'landmarks')
        if np.linalg.norm(np.asarray(points[source_name])[:2]-np.asarray(points[target_name])[:2])>.06:
            raise PolicyFailure('fold_not_observed',source_name)
