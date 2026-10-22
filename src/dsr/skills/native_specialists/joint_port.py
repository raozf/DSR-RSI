"""Original public X5 IK and sampled robot/table/cross-arm collision gate.

The emitted action is the qualified joint solution, avoiding an independent
simulator IK branch. This does not claim complete clutter collision coverage.
"""
import numpy as np
from arrange_rgb.skill_choices import IK
from arrange_rgb.collision_gate import ContinuousGate
from .control import PolicyFailure


class JointPort:
    def __init__(self):self.ik=IK()

    def convert(self,action,state):
        result={a+s:list(state[a+s]) for a in ('left','right') for s in ('_arm_joint_state','_ee_joint_state')}
        for arm in ('left','right'):
            moving=not np.allclose(action[arm+'_ee_pose'],state[arm+'_ee_pose'],atol=1e-6,rtol=0)
            gripping=abs(action[arm+'_ee_joint_state'][0]-state[arm+'_ee_joint_state'][0])>1e-6
            if not moving and not gripping:continue
            gate=ContinuousGate(state,arm)
            outcome=self.ik.path(arm,[('specialist',action)],state,gate)
            if not outcome['passed']:raise PolicyFailure('native_joint_path_rejected',outcome.get('reason','unknown'))
            records=outcome['records']
            if not records:raise PolicyFailure('native_joint_solution_missing')
            result[arm+'_arm_joint_state']=records[-1]['q']
            result[arm+'_ee_joint_state']=list(action[arm+'_ee_joint_state'])
        return result
