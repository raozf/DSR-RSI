"""Descriptor-compatible worker. No changes to the running frozen evaluator."""
from collections import deque
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .control import Robot, PolicyFailure, rotation, vector


class Episode:
    def __init__(self, name, memory, perception, log, startup=None, tick_factory=None):
        self.name, self.memory, self.perception = name, memory, perception
        self.startup = startup
        self.log = Path(log)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.channel = uuid.uuid4().hex
        self.sequence = self.steps = 0
        self.observation = self.robot = self.generator = None
        self.queue = deque()
        self.pending = None
        self.command_pending = None
        self.settle = 0
        self.previous = {}
        self.result = self.failure = None
        self.tick = tick_factory(self) if tick_factory is not None else None
        self.last_action_sequence = -1

    def event(self, **data):
        with self.log.open("a") as stream:
            stream.write(json.dumps(dict(sequence=self.sequence, steps=self.steps, **data), allow_nan=False) + "\n")

    def observe(self, observation):
        # Strip all evaluator/debug/ground-truth fields at the worker boundary.
        self.observation = {k:deepcopy(observation[k]) for k in ("vision","state","instruction")}
        self.sequence += 1

    def hold(self):
        return {arm+suffix:vector(self.observation["state"][arm+suffix], n).tolist()
                for arm in ("left","right") for suffix,n in (("_ee_pose",7),("_ee_joint_state",1))}

    def target_reached(self):
        arm, goal, q = self.pending
        actual = vector(self.observation["state"][arm+"_ee_pose"],7)
        return np.linalg.norm(actual[:3]-goal)<.012 and (rotation(q).inv()*rotation(actual[3:])).magnitude()<.10

    def check_observation(self, episode):
        if episode is not self or self.observation is None or self.sequence <= self.last_action_sequence:
            raise PolicyFailure('fresh_observation_required')
        self.check_action(self.hold())
        self.last_action_sequence = self.sequence
        return self

    def check_action(self, action):
        expected = {a+s for a in ('left','right') for s in ('_ee_pose','_ee_joint_state')}
        if set(action) != expected:
            raise PolicyFailure('action_fields')
        for arm in ('left','right'):
            pose = vector(action[arm+'_ee_pose'], 7)
            rotation(pose[3:])
            grip = vector(action[arm+'_ee_joint_state'], 1)[0]
            if not 0 <= grip <= 1:
                raise PolicyFailure('gripper_contract')
        return action

    def action(self):
        if self.tick is not None:
            from PhysicalRSI_core.contracts import Context
            return self.tick(self, Context(self.channel))
        self.check_observation(self)
        return self.check_action(self._advance())

    def _advance(self):
        self.steps += 1
        if self.robot is not None:
            self.robot.native_steps = self.steps
            # Proprioception remains usable when the RGB observation fails.
            # Recovery must start from the measured EEF, not the last goal.
            for arm in ('left','right'):
                actual=list(self.observation['state'][arm+'_ee_pose'])
                self.robot.ee[arm]=actual
                if arm in self.robot.holding:
                    self.robot.holding[arm].update(ee_position=np.asarray(actual[:3]),ee_q=actual[3:])
        if self.failure or self.result is not None:
            return self.hold()
        try:
            if self.steps > self.memory["max_steps"]:
                raise PolicyFailure("physical_step_budget")
            if self.queue:
                return self.queue.popleft()
            if self.pending:
                if not self.target_reached():
                    self.settle += 1
                    if self.settle > 12:
                        raise PolicyFailure("eef_target_unreachable")
                    arm, p, q = self.pending
                    action=self.hold(); action[arm+"_ee_pose"]=[*p,*q]
                    return action
                self.pending=None; self.settle=0
            if self.command_pending is not None:
                command = self.command_pending
                self.command_pending = None
            elif self.generator is None:
                home={arm:self.hold()[arm+"_ee_pose"] for arm in ("left","right")}
                self.robot=Robot(self.memory,self.observation["instruction"],home)
                if self.startup is not None:
                    from PhysicalRSI_core.contracts import Context
                    self.generator=self.startup(self.robot,Context(self.channel))
                else:
                    raise PolicyFailure("missing_policy_startup")
                command=next(self.generator)
            else:
                command=self.generator.send(None)
            for _ in range(24):
                self.event(command=command)
                op=command["op"]
                if op=="observe":
                    if self.robot.sequence == self.sequence:
                        self.command_pending = command
                        return self.hold()
                    tracked=deepcopy(self.previous)
                    for arm,held in {**self.robot.hypotheses, **self.robot.holding}.items():
                        pose=self.observation["state"][arm+"_ee_pose"]
                        if held["id"] in tracked:
                            tracked[held["id"]]["predicted_position"]=(np.asarray(pose[:3])+rotation(pose[3:]).apply(held["offset"])).tolist()
                            tracked[held["id"]]["quaternion"] = (rotation(pose[3:])*held['rotation']).as_quat()[[3,0,1,2]].tolist()
                    try:
                        objects=self.perception.detect(self.observation,tracked)
                    except PolicyFailure as error:
                        # Give this same policy its bounded recovery path.
                        command=self.generator.throw(error)
                        continue
                    self.previous.update(deepcopy(objects))
                    scene=dict(source="rgb",frame="world",unit="m",channel=self.channel,
                               sequence=self.sequence,objects=objects,robot_state=deepcopy(self.observation["state"]))
                    command=self.generator.send(scene)
                    continue
                action=self.hold()
                if op=="wait":
                    count=command["steps"]
                    if type(count) is not int or not 1<=count<=25:
                        raise PolicyFailure("wait_budget")
                    self.queue.extend(deepcopy(action) for _ in range(count))
                elif op=="grip":
                    arm=command["arm"]; value=command["value"]
                    if arm not in ("left","right") or not 0<=value<=1:
                        raise PolicyFailure("gripper_contract")
                    action[arm+"_ee_joint_state"]=[float(value)]
                    self.queue.extend(deepcopy(action) for _ in range(5))
                elif op=="move":
                    arm=command["arm"]
                    if arm not in ("left","right"):
                        raise PolicyFailure("arm_contract")
                    p=vector(command["position"],3); q=vector(command["quaternion"],4)
                    if not (-.60<p[0]<.60 and -.60<p[1]<.40 and .78<p[2]<1.5):
                        raise PolicyFailure("workspace_limit")
                    start=vector(action[arm+"_ee_pose"],7)
                    a,b=rotation(start[3:]),rotation(q)
                    distance=np.linalg.norm(p-start[:3]); angle=(a.inv()*b).magnitude()
                    speed=self.memory["max_translation_step_m"]*(.7 if command.get("slow") else 1.)
                    count=max(2,math.ceil(distance/speed),math.ceil(angle/self.memory["max_rotation_step_rad"]))
                    if count>180 or self.steps+count>self.memory["max_steps"]:
                        raise PolicyFailure("motion_budget")
                    curve=Slerp([0,1],Rotation.from_quat([a.as_quat(),b.as_quat()]))
                    for t in np.linspace(0,1,count+1)[1:]:
                        row=deepcopy(action)
                        row[arm+"_ee_pose"]=[*(start[:3]*(1-t)+p*t),*curve(t).as_quat()[[3,0,1,2]]]
                        # Straight-line clearance check against the other EEF.
                        other="left" if arm=="right" else "right"
                        if np.linalg.norm(np.asarray(row[arm+"_ee_pose"][:3])-action[other+"_ee_pose"][:3])<.065:
                            raise PolicyFailure("cross_arm_clearance")
                        self.queue.append(row)
                    self.pending=(arm,p.tolist(),q.tolist())
                else:
                    raise PolicyFailure("unknown_primitive",str(op))
                return self.queue.popleft()
            raise PolicyFailure("no_physical_progress")
        except StopIteration as completed:
            self.result=completed.value
            self.event(controller_result=self.result,native_success=None)
        except (PolicyFailure,ValueError,KeyError) as error:
            self.queue.clear()
            self.failure=dict(reason=getattr(error,"reason","observation_or_configuration"),detail=str(error),
                              stage=self.robot.stage if self.robot else "startup")
            self.event(controller_failure=self.failure,native_success=None)
        return self.hold()
