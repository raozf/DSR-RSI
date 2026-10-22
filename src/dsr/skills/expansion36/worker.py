"""Descriptor-compatible worker. No changes to the running frozen evaluator."""
from collections import deque
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .catalog import POLICIES
from .control import Robot, PolicyFailure, rotation, vector
from .perception import RGBScene
try:
    from .trace import append_event
except ModuleNotFoundError as error:
    if error.name != __package__ + '.trace':
        raise
    from dsr.skills.native_specialists.trace import append_event


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
        self.command_error = None
        self.perception_attempts = 0
        self.finished = False
        self.trace_error = None

    def event(self, **data):
        error = append_event(self.log, dict(channel=self.channel, sequence=self.sequence,
                                            steps=self.steps, **data))
        if error is not None:
            self.trace_error = error

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
        try:
            if self.tick is not None:
                from PhysicalRSI_core.contracts import Context
                return self.check_action(self.tick(self, Context(self.channel)))
            self.check_observation(self)
            return self.check_action(self._advance())
        except PolicyFailure:
            raise
        except Exception as error:
            if not self.failure:
                self.fail(PolicyFailure('controller_exception', type(error).__name__ + ': ' + str(error)))
            return self.hold()

    def fail(self, error):
        self.queue.clear()
        self.pending = self.command_pending = self.command_error = None
        self.failure = dict(reason=getattr(error, 'reason', 'observation_or_configuration'),
                            detail=str(error), stage=self.robot.stage if self.robot else 'startup')
        self.event(controller_failure=self.failure, native_success=None)

    def _advance(self):
        self.steps += 1
        if self.robot is not None:
            self.robot.native_steps=self.steps
            self.robot.ee={a:list(self.observation['state'][a+'_ee_pose']) for a in ('left','right')}
        if self.failure or self.finished:
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
            if self.command_error is not None:
                error,self.command_error=self.command_error,None
                command=self.generator.throw(error)
            elif self.command_pending is not None:
                command = self.command_pending
                self.command_pending = None
            elif self.generator is None:
                home={arm:self.hold()[arm+"_ee_pose"] for arm in ("left","right")}
                self.robot=Robot(self.memory,self.observation["instruction"],home)
                self.robot.native_steps=self.steps
                if self.startup is not None:
                    from PhysicalRSI_core.contracts import Context
                    self.generator=self.startup(self.robot,Context(self.channel))
                else:
                    module=importlib.import_module(__package__+".policies."+POLICIES[self.name]["module"])
                    self.generator=module.policy(self.robot,deepcopy(self.memory))
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
                    for arm,held in self.robot.holding.items():
                        pose=self.observation["state"][arm+"_ee_pose"]
                        if held["id"] in tracked:
                            tracked[held["id"]]["predicted_position"]=(np.asarray(pose[:3])+rotation(pose[3:]).apply(held["offset"])).tolist()
                    try:objects=self.perception.detect(self.observation,tracked)
                    except PolicyFailure as error:
                        self.event(perception_failure=dict(reason=error.reason,detail=str(error)),
                            perception_diagnostic=getattr(self.perception,'last_depth_calibration',None))
                        if self.perception_attempts<3:
                            self.perception_attempts+=1
                            self.command_pending=command
                            self.queue.extend(deepcopy(self.hold()) for _ in range(2))
                            self.event(recovery=dict(reason=error.reason,attempt=self.perception_attempts,
                                strategy='fresh_rgb_same_suspended_observe'),native_success=None)
                            return self.hold()
                        self.perception_attempts=0
                        command=self.generator.throw(error)
                        continue
                    if self.perception_attempts:
                        self.event(recovery_finished=dict(resume='same_suspended_observe'),native_success=None)
                    self.perception_attempts=0
                    self.previous.update(deepcopy(objects))
                    scene=dict(source="rgb",frame="world",unit="m",channel=self.channel,
                               sequence=self.sequence,objects=objects)
                    command=self.generator.send(scene)
                    continue
                action=self.hold()
                if op=='recovery':
                    self.event(recovery=command,native_success=None)
                    return action
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
            self.finished=True
            self.event(controller_result=self.result,native_success=None)
        except (PolicyFailure,ValueError,KeyError) as error:
            self.queue.clear()
            self.pending=None;self.settle=0
            if (isinstance(error,PolicyFailure) and error.reason in
                    {'eef_target_unreachable','cross_arm_clearance'} and self.generator is not None
                    and inspect.getgeneratorstate(self.generator)!=inspect.GEN_CLOSED):
                self.command_error=error
                self.event(action_rejected=dict(reason=error.reason,detail=str(error)),native_success=None)
                return self.hold()
            self.failure=dict(reason=getattr(error,"reason","observation_or_configuration"),detail=str(error),
                              stage=self.robot.stage if self.robot else "startup")
            self.event(controller_failure=self.failure,native_success=None)
        except Exception as error:
            self.fail(PolicyFailure('controller_exception', type(error).__name__ + ': ' + str(error)))
        return self.hold()


class Model:
    def __init__(self, model_cfg):
        root=Path(__file__).parent
        manifest=json.loads((root/"package.json").read_text())
        expected=model_cfg["expansion_package_sha256"]
        if hashlib.sha256((root/"package.json").read_bytes()).hexdigest()!=expected:
            raise ValueError("Expansion package manifest changed")
        for relative,digest in manifest["files"].items():
            p=(root/relative).resolve()
            if not p.is_relative_to(root.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
                raise ValueError("Expansion source changed: "+relative)
        self.name=model_cfg["expansion_policy"]
        self.memory=json.loads((root/(self.name+".memory.json")).read_text())
        self.perception=RGBScene(self.name)
        self.output=Path(os.environ["RDJ_OUTPUT_DIR"])
        from .program import compile_start, compile_tick
        spec=json.loads((root/(self.name+".program.json")).read_text())
        self.startup=compile_start(self.name,self.memory,spec,self.output/"memory",
                                  model_cfg["expansion_controller_revision"])
        self.tick_factory=lambda ep: compile_tick(self.name,spec,self.output/'memory'/ep.channel,
                                  model_cfg['expansion_controller_revision'],ep)
        self.reset()

    def reset(self):
        self.episodes={};self.indices=[];self.consumed=set()

    def update_obs_batch(self, observations):
        self.indices=[];self.consumed=set()
        for observation in observations:
            i=int(observation["env_idx"])
            if i in self.indices:
                raise ValueError("Duplicate environment index")
            if i not in self.episodes:
                self.episodes[i]=Episode(self.name,self.memory,self.perception,
                                        self.output/(f"{i}-"+uuid.uuid4().hex+".jsonl"),self.startup,self.tick_factory)
            self.episodes[i].observe(observation);self.indices.append(i)

    def update_obs(self, observation):
        self.update_obs_batch([dict(observation,env_idx=0)])

    def get_action_batch(self, indices=None):
        indices=self.indices if indices is None else list(indices)
        if len(set(indices))!=len(indices) or any(i not in self.indices or i in self.consumed for i in indices):
            raise ValueError("Requested environment has no fresh observation")
        self.consumed.update(indices)
        return [[self.episodes[i].action()] for i in indices]

    def get_action(self):
        return self.get_action_batch([0])[0]
