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

from .control import PolicyFailure, rotation, vector, groups, feature
from .motion import SemanticRobot as Robot
from .trace import append_event


class Episode:
    MAX_PERCEPTION_RECOVERIES = 3

    def __init__(self, name, memory, perception, log, startup=None, tick_factory=None,
                 strategy=None, strategy_agent=None):
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
        self.handoffs = 0
        self.recovery_generator = None
        self.suspended_command = None
        self.command_error = None
        self.perception_recoveries = 0
        self.perception_attempts = 0
        self.finished = False
        self.trace_error = None
        self.strategy, self.strategy_agent = strategy, strategy_agent
        self.strategy_response = None

    def interrupt_strategy(self, error):
        """Cancel unacknowledged executor work, retaining the shared Robot."""
        if self.strategy is None or not self.strategy.can_interrupt():
            return False
        if self.recovery_generator is not None:
            self.recovery_generator.close()
            self.recovery_generator = None
        self.suspended_command = None
        self.reject_action(error)
        return True

    def reject_action(self, error):
        """Reject the unexecuted path without acknowledging its primitive."""
        self.queue.clear()
        self.pending = None
        self.command_pending = None
        self.settle = 0
        generator = self.active_generator()
        if generator is None or inspect.getgeneratorstate(generator) == inspect.GEN_CLOSED:
            self.fail(error)
            return
        self.command_error = error
        self.event(action_rejected=dict(reason=error.reason,detail=str(error)),native_success=None)

    def fail(self, error):
        self.queue.clear()
        self.pending = self.command_pending = self.command_error = None
        self.failure = dict(reason=getattr(error, 'reason', 'observation_or_configuration'),
                            detail=str(error), stage=self.robot.stage if self.robot else 'startup')
        self.event(controller_failure=self.failure, native_success=None)

    def active_generator(self):
        return self.recovery_generator if self.recovery_generator is not None else self.generator

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
            # Preserve explicit freshness and motor-output contract rejection.
            raise
        except Exception as error:
            if not self.failure:
                self.fail(PolicyFailure('controller_exception', type(error).__name__ + ': ' + str(error)))
            return self.hold()

    def _advance(self):
        self.steps += 1
        if self.robot is not None:
            self.robot.native_steps = self.steps
            if self.strategy is not None:
                # A watchdog can interrupt a partially executed grip command.
                # Preserve measured closure even before that primitive's ack.
                self.robot.closed_arms={a for a in ('left','right')
                    if self.observation['state'][a+'_ee_joint_state'][0]<.95}
            # Proprioception remains usable when the RGB observation fails.
            # Recovery must start from the measured EEF, not the last goal.
            for arm in ('left','right'):
                actual=list(self.observation['state'][arm+'_ee_pose'])
                self.robot.ee[arm]=actual
                if arm in self.robot.holding:
                    self.robot.holding[arm].update(ee_position=np.asarray(actual[:3]),ee_q=actual[3:])
        if self.failure or self.finished:
            return self.hold()
        try:
            if self.steps > self.memory["max_steps"]:
                raise PolicyFailure("physical_step_budget")
            if (self.strategy is not None and self.command_error is None
                    and self.strategy.watchdog(self.steps)):
                if __package__ == 'dsr.skills.native_specialists':
                    from dsr.harness.strategy import StrategyError
                else:
                    from .strategy_runtime import StrategyError
                self.interrupt_strategy(StrategyError('strategy_segment_stalled'))
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
            if self.strategy_response is not None:
                decision, error = self.strategy_response
                self.strategy_response = None
                command = self.generator.throw(error) if error else self.generator.send(decision)
            elif self.command_error is not None:
                error,self.command_error=self.command_error,None
                command=self.active_generator().throw(error)
            elif self.command_pending is not None:
                command = self.command_pending
                self.command_pending = None
            elif self.recovery_generator is not None:
                command=self.recovery_generator.send(None)
            elif self.generator is None:
                home={arm:self.hold()[arm+"_ee_pose"] for arm in ("left","right")}
                self.robot=Robot(self.memory,self.observation["instruction"],home)
                self.robot.native_steps=self.steps
                self.robot.closed_arms={a for a in ('left','right')
                    if self.observation['state'][a+'_ee_joint_state'][0]<.95}
                if self.startup is not None:
                    from PhysicalRSI_core.contracts import Context
                    self.generator=self.startup(self.robot,Context(self.channel))
                else:
                    raise PolicyFailure("missing_policy_startup")
                command=next(self.generator)
            else:
                command=self.active_generator().send(None)
            for _ in range(24):
                self.event(command=command,controller=getattr(self.robot,'controller',self.name))
                op=command["op"]
                if op == 'consult_strategy':
                    if (self.strategy is None or self.strategy_agent is None or self.queue
                            or self.pending or self.recovery_generator is not None):
                        raise PolicyFailure('strategy_consultation_contract')
                    self.event(strategy=dict(kind='consultation', request=command['request']), native_success=None)
                    try:
                        decision = self.strategy_agent.decide(command['request'], deepcopy(self.observation))
                    except (ValueError, RuntimeError, TimeoutError) as error:
                        self.strategy_response = (None, error)
                    else:
                        self.strategy_response = (decision, None)
                    # The plan is applied on the next real frame. No queued
                    # physical action or old observation crosses consultation.
                    return self.hold()
                if op == 'strategy_event':
                    if self.strategy is None:
                        raise PolicyFailure('strategy_event_without_supervisor')
                    self.event(strategy=command['record'], native_success=None)
                    return self.hold()
                if op == 'select_perception':
                    profile=command.get('profile')
                    if (self.strategy is None or profile not in
                            getattr(self.perception,'supported_profiles',())):
                        raise PolicyFailure('perception_profile_contract')
                    self.perception.perception_profile=profile
                    self.robot.perception_status=dict(profile=profile,fresh_observation_required=True)
                    self.event(perception_profile=profile,native_success=None)
                    command=self.generator.send(None)
                    continue
                if op=='handoff':
                    receipt=command['receipt']
                    if (self.handoffs or self.queue or self.pending or self.robot.holding or self.robot.hypotheses
                            or self.robot.closed_arms or receipt['channel']!=self.channel
                            or receipt['sequence']!=self.sequence
                            or receipt['consumed_steps']!=self.steps
                            or receipt['total_step_limit']!=self.memory['max_steps']):
                        raise PolicyFailure('handoff_state_contract')
                    self.handoffs+=1
                    self.event(handoff=command,native_success=None)
                    # Return one hold action, then resume stage B only from the
                    # next real observation. Old actions cannot cross this boundary.
                    return self.hold()
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
                        self.perception.require_carton_lid=getattr(self.robot,'carton_lid_required',False)
                        self.perception.required_objects=command.get('required')
                        objects=self.perception.detect(self.observation,tracked)
                        # Validate requested geometry before acknowledging the
                        # observation. Missing features leave the task suspended
                        # here, so recovery can deliver a new scene to the same
                        # subgoal without restarting earlier manipulation.
                        for kind,request in command.get('required',{}).items():
                            items=groups({'objects':objects},kind,request.get('count'))
                            for item in items:
                                for key in request.get('features',[]):feature(item,key)
                    except PolicyFailure as error:
                        diagnostic=getattr(self.perception,'last_detection',{})
                        self.robot.perception_status=dict(
                            profile=getattr(self.perception,'perception_profile','default'),
                            required=deepcopy(command.get('required')),
                            failure=dict(reason=error.reason,detail=str(error)),
                            objects=deepcopy(diagnostic.get('objects',{})),
                            independent_parts=deepcopy(diagnostic.get('independent_parts',[])))
                        self.event(perception_failure=dict(reason=error.reason,detail=str(error)),
                                   perception_diagnostic=getattr(self.perception,'last_detection',None))
                        if self.memory.get('recover_perception',False):
                            if self.perception_attempts >= self.memory.get(
                                    'max_perception_recoveries', self.MAX_PERCEPTION_RECOVERIES):
                                self.event(recovery_exhausted=dict(reason=error.reason,
                                    attempts=self.perception_attempts),native_success=None)
                                self.perception_attempts=0
                                # Preserve task-specific alternatives (for example,
                                # the egg policy's separate lid observation path).
                                command=self.generator.throw(error)
                                continue
                            # Leave the task generator suspended at observe.
                            # Recovery primitives consume the same native budget.
                            self.perception_recoveries+=1
                            self.perception_attempts+=1
                            self.perception.recovery_attempt=self.perception_recoveries
                            self.suspended_command=command
                            self.recovery_generator=self.robot.recover(error)
                            command=next(self.recovery_generator)
                            continue
                        command=self.generator.throw(error)
                        continue
                    self.previous.update(deepcopy(objects))
                    self.robot.perception_status=dict(
                        profile=getattr(self.perception,'perception_profile','default'),
                        accepted_sequence=self.sequence,
                        feature_errors={k:v['feature_error'] for k,v in objects.items() if 'feature_error' in v})
                    self.perception_attempts=0
                    scene=dict(source="rgb",frame="world",unit="m",channel=self.channel,
                               sequence=self.sequence,objects=objects,robot_state=deepcopy(self.observation["state"]))
                    self.event(observation_accepted=dict(
                        controller=getattr(self.robot,'controller',self.name),
                        grippers={arm:self.observation['state'][arm+'_ee_joint_state']
                                  for arm in ('left','right')}))
                    command=self.generator.send(scene)
                    continue
                action=self.hold()
                if op=="recovery":
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
            if self.recovery_generator is not None:
                self.recovery_generator=None
                self.command_pending,self.suspended_command=self.suspended_command,None
                self.event(recovery_finished=dict(resume='same_suspended_observe'),native_success=None)
                return self.hold()
            self.result=completed.value
            self.finished=True
            self.event(controller_result=self.result,native_success=None)
        except (PolicyFailure,ValueError,KeyError) as error:
            if self.recovery_generator is not None and self.suspended_command is not None:
                # A failed view-clearing action does not cancel the task
                # generator. Retry its suspended observation after a fresh
                # simulator frame; the attempt cap above bounds this loop.
                self.event(recovery_failed=dict(reason=getattr(error,'reason','recovery_error'),
                    detail=str(error),attempt=self.perception_recoveries),native_success=None)
                self.recovery_generator=None
                self.queue.clear()
                self.pending=self.command_error=None
                self.settle=0
                self.command_pending,self.suspended_command=self.suspended_command,None
                return self.hold()
            if (isinstance(error,PolicyFailure) and error.reason in
                    {'eef_target_unreachable','cross_arm_clearance'}
                    and self.memory.get('recover_motion',False)):
                self.reject_action(error)
                return self.hold()
            hard = {'physical_step_budget','command_budget','episode_changed','fresh_observation_required',
                    'action_fields','invalid_orientation','gripper_contract','observation_contract',
                    'stale_observation','strategy_consultation_contract'}
            if getattr(error, 'reason', None) not in hard and self.interrupt_strategy(error):
                return self.hold()
            self.fail(error)
        except Exception as error:
            # A broken generator cannot safely be replayed after manipulation.
            # Preserve its failure and let the native evaluator finish this
            # layout; never crash the service or claim controller completion.
            failure = PolicyFailure('controller_exception', type(error).__name__ + ': ' + str(error))
            if not self.interrupt_strategy(failure):
                self.fail(failure)
        return self.hold()
