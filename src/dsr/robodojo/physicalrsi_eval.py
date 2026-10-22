"""Native metrics and audit wrapper around the unmodified upstream eval loop."""
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from dsr.system2.codex_backend import write_json
from dsr.robodojo.adapter import compact

from dsr.paths import repository_root, physicalrsi_root
ROOT=repository_root()
sys.path.insert(0,str(physicalrsi_root()))
from PhysicalRSI_baselines.robodojo.evaluation import eval_one_episode as upstream_eval

class ObservedEnvironment:
    def __init__(self,env,directory,identity,*,evidence=None,progress_root=None,phase='execution'):
        self.env,self.directory,self.identity=env,directory,identity
        self.evidence=evidence
        self.progress_root=progress_root or directory.parents[1]
        self.phase=phase
        self.steps=0
        self.trace=(directory/'trace.jsonl').open('w')
    def is_episode_end(self):return self.env.is_episode_end()
    def get_obs(self):
        from XPolicyLab.policy.OpenDM.deploy import _get_valid_obs
        observation=_get_valid_obs(self.env)
        if self.evidence is not None:
            self.evidence.observation(observation)
        return observation
    def take_action(self,action):
        self.env.take_action(action)
        self.steps+=1
        if self.evidence is not None:
            self.evidence.executed(action)
            if self.steps == 1:
                self.evidence.record_stage(
                    'controller_execution',
                    observed=dict(event='first_native_action_executed',
                                  internal_task_stage='unobserved'),
                    outcome=None)
        self.trace.write(json.dumps({'step':self.steps,'action':compact(action)})+'\n')
        if self.steps%25==0:
            self.trace.flush()
            write_json(self.progress_root/'progress.json',
                       dict(self.identity,phase=self.phase,steps=self.steps))

def eval_one_episode(env,client):
    output=Path(os.environ['DSR_AGENT_OUTPUT'])
    episode=uuid.uuid4().hex
    directory=output/'episodes'/episode;directory.mkdir(parents=True,exist_ok=False)
    identity=dict(task=str(env.eval_cfg['task_name']),seed=int(env.eval_cfg['seed']),
                  layout_id=int(env.env_seeds[0]),episode_id=episode)
    write_json(output/'progress.json',dict(identity,phase='episode_start_agent'))
    retry_enabled=os.environ.get('DSR_EPISODE_RETRY','0')=='1'
    # Evidence is passive and independent of the physical-restart switch. The
    # wrapper forwards the same observations/actions and makes no extra RPCs.
    from dsr.robodojo.episode_retry import EpisodeEvidence, RecordingClient
    evidence=EpisodeEvidence(directory / 'evidence', identity=identity)
    evaluation_client=RecordingClient(client,evidence)
    wrapper=ObservedEnvironment(env,directory,identity,evidence=evidence)
    started=time.monotonic()
    try:
        upstream_eval(wrapper,evaluation_client)
        binding=client.call(func_name='dsr_episode_identity')
        if 0 in env.unstable_envs:
            summary=evidence.finalize(outcome={'status':'UNSTABLE'})
            write_json(directory/'report.json',dict(identity,status='UNSTABLE',binding=binding,
                                                    evidence=summary))
            return
        success=bool(env.success[0])
        score=(1.0 if success else float(env.reward_manager.get_score()[0])/100
               if hasattr(env,'get_score') else 0.0)
        if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid native score')
        first=dict(identity,status='COMPLETED',success=success,score=score,steps=wrapper.steps)
        report=dict(identity,status='COMPLETED',mode='frozen',
                    protocol="physicalrsi_frozen_episode_start/v1",first_attempt=first,
                    agent_calls=1,accepted=False,physical_rollouts=1,physical_steps=wrapper.steps,
                    binding=binding,wall_seconds=time.monotonic()-started)
        report['task_policy_route'] = evidence.task_policy_route
        env._dsr_first_result=first
        # Persist available observations before any reset. `last` is explicitly
        # the last policy-visible observation, not an invented terminal image.
        report['evidence']=evidence.finalize(outcome=first)
        if retry_enabled:
            from dsr.system2.episode_retry import PROTOCOL
            report.update(protocol=PROTOCOL,retry_attempted=False,retry_success=None,
                          final_success=success,recovery_status='NOT_NEEDED' if success else 'PENDING',
                          retry_requested=not success,retry_planner_started=False,
                          retry_execution_started=False,retry_completed=False,retry_steps=0)
            # Persist valid first-attempt evidence before any reset or retry RPC.
            write_json(directory/'report.json',report)
            if not success:
                from dsr.robodojo.episode_retry import retry_failed_episode
                try:
                    wrapper.trace.flush()
                    retry_failed_episode(env,client,directory,identity,first,evidence,report,
                                         wrapper_class=ObservedEnvironment,execute=upstream_eval)
                except Exception as exc:
                    # Even failure to prepare a recovery directory cannot
                    # invalidate a completed first-attempt benchmark record.
                    report.update(recovery_status='INFRASTRUCTURE_FAILED',
                                  recovery_error=f'{type(exc).__name__}: {exc}')
            report['wall_seconds']=time.monotonic()-started
        write_json(directory/'report.json',report)
        with (output/'episode_events.jsonl').open('a') as stream:
            event=dict(identity,report=str(directory/'report.json'),
                first_success=success,first_score=score,agent_calls=report['agent_calls'],accepted=report['accepted'],
                physical_rollouts=report['physical_rollouts'],physical_steps=report['physical_steps'],
                candidate_success=report.get('retry_success'),protocol=report['protocol'],binding=binding,
                evidence=report.get('evidence'),recovery_status=report.get('recovery_status'))
            if retry_enabled:
                event.update(retry_attempted=report['retry_attempted'],retry_success=report['retry_success'],
                             final_success=report['final_success'],retry_binding=report.get('retry_binding'))
                event.update({key:report.get(key) for key in ('retry_requested',
                    'retry_planner_started','retry_execution_started','retry_completed','retry_steps')})
            stream.write(json.dumps(event)+'\n')
        env._dsr_first_result=first
        write_json(output/'progress.json',dict(identity,phase='completed',steps=wrapper.steps))
    except BaseException as exc:
        summary=evidence.finalize(outcome={'status':'INFRASTRUCTURE_FAILED','error':str(exc)})
        write_json(directory/'report.json',dict(identity,status='INFRASTRUCTURE_FAILED',
            error=str(exc),steps=wrapper.steps,evidence=summary))
        raise
    finally:
        wrapper.trace.close()
