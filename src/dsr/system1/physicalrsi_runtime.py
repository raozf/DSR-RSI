"""Run pinned PhysicalRSI skill_model; substitute only DM0.5 and Codex."""
from copy import deepcopy
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sys

from dsr.paths import repository_root, physicalrsi_root
ROOT=repository_root()
UPSTREAM=physicalrsi_root()
if str(UPSTREAM) not in sys.path:sys.path.insert(0,str(UPSTREAM))

from PhysicalRSI_core.infra.storage import digest
from PhysicalRSI_baselines.robodojo.execution_skills import ExecutionSkill, load_skill as native_load_skill
from dsr.system2.physicalrsi_codex import AgentPlanner
from dsr.system2.episode_retry import CONTEXT_KEY, PROTOCOL

class ParkedDM05:
    """Keep weights on CPU when a code skill owns the policy GPU.

    Weight dtype and inference mathematics are unchanged. PhysicalRSI resets
    the preloaded skill at episode boundaries, which is the only parking point.
    """
    def __init__(self,implementation):
        self.implementation=implementation
        self.active=False
        self._park()
    def _park(self):
        import torch
        self.implementation._inference.model.to('cpu')
        self.active=False
        torch.cuda.empty_cache()
    def update_obs_batch(self,rows):
        return self.implementation.update_obs_batch(rows)
    def get_action_batch(self,indices):
        import torch
        if not self.active:
            self.implementation._inference.model.to(self.implementation._inference.device)
            self.active=True
        # Preserve FP32 master weights and BF16 mixed computation, but do not
        # retain a second cached BF16 copy of the entire VLM on a shared GPU.
        # The inner DM0.5 autocast inherits this cache setting.
        device_type = torch.device(self.implementation._inference.device).type
        # CPU inference keeps the upstream recipe's own autocast policy.
        # The cache override only exists to limit the CUDA master-weight copy.
        precision = torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False) if device_type == 'cuda' else nullcontext()
        with precision:
            return self.implementation.get_action_batch(indices)
    def reset(self):
        self.implementation.reset()
        if self.active:self._park()
    def close(self):
        self.reset()

def load_skill(descriptor, deployment):
    if descriptor.get('implementation',descriptor['name'])!='dm05':
        if deployment.get('experiment_candidate'):
            descriptor=deepcopy(descriptor)
            descriptor['configuration']['output']=str(
                Path(os.environ['PHYSICALRSI_EVAL_OUTPUT'])/'probe-code'/descriptor['name'])
        elif descriptor['configuration'].get('experimental_validation', {}).get('status') == 'experimentally_validated':
            descriptor=deepcopy(descriptor)
            descriptor['configuration']['output']=str(
                Path(os.environ['PHYSICALRSI_EVAL_OUTPUT'])/'strategy-code'/descriptor['name'])
        return native_load_skill(descriptor,deployment)
    from XPolicyLab.policy.OpenDM.model import Model as DM05
    config=deepcopy(descriptor['configuration'])
    for field in ('task_name','env_cfg_type','action_type'):
        config[field]=deployment[field]
    implementation=ParkedDM05(DM05(config))
    return ExecutionSkill(descriptor['name'],implementation,digest(descriptor))

spec=importlib.util.spec_from_file_location(
    'PhysicalRSI_baselines.robodojo._dsr_skill_model',
    UPSTREAM/'PhysicalRSI_baselines/robodojo/skill_model.py')
upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
# The pinned dependency retains the episode state machine, memory snapshots,
# operation plans, observation ordering and action queues.
upstream.AgentPlanner=AgentPlanner
upstream.load_skill=load_skill

class Model(upstream.Model):
    def __init__(self,model_cfg):
        super().__init__(model_cfg)
        try:
            from dsr.robodojo.task_policy_routing import (
                ORIGINAL_CODE_ONLY, OriginalCodeRouter, RoutedPlanner, TaskPolicyRouter)
            if 'task_policy_routing' in self.config:
                self.task_router=TaskPolicyRouter(self.config,self.deployment)
                self.planner=RoutedPlanner(self.planner,self.task_router,self.root/'routing')
            elif self.deployment.get('task_name') in ORIGINAL_CODE_ONLY:
                # Explicit task constraints apply to isolated probe libraries too.
                self.task_router=OriginalCodeRouter(self.config,self.deployment)
                self.planner=RoutedPlanner(self.planner,self.task_router,self.root/'routing')
            elif 'task_policy_probe' in self.config:
                from dsr.robodojo.task_policy_probe import TaskProbeRouter
                self.task_router=TaskProbeRouter(self.config,self.deployment)
                self.planner=RoutedPlanner(self.planner,self.task_router,self.root/'routing')
            for name,entry in self.config['skills'].items():
                if entry.get('implementation',name)=='dm05':
                    router=getattr(self,'task_router',None)
                    if router is not None and not router.allows_vla(name):
                        continue
                    self.preloaded[name]=load_skill(entry,self.deployment)
        except BaseException:
            self.close();raise

    def _start_episode(self,index,observation):
        # The pinned upstream still freezes all operation memory and binds the
        # complete executor. The router filters and validates its planner input.
        session=super()._start_episode(index,observation)
        if getattr(self,'task_router',None) is not None:
            from PhysicalRSI_core.infra.storage import atomic_json
            receipt=deepcopy(self.planner.last_route)
            receipt.update(episode_id=session['context'].episode,
                           selected_executor=session['skill'].describe())
            atomic_json(self.root/'episodes'/session['context'].episode/'task-policy-route.json',receipt)
        return session

    def dsr_episode_identity(self):
        return {**self.physicalrsi_identity(),'evidence_root':str(self.root),
                'episodes':{str(i):s['context'].episode for i,s in self.sessions.items()},
                'groups':{str(i):s['group'] for i,s in self.sessions.items()}}

    def dsr_bound_plan(self):
        """Return the actual first-action binding from the immutable episode files."""
        if set(self.sessions) != {0}:
            raise ValueError('Bound-plan capture requires one active environment')
        session = self.sessions[0]
        directory = self.root/'episodes'/session['context'].episode
        episode = json.loads((directory/'episode.json').read_text())
        decision = json.loads((directory/'agent-decision.json').read_text())
        plan = decision['plan']
        if plan.get('operations'):
            implementation = 'code-policy'
        else:
            selected = self.catalog[plan['composition']]['steps'][-1]
            descriptor = self.config['skills'][selected]
            implementation = descriptor.get('implementation', selected)
        result=dict(schema='dsr.bound-plan/v1', episode_id=session['context'].episode,
                    original_plan=plan,
                    original_executor=dict(episode['skill'], implementation=implementation))
        route=directory/'task-policy-route.json'
        if route.is_file():result['task_policy_route']=json.loads(route.read_text())
        return result

    def _update_and_capture(self, observation):
        first = not self.sessions
        result = super().update_obs(observation)
        if not first:
            return result
        # The existing update_obs RPC has already planned and bound System 1.
        # Return metadata on that same call, before any action is delivered.
        try:
            return self.dsr_bound_plan()
        except Exception as exc:
            return dict(schema='dsr.bound-plan/v1', capture_error=f'{type(exc).__name__}: {exc}')

    def update_obs(self, obs):
        context = obs.get(CONTEXT_KEY)
        if context is not None:
            if self.sessions:
                raise ValueError('Retry planning requires a fresh episode binding')
            if not isinstance(context, dict) or context.get('protocol') != PROTOCOL:
                raise ValueError('Invalid single-retry handoff')
            if (type(context.get('retry_index')) is not int or context['retry_index'] != 1
                    or type(context.get('max_restarts')) is not int or context['max_restarts'] != 1
                    or context.get('terminal', {}).get('success') is not False):
                raise ValueError('Retry requires one completed failed first attempt')
            if obs.get('instruction') != context.get('original_task_goal'):
                raise ValueError('Retry must retain the original task goal')
            row = dict(obs)
            row.pop(CONTEXT_KEY)
            images = row.pop('dsr_retry_images', [])
            self.planner.set_retry_context(context, image_urls=images)
            return self._update_and_capture(row)
        # The first-attempt path forwards exactly the original observation.
        # Existing sessions identify retry-only guidance; reset clears them.
        decision = getattr(self.planner, 'last_retry_decision', None)
        if self.sessions and decision is not None:
            row = dict(obs)
            row['instruction'] = obs['instruction'] + '\n' + decision['instruction_suffix']
            return self._update_and_capture(row)
        return self._update_and_capture(obs)

    def dsr_failure_evidence(self):
        """Read the actual bound plan after failure without running a policy."""
        if set(self.sessions) != {0}:
            raise ValueError('Failure handoff requires one completed bound environment')
        bound = self.dsr_bound_plan()
        session = self.sessions[0]
        directory = self.root/'episodes'/session['context'].episode
        operation_path = directory/'operation-plan.json'
        complete_plan = dict(bound['original_plan'])
        if operation_path.is_file():
            complete_plan['bound_stage_programs'] = json.loads(operation_path.read_text())
        execution_path = directory/'execution.json'
        return dict(original_plan=complete_plan,
                    original_executor=bound['original_executor'],
                    binding=self.dsr_episode_identity(),
                    server_buffered_actions=deepcopy(self.action_queues.get(0, [])),
                    execution=json.loads(execution_path.read_text()) if execution_path.is_file()
                              else dict(delivered_chunks=0, steps=0),
                    failed_stage='unknown; infer only from supplied RGB and action evidence',
                    retry_decision=getattr(self.planner, 'last_retry_decision', None))
