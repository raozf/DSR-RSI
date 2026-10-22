"""Failure-only handoff to the full episode-start Agent and frozen skill loop."""

import base64
from collections import deque
from copy import deepcopy
import io
from itertools import islice
import json
import math
from pathlib import Path
import traceback

from dsr.robodojo.adapter import compact, instruction_of, save_images
from dsr.evidence import EvidenceRecorder, _json_value
from dsr.system2.codex_backend import write_json
from dsr.system2.episode_retry import CONTEXT_KEY, PROTOCOL


class EpisodeEvidence:
    """Observe existing calls without adding policy queries or robot actions."""

    def __init__(self, directory=None, *, identity=None, attempt=1, plan=None, config=None):
        self.initial = None
        self.last = None
        self.frames = deque(maxlen=2)
        self.chunk = []
        self.chunk_length = 0
        self.chunk_executed = 0
        self.horizons = []
        self.actions = []
        self.samples = []
        self.planner_requests = 0
        self.task_policy_route = None
        # Passive recorder: it observes calls already made by the evaluator
        # and never requests an extra frame or robot action.
        self.recorder = EvidenceRecorder(directory, identity=identity,
                                         attempt=attempt, plan=plan, config=config)

    def observation(self, observation):
        projected = self.recorder.observe(observation)
        if projected is None:
            return
        if self.initial is None:
            self.initial = projected
        self.last = projected
        self.frames.append(projected)

    def returned_chunk(self, actions):
        self.chunk_length = len(actions)
        self.chunk = [_json_value(action) for action in islice(iter(actions),
                      self.recorder.config.max_actions)]
        self.chunk_executed = 0
        self.horizons.append(self.chunk_length)
        del self.horizons[:-self.recorder.config.max_samples]
        self.recorder.returned_chunk(actions)

    def executed(self, action):
        self.actions.append(_json_value(action))
        del self.actions[:-self.recorder.config.max_actions]
        self.chunk_executed += 1
        self.recorder.executed(action)
        if self.recorder.steps % 25 == 0:
            self.samples.append(dict(step=self.recorder.steps, observation=_json_value(self.last), action=_json_value(action)))
            del self.samples[:-self.recorder.config.max_samples]

    def bind_plan(self, plan, controller=None):
        return self.recorder.bind_plan(plan, controller)

    def plan_capture_error(self, message):
        return self.recorder.plan_capture_error(message)

    def record_stage(self, stage, **kwargs):
        return self.recorder.record_stage(stage, **kwargs)

    def failure_package(self, **kwargs):
        return self.recorder.failure_package(**kwargs)

    def finalize(self, directory=None, *, identity=None, outcome=None,
                 plan=None, policy=None):
        return self.recorder.finalize(directory, identity=identity, outcome=outcome,
                                      plan=plan, policy=policy)

    def summary(self):
        return self.recorder.summary()


class RecordingClient:
    def __init__(self, client, evidence, *, context=None, image_urls=()):
        self.client, self.evidence = client, evidence
        self.context, self.image_urls = context, list(image_urls)

    def call(self, *, func_name, **kwargs):
        if func_name == 'update_obs' and self.context is not None:
            row = dict(kwargs['obs'])
            row[CONTEXT_KEY] = self.context
            row['dsr_retry_images'] = self.image_urls
            kwargs = dict(kwargs, obs=row)
            self.context = None
            self.evidence.planner_requests += 1
        result = self.client.call(func_name=func_name, **kwargs)
        if (func_name == 'update_obs' and isinstance(result, dict)
                and result.get('schema') == 'dsr.bound-plan/v1'):
            if 'capture_error' in result:
                self.evidence.plan_capture_error(result['capture_error'])
            elif (not isinstance(result.get('original_plan'), dict)
                  or not isinstance(result.get('original_executor'), dict)):
                self.evidence.plan_capture_error('Incomplete bound-plan response')
            else:
                self.evidence.bind_plan(result['original_plan'], result['original_executor'])
                # Keep the first-action route with the simulator-owned report.
                # The policy server's filesystem need not survive the next task.
                if self.evidence.task_policy_route is None:
                    self.evidence.task_policy_route = _json_value(result.get('task_policy_route'))
        if func_name == 'get_action':
            self.evidence.returned_chunk(result)
        return result


def _sample(rows, maximum):
    if len(rows) <= maximum:
        return rows
    indices = {round(i * (len(rows) - 1) / (maximum - 1)) for i in range(maximum)}
    return [rows[i] for i in sorted(indices)]


def build_handoff(evidence, server, terminal, directory):
    """Materialize RGB only after failure, with exact labels and unknowns."""
    from PIL import Image
    from PhysicalRSI_baselines.robodojo.agent_planner import agent_observation
    terminal = agent_observation(terminal)
    # Write the structured package before reset. It contains permitted RGB,
    # proprioception, actions and explicit stage observations only. It does not
    # assert a causal root cause from a terminal failure.
    package = evidence.failure_package(
        terminal={'success': False}, terminal_observation=terminal,
        original_plan=server.get('original_plan'),
        controller=server.get('original_executor'))
    write_json(directory/'failure-evidence.json', package)
    images, labels, catalog = [], [], []
    observations = [('initial', evidence.initial)]
    observations += [(f'pre_terminal_{i}', value) for i, value in enumerate(evidence.frames)]
    observations.append(('terminal', terminal))
    legacy = []
    for prefix, observation in observations:
        if observation is None:
            continue
        paths = save_images(observation, directory/'images', prefix)
        for path in paths:
            ref = Path(path).stem
            legacy.append(dict(ref=ref, observation=prefix, path=path))

    def attach_legacy(item):
        if item['ref'] in labels or len(images) >= evidence.recorder.config.max_agent_images:
            return
        stream = io.BytesIO()
        with Image.open(item['path']) as image:
            image.convert('RGB').save(stream, format='PNG')
        images.append('data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode())
        labels.append(item['ref'])
        catalog.append(dict(item, kind='rgb'))

    def attach_keyframe(frame):
        if (frame['ref'] in labels or len(images) >= evidence.recorder.config.max_agent_images
                or evidence.recorder.directory is None):
            return
        path = evidence.recorder.directory / frame['path']
        data = path.read_bytes()
        from dsr.evidence import sha256
        if sha256(path) != frame['sha256']:
            raise ValueError('Recorded recovery image hash changed')
        images.append('data:image/png;base64,' + base64.b64encode(data).decode())
        labels.append(frame['ref'])
        catalog.append(dict(frame, kind='rgb', path=str(path)))

    # Reserve the scarce slots for both ends of the attempt and observable
    # interaction boundaries before adding redundant camera views.
    for prefix in ('terminal', 'initial'):
        available = [item for item in legacy if item['observation'] == prefix]
        if available:
            attach_legacy(next((item for item in available if item['ref'].endswith('_cam_head')),
                               available[0]))
    stage_frames = [frame for frame in evidence.recorder.keyframes
                    if frame['reason'] in ('stage_boundary', 'pre_event', 'post_event')]
    for frame in stage_frames[:evidence.recorder.config.max_agent_images // 2]:
        attach_keyframe(frame)
    for item in legacy:
        attach_legacy(item)
    for frame in evidence.recorder.keyframes:
        if frame['reason'] not in ('initial', 'final_available_observation'):
            attach_keyframe(frame)
    stage_events = package['stage_events']
    catalog += [dict(ref=row['ref'], kind='controller_stage', content=row) for row in stage_events]
    stage_refs = {ref for row in stage_events for ref in row.get('evidence_refs', [])}
    catalog += [dict(ref=row['ref'], kind='observation_trace', content=row)
                for row in evidence.recorder.events if row['ref'] in stage_refs
                and row['ref'] not in {entry['ref'] for entry in catalog}]
    catalog.append(dict(ref='evidence_plan', kind='passive_collection_plan', content=package['evidence_plan']))
    catalog += [dict(ref=name, kind=kind) for name, kind in (
        ('terminal', 'native_outcome'), ('original_plan', 'bound_episode_plan'),
        ('original_executor', 'bound_implementation'), ('action_queue', 'delivered_vs_executed'),
        ('action_sequence', 'sampled_executed_actions'), ('state_history', 'observed_proprioception'))]
    instruction = instruction_of(evidence.initial)
    context = dict(protocol=PROTOCOL, retry_index=1, max_restarts=1,
        original_task_goal=instruction,
        original_plan=server['original_plan'], original_executor=server['original_executor'],
        failed_stage=server.get('failed_stage', 'unknown'),
        terminal=dict(success=False),
        initial_observation=compact(evidence.initial), terminal_observation=compact(terminal),
        observed_state_history=_sample(evidence.samples, 12),
        scene_relations=dict(status='unknown; infer from attached RGB with explicit uncertainty'),
        cause_candidates=dict(status='not diagnosed; Agent must return categories, confidence and evidence'),
        required_changes=dict(status='Agent must compare the failed plan with registered alternatives'),
        action_sequence=dict(executed_count=evidence.recorder.steps,
            sampled_actions=_sample(evidence.actions, 32),
            last_actions=evidence.actions[-8:],
            full_trace_path=str(directory.parent/'trace.jsonl'),
            sampling='full trace is retained on disk; planner receives sampled actions and terminal tail'),
        action_horizon=dict(observed_chunk_lengths=_sample(evidence.horizons, 32)),
        action_queue=dict(last_delivered_chunk=compact(evidence.chunk),
            delivered_count=evidence.chunk_length,
            unrecorded_delivered_actions=max(0, evidence.chunk_length - len(evidence.chunk)),
            executed_from_last_chunk=evidence.chunk_executed,
            unexecuted_delivered_actions=compact(evidence.chunk[evidence.chunk_executed:]),
            server_buffered_actions=compact(server['server_buffered_actions'])),
        previous_retry_guidance='', evidence_catalog=catalog, image_labels=labels,
        evidence_plan=evidence.recorder.plan,
        evidence_manifest=evidence.summary(),
        failure_evidence_package=package,
        keyframe_refs=[label for label in labels if label.startswith('keyframe_')],
        stage_events=stage_events,
        earliest_observed_deviation=evidence.recorder.earliest_deviation,
        completed_stages=list(evidence.recorder.completed_stages),
        first_attempt_execution=server['execution'],
        available_alternatives='The same registered catalogue and operation contracts are supplied by the episode-start planner; no new skill is admitted')
    write_json(directory/'failure_context.json', context)
    return context, images


def retry_failed_episode(env, client, directory, identity, first, evidence, report, *, wrapper_class, execute):
    """One reset and one invocation of the original evaluation/planning path."""
    if first.get('status') != 'COMPLETED' or first.get('success') is not False:
        raise ValueError('Only a completed failed first attempt may restart')
    directory = directory/'retry'
    directory.mkdir(exist_ok=False)
    report.update(protocol=PROTOCOL, retry_attempted=False, retry_success=None,
                  final_success=bool(first['success']), recovery_status='PREPARING',
                  retry_requested=True, retry_planner_started=False,
                  retry_execution_started=False, retry_completed=False,
                  retry_phase='evidence', retry_steps=0)
    # These fields feed native aggregation. Simulator reset must not cause the
    # first attempt to be skipped as unstable or counted as a second episode.
    fields = ('success', 'end_flag', 'take_action_cnt', 'episode_nums',
              'unstable_envs', 'unstable_nums', 'success_nums', 'fail_nums',
              'total_score', 'env_seeds', 'current_env_seed_map')
    snapshot = {name: deepcopy(getattr(env, name)) for name in fields if hasattr(env, name)}
    retry_evidence = EpisodeEvidence(directory / 'evidence', identity=identity,
                                     attempt=2)
    wrapper = None
    try:
        server = client.call(func_name='dsr_failure_evidence')
        terminal = env.get_obs_batch([0], last_frame=True)[0]
        context, images = build_handoff(evidence, server, terminal, directory)
        context['terminal'].update(score=first['score'], steps=first['steps'])
        write_json(directory/'failure_context.json', context)
        # Finalize the first video before reset aborts its open writers. Retry
        # videos use another directory and cannot replace the original rollout.
        if hasattr(env, 'save_video'):
            video_path = directory.parent/'first_rollout.mp4'
            if hasattr(env, 'save_dir') and hasattr(env, 'success_nums') and hasattr(env, 'fail_nums'):
                video_path = Path(env.save_dir)/f'episode_{env.success_nums + env.fail_nums:07d}.mp4'
            try:
                env.save_video(0, str(video_path), 'fail')
            except Exception as exc:
                report['first_video_error'] = f'{type(exc).__name__}: {exc}'
        report['retry_phase'] = 'reset'
        write_json(directory.parent/'report.json', report)
        reset = getattr(env, 'reset_same_layout', env.reset)
        reset(seed=[identity['layout_id']])
        if int(env.env_seeds[0]) != identity['layout_id']:
            raise ValueError('Retry layout differs from the first attempt')
        env.run_reward()
        if hasattr(env, 'get_score'):
            env.get_score()
        wrapper = wrapper_class(env, directory, identity, evidence=retry_evidence,
                                progress_root=directory.parents[2], phase='retry_execution')
        retry_client = RecordingClient(client, retry_evidence, context=context, image_urls=images)
        # This exact loop calls reset, update_obs -> AgentPlanner -> registered
        # binding, and the selected executor. It is not the DM0.5-only adapter.
        report['retry_attempted'] = True
        report['retry_phase'] = 'planning_and_execution'
        write_json(directory.parent/'report.json', report)
        execute(wrapper, retry_client)
        binding = client.call(func_name='dsr_episode_identity')
        if 0 in env.unstable_envs:
            raise RuntimeError('Retry simulator was unstable')
        success = bool(env.success[0])
        score = 1.0 if success else float(env.reward_manager.get_score()[0])/100 if hasattr(env, 'get_score') else 0.0
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid retry native score')
        server = client.call(func_name='dsr_failure_evidence')
        retry = dict(identity, status='COMPLETED', success=success, score=score,
                     steps=wrapper.steps, binding=binding, retry_decision=server.get('retry_decision'))
        write_json(directory/'result.json', retry)
        report.update(retry_attempt=retry, retry_success=success,
                      final_success=bool(first['success'] or success),
                      accepted=success, recovery_status='COMPLETED', retry_binding=binding,
                      retry_completed=True, retry_phase='completed')
    except Exception as exc:
        report.update(recovery_status='INFRASTRUCTURE_FAILED',
                      recovery_error=f'{type(exc).__name__}: {exc}')
        write_json(directory/'error.json', dict(error=report['recovery_error'],
            phase=report['retry_phase'], traceback=traceback.format_exc()))
    finally:
        report['agent_calls'] = 1 + retry_evidence.planner_requests
        report['retry_planner_started'] = retry_evidence.planner_requests > 0
        report['retry_steps'] = wrapper.steps if wrapper is not None else 0
        report['retry_execution_started'] = report['retry_steps'] > 0
        report['physical_rollouts'] = 1 + int(report['retry_execution_started'])
        report['physical_steps'] = first['steps'] + report['retry_steps']
        try:
            if wrapper is not None and hasattr(env, 'save_video'):
                env.save_video(0, str(directory/'rollout.mp4'), 'success' if report['retry_success'] else 'fail')
        except Exception as exc:
            report['retry_video_error'] = str(exc)
        try:
            if report['retry_attempted']:
                client.call(func_name='reset')
        except Exception as exc:
            report['retry_cleanup_error'] = str(exc)
        finally:
            try:
                retry_evidence.finalize(outcome=report.get('retry_success'))
            except Exception as exc:
                report['retry_evidence_error'] = f'{type(exc).__name__}: {exc}'
            for name, value in snapshot.items():
                setattr(env, name, value)
            env._dsr_first_result = first
            if wrapper is not None:
                wrapper.trace.close()
    return report
