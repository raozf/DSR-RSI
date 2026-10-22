#!/usr/bin/env python3
"""Durable full RoboDojo evaluation with disjoint policy/simulator GPU pairs.

OpenDM is the executor baseline. DSRRSI defaults to the pinned PhysicalRSI
episode-start selection protocol; failure adaptation is a separate mode.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback

import yaml
from verify_dm05_checkpoint import verify
from check_dm05_training import validate as validate_training

from dsr.paths import repository_root
from dsr.robodojo.accuracy_output import (
    read_evaluation_checkpoint, task_accuracy, write_task_accuracy)
from dsr.runtime_config import (checkpoint_root, interpreter, process_environment,
    simulator_command, preflight, local_state_root, verify_state_lock)
ROOT = repository_root()
ROBO = ROOT / 'vendor/RoboDojo'
POLICY = ROOT / 'vendor/XPolicyLab/policy/OpenDM'
CHECKPOINT = checkpoint_root() / 'DM05-MEM-Robodojo-Sim'
POLICY_PYTHON = interpreter('policy', required=False)
AGENT_CONFIG = Path(os.environ['DSR_CODEX_HOME'])/'config.toml' if os.environ.get('DSR_CODEX_HOME') else None


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def evaluation_state(args):
    """Keep operational state outside the v6 result-file contract."""
    output = (Path(args.output_root)/args.run_id).resolve()
    root = (Path(args.state_root) if getattr(args, 'state_root', None)
            else local_state_root()/'evaluation-state')
    state = (root/args.run_id).resolve()
    if state == output or state.is_relative_to(output):
        raise ValueError('state-root must be outside the evaluation output directory')
    return state


def capture_server_output(process):
    """Drain worker stdout into bounded RAM for failure diagnosis; create no log."""
    process._dsr_output_tail = deque(maxlen=64)
    if getattr(process, 'stdout', None) is None:
        return
    def drain():
        try:
            while block := process.stdout.read(1024):
                process._dsr_output_tail.append(block.decode(errors='replace'))
        finally:
            process.stdout.close()
    reader = threading.Thread(target=drain, daemon=True)
    process._dsr_output_reader = reader
    reader.start()


def server_output(process):
    reader = getattr(process, '_dsr_output_reader', None)
    if reader is not None and process.poll() is not None:
        reader.join(timeout=1)
    return ''.join(getattr(process, '_dsr_output_tail', ()))


def prepare_skill_library(library, source):
    """Refuse changed resume inputs; never overwrite existing run evidence."""
    if source:
        payload = Path(source).resolve(strict=True).read_bytes()
        if library.exists() and library.read_bytes() != payload:
            raise ValueError('Run skill library differs from the requested frozen source')
        if not library.exists():
            library.write_bytes(payload)
        return {'path': str(Path(source).resolve()),
                'sha256': hashlib.sha256(payload).hexdigest()}
    return None


def validate_probe_run(config, bindings_path, library_path, keys, args):
    """Preflight the entire measurement schedule before starting any GPU."""
    if not bindings_path and 'task_policy_probe' not in config:
        if getattr(args, 'allow_unverified_low_sr_measurement', False):
            raise ValueError('Unverified measurement mode requires an isolated probe')
        return None
    if (not bindings_path or 'task_policy_probe' not in config
            or args.policy_name != 'DSRRSI' or args.agent_strategy != 'physicalrsi'
            or args.agent_mode != 'frozen' or args.attempts != 1
            or args.episode_retry or args.sequential_seeds):
        raise ValueError('Isolated probes require bindings, one attempt, no retry or seed restart loop')
    from dsr.robodojo.task_policy_probe import TaskProbeRouter
    payload = Path(bindings_path).read_bytes()
    record = json.loads(payload)
    library_sha = hashlib.sha256(Path(library_path).read_bytes()).hexdigest()
    if (record.get('schema') != 'dsr.low-sr-probe-bindings/v1'
            or record.get('library_sha256') != library_sha
            or record.get('promotion_allowed') is not False):
        raise ValueError('Probe bindings do not match this frozen library')
    for key in keys:
        candidate = record.get('bindings', {}).get(key)
        task = key.split('/', 1)[1]
        if not isinstance(candidate, str) or not candidate:
            raise ValueError('Probe schedule lacks an explicit candidate for ' + key)
        TaskProbeRouter(config, dict(task_name=task, experiment_candidate=candidate))
    from dsr.robodojo.low_sr_readiness import enforce_large_probe
    enforce_large_probe(library_path,len(getattr(args,'policy_gpus',())),
        allow_unverified_measurement=getattr(args,'allow_unverified_low_sr_measurement',False))
    return hashlib.sha256(payload).hexdigest()


def check_run_storage(output, state, *, output_gib=10, state_gib=1):
    """Account for the artifact and lock stores separately before execution."""
    import shutil
    for role, path, minimum in (('artifacts', output, output_gib),
                                ('active state', state, state_gib)):
        free = shutil.disk_usage(path).free / 1024**3
        if free < minimum:
            raise RuntimeError(f'Insufficient {role} storage at {path}: '
                               f'{free:.2f} GiB free, {minimum:g} GiB reserved')


def validate_probe_episode(event, task, seed, operation, library, agent_output):
    """Count only the specified first-attempt controller, with native evidence."""
    from dsr.robodojo.task_policy_routing import CODE_COMPOSITION
    report_path = Path(event['report']).resolve(strict=True)
    if not report_path.is_relative_to(Path(agent_output).resolve() / 'episodes'):
        raise ValueError('Probe report is outside this task run')
    report = json.loads(report_path.read_text())
    bound = json.loads((report_path.parent / 'evidence/bound-plan.json').read_text())
    route = report.get('task_policy_route') or {}
    plan = bound.get('plan', {})
    from dsr.robodojo.task_policy_probe import PORTFOLIO
    if operation == PORTFOLIO:
        task_route = library['task_policy_probe']['tasks'][task]
        operation = route.get('selected_candidate')
        if (task_route.get('selection') != 'agent_portfolio'
                or operation not in task_route['candidates']
                or route.get('selection_mode') != 'agent_portfolio'
                or set(route.get('allowed_operations', [])) != set(task_route['candidates'])):
            raise ValueError('Probe result is outside its frozen agent portfolio')
        required = sorted(name for name in task_route['candidates']
                          if library['program_registry'][name].get('stage_composition'))
        if (route.get('required_stage_operations', []) != required
                or required and operation not in required):
            raise ValueError('Portfolio result bypassed its required staged operations')
    entry = library['program_registry'].get(operation)
    if not isinstance(entry, dict):
        raise ValueError('Probe selected an operation absent from its frozen registry')
    source = entry.get('source_name', operation)
    registry_stages = entry.get('stage_composition') or []
    selected_stage = route.get('selected_stage_composition')
    if registry_stages:
        if (not isinstance(selected_stage, dict)
                or selected_stage.get('controllers') != registry_stages):
            raise ValueError('Probe route omitted or changed the registry stage composition')
        if (route.get('selection_mode') == 'agent_portfolio'
                and operation not in (route.get('required_stage_operations') or [])):
            raise ValueError('Portfolio route selected a non-staged operation')
    elif selected_stage is not None:
        raise ValueError('Probe route claims stages for a non-staged registry operation')
    for record in (event, report):
        if (record.get('task') != task or record.get('seed') != seed
                or record.get('layout_id') != event['layout_id']
                or record.get('physical_rollouts') != 1
                or record.get('retry_attempted', False)
                or record.get('protocol') != 'physicalrsi_frozen_episode_start/v1'):
            raise ValueError('Probe identity or first-attempt protocol mismatch')
    if (report.get('status') != 'COMPLETED' or bound.get('attempt') != 1
            or bound.get('step') != 0 or plan.get('composition') != CODE_COMPOSITION
            or plan.get('operations') != [operation]
            or bound.get('controller', {}).get('name') != source
            or bound.get('controller', {}).get('implementation') != 'code-policy'
            or report.get('binding', {}).get('groups') != {'0': source}
            or route.get('schema') != 'dsr.task-policy-probe-selection/v1'
            or route.get('exact_configuration') != task
            or route.get('selected_candidate') != operation
            or route.get('promotion_allowed') is not False
            or route.get('selected_plan') != plan):
        raise ValueError('Probe result lacks the exact candidate first-action binding')
    handoff = validate_stage_handoff(route, report, source, library, report_path)
    from dsr.robodojo.strategy_evidence import validate_strategy
    strategy = validate_strategy(route, report, source, report_path, entry.get('execution_strategy'))
    result = {'report': str(report_path), 'operation': operation,
              'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest()}
    if handoff is not None:
        result['stage_handoff'] = handoff
    if strategy is not None:
        result['execution_strategy'] = strategy
    return result


def validate_stage_handoff(route, report, source, library, report_path):
    """Delegate stage validation to the episode-owned evidence checker."""
    from dsr.robodojo.stage_evidence import validate_stage_handoff as validate
    controllers = library['program_registry'][route['selected_candidate']].get('stage_composition')
    return validate(route, report, source, report_path, controllers)


def validate_formal_episode(event, task, seed, library, agent_output, native_detail):
    from dsr.robodojo.formal_evidence import validate_formal_episode as validate
    return validate(event, task, seed, library, agent_output, native_detail)


def progress_signature(source, directory):
    """Ignore noisy simulator logs; only completed results/control progress count."""
    signature = []
    for path in (Path(source), Path(directory)/'agent/progress.json'):
        try:
            stat = path.stat()
            signature.append((stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            signature.append(None)
    return tuple(signature)


def single_retry_metrics(events, protocol='single_retry_same_layout/v1'):
    """Count original episodes, preserving first-pass and recovery metrics."""
    episodes = {}
    invalid = 0
    for event in events:
        if event.get('protocol') != protocol:
            continue
        fields = ('first_success', 'retry_attempted', 'final_success')
        stages = ('retry_requested', 'retry_planner_started',
                  'retry_execution_started', 'retry_completed')
        invalid_telemetry = (
            any(key in event and type(event[key]) is not bool for key in stages)
            or any(event.get(later) is True and event.get(earlier) is False
                   for earlier, later in zip(stages, stages[1:]))
            or (event.get('first_success') is True
                and any(event.get(key) is True for key in stages))
            or (event.get('retry_completed') is True
                and type(event.get('retry_success')) is not bool)
            or (event.get('retry_completed') is False
                and type(event.get('retry_success')) is bool)
            or ('retry_steps' in event and (type(event['retry_steps']) is not int
                or event['retry_steps'] < 0
                or (type(event.get('retry_execution_started')) is bool
                    and event['retry_execution_started'] != (event['retry_steps'] > 0))))
        )
        if (any(type(event.get(key)) is not bool for key in fields)
                or invalid_telemetry
                or (event.get('retry_success') is not None
                    and type(event['retry_success']) is not bool)
                or event['final_success'] != (
                    event['first_success'] or event.get('retry_success') is True)
                or (event['first_success'] and event['retry_attempted'])
                or (event.get('retry_success') is True and not event['retry_attempted'])
                or (event.get('retry_success') is not None and not event['retry_attempted'])
                or not isinstance(event.get('task'), str)
                or any(type(event.get(key)) is not int for key in ('seed', 'layout_id'))):
            invalid += 1
            continue
        # A resumed job can append the same completed layout again. Its latest
        # complete record replaces the prior record, rather than adding a trial.
        episodes[(event['task'], event['seed'], event['layout_id'])] = event

    def counts(rows):
        total = len(rows)
        first = sum(row['first_success'] for row in rows)
        recovered = sum(not row['first_success'] and row.get('retry_success') is True for row in rows)
        attempted = sum(row['retry_attempted'] for row in rows)
        completed_retries = sum(row['retry_attempted'] and type(row.get('retry_success')) is bool for row in rows)
        # Historical events lack stage telemetry. Missing is not evidence that
        # no action happened, and must not become a fabricated zero rate.
        telemetry = [row for row in rows if not row['first_success']]
        def stage_count(key):
            return (sum(row[key] for row in telemetry)
                    if all(type(row.get(key)) is bool for row in telemetry) else None)

        def rate(numerator, denominator):
            return numerator / denominator if numerator is not None and denominator else None

        requests = stage_count('retry_requested')
        planners = stage_count('retry_planner_started')
        execution_starts = stage_count('retry_execution_started')
        eligible = total - first
        return {
            'episodes': total, 'first_successes': first, 'recovered_failures': recovered,
            'final_successes': first + recovered,
            'retry_eligible': total - first,
            'retry_attempts': attempted,
            'completed_retries': completed_retries,
            'retry_requests': requests,
            'retry_planner_starts': planners,
            'retry_execution_starts': execution_starts,
            'retry_execution_coverage': rate(execution_starts, eligible),
            'retry_completion_rate': completed_retries / attempted if attempted else None,
            'retry_request_coverage': rate(requests, eligible),
            'retry_loop_entry_coverage': rate(attempted, eligible),
            'retry_planner_coverage': rate(planners, eligible),
            'retry_execution_per_request': rate(execution_starts, requests),
            'retry_completion_per_execution': rate(completed_retries, execution_starts),
            'retry_recovery_rate': rate(recovered, eligible),
            'executed_retry_success_rate': rate(recovered, execution_starts),
            'recovery_errors': sum(row.get('recovery_status') == 'INFRASTRUCTURE_FAILED' for row in rows),
            'first_success_rate': first / total if total else None,
            'final_success_rate': (first + recovered) / total if total else None,
            'conditional_retry_success_rate': recovered / attempted if attempted else None,
            'conditional_retry_denominator': 'all attempted retries, including infrastructure failures',
            'first_success_wilson_95': wilson_interval(first, total),
            'final_success_wilson_95': wilson_interval(first + recovered, total),
            'conditional_retry_wilson_95': wilson_interval(recovered, attempted),
        }

    rows = list(episodes.values())
    return {
        **counts(rows),
        'protocol': protocol, 'success_rate_unit': 'fraction',
        'aggregation': 'Unique task/seed/layout episodes; episode-weighted, not the official dimension mean',
        'invalid_records': invalid,
        'per_task': {task: counts([row for row in rows if row['task'] == task])
                     for task in sorted({row['task'] for row in rows})},
    }


def wilson_interval(successes, trials):
    """Descriptive binomial interval; episodes may still share task structure."""
    if not trials:
        return None
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / denominator
    radius = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    return [max(0.0, center-radius), min(1.0, center+radius)]


def agent_metrics(root):
    events = []
    for path in Path(root).glob('tasks/seed_*/*/agent/episode_events.jsonl'):
        # A interrupted append must not take down the status heartbeat.
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    routing_retry = 'episode_start_single_retry_same_layout/v1'
    frozen = [e for e in events if e.get('protocol') in {
        'physicalrsi_frozen_episode_start/v1', routing_retry}]
    skills = {}
    retry_skills = {}
    for event in frozen:
        for name in event.get('binding', {}).get('groups', {}).values():
            skills[name] = skills.get(name, 0) + 1
        for name in (event.get('retry_binding') or {}).get('groups', {}).values():
            retry_skills[name] = retry_skills.get(name, 0) + 1
    return {
        'completed_episode_records': len(events),
        'agent_calls': sum(e.get('agent_calls', 0) for e in events),
        'accepted_repairs': sum(bool(e.get('accepted')) for e in events),
        'physical_rollouts': sum(e.get('physical_rollouts', 0) for e in events),
        'physical_steps': sum(e.get('physical_steps', 0) for e in events),
        'candidate_successes': sum(e.get('candidate_success') is True for e in events),
        'episode_start_selections': len(frozen),
        'selected_skills': skills,
        'retry_selected_skills': retry_skills,
        'single_retry_recovery': single_retry_metrics(events),
        'episode_start_retry_recovery': single_retry_metrics(events, routing_retry),
        'note': 'All completed attempts including retried jobs; separate from first-pass benchmark metrics',
    }


def vendor_module(name):
    path = ROBO / 'scripts/internal' / (name + '.py')
    spec = importlib.util.spec_from_file_location('full_eval_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory():
    records = vendor_module('task_inventory').build_inventory()['tasks']
    counts = yaml.safe_load((ROBO / 'task/RoboDojo/config/_task.yml').read_text())
    for record in records:
        record['episodes'] = int(counts['tasks'].get(record['name'], {}).get(
            'eval_nums', counts['common']['eval_nums']))
    if any(not r['runnable'] for r in records):
        raise RuntimeError('A benchmark task is missing its class or config')
    official = vendor_module('summarize_result')
    required = set(official.ALL_TASKS) | {n + '_random' for n in official.GENERALIZATION_TASKS}
    if {r['name'] for r in records} != required:
        raise RuntimeError('Runnable task inventory does not match the official benchmark')
    return records


def validate_result(data, expected, partial=False, expected_layout_ids=None):
    """Reject incomplete, duplicate, non-finite, or internally inconsistent metrics."""
    if not isinstance(data, dict) or not isinstance(data.get('details'), dict):
        raise ValueError('Missing episode details')
    details = data['details']
    count = data.get('eval_time')
    if type(count) is not int or count != len(details) or count > expected or count < 0:
        raise ValueError('Episode count mismatch')
    if not partial and count != expected:
        raise ValueError('Native episode count not reached')
    if set(details) != {str(i) for i in range(count)}:
        raise ValueError('Episode indices must be contiguous integers')
    layouts, successes, scores = [], 0, []
    for entry in details.values():
        layout = entry.get('layout_id')
        if type(layout) is not int or layout < 0 or type(entry.get('success')) is not bool:
            raise ValueError('Invalid layout or success field')
        score = entry.get('score')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid episode score')
        layouts.append(layout)
        successes += int(entry['success'])
        scores.append(score)
    if len(set(layouts)) != count:
        raise ValueError('Duplicate evaluation layout')
    if expected_layout_ids is not None:
        required = set(expected_layout_ids)
        if len(required) != expected:
            raise ValueError('Required layout count differs from native episode count')
        if not set(layouts) <= required:
            raise ValueError('Episode layouts are outside the required native layout set')
        if not partial and set(layouts) != required:
            raise ValueError('Required native layout coverage not reached')
    sr = successes / count if count else 0.0
    score = 100 * sum(scores) / count if count else 0.0
    if not math.isclose(float(data.get('success_rate', -1)), sr, abs_tol=1e-7):
        raise ValueError('Reported success rate differs from episode records')
    if not math.isclose(float(data.get('score', -1)), score, abs_tol=1e-6):
        raise ValueError('Reported score differs from episode records')
    return {'episodes_completed': count, 'success_rate': sr, 'score': score}


def aggregate(rows, records, seeds):
    """Match vendor: merge 25+25, average tasks within dimension, then dimensions."""
    official = vendor_module('summarize_result')
    complete = {(r['task'], r['seed']): r for r in rows if r.get('status') == 'COMPLETED'}
    per_seed = {}
    for seed in seeds:
        dims = {}
        for dimension, tasks in official.DIMENSIONS.items():
            values = []
            for task in tasks:
                needed = [task, task + '_random'] if task in official.GENERALIZATION_TASKS else [task]
                if any((n, seed) not in complete for n in needed):
                    continue
                entries = [complete[(n, seed)] for n in needed]
                values.append((statistics.mean(r['success_rate'] for r in entries),
                               statistics.mean(r['score'] for r in entries)))
            dims[dimension] = {'completed_tasks': len(values), 'total_tasks': len(tasks),
                              'success_rate': statistics.mean(v[0] for v in values) if len(values) == len(tasks) else None,
                              'score': statistics.mean(v[1] for v in values) if len(values) == len(tasks) else None}
        all_done = all(d['success_rate'] is not None for d in dims.values())
        gen_split = {}
        for variant, suffix in [('Std', ''), ('Rand', '_random')]:
            names = [n + suffix for n in official.GENERALIZATION_TASKS]
            entries = [complete[(n, seed)] for n in names if (n, seed) in complete]
            gen_split[variant] = {'success_rate': statistics.mean(e['success_rate'] for e in entries) if len(entries) == len(names) else None,
                                  'score': statistics.mean(e['score'] for e in entries) if len(entries) == len(names) else None}
        per_seed[str(seed)] = {'dimensions': dims, 'generalization_split': gen_split,
                              'overall_success_rate': statistics.mean(d['success_rate'] for d in dims.values()) if all_done else None,
                              'overall_score': statistics.mean(d['score'] for d in dims.values()) if all_done else None}
    done = (len(complete) == len(records) * len(seeds)
            and all(r["overall_success_rate"] is not None for r in per_seed.values()))
    output = {'aggregation': 'Mean across five dimension means (official RoboDojo)',
              'success_rate_unit': 'fraction', 'score_unit': 'percent', 'per_seed': per_seed,
              'overall_success_rate': None, 'overall_score': None,
              'success_rate_population_std': None, 'score_population_std': None}
    if done:
        for metric, stdkey in [('overall_success_rate', 'success_rate_population_std'),
                               ('overall_score', 'score_population_std')]:
            values = [r[metric] for r in per_seed.values()]
            output[metric] = statistics.mean(values)
            output[stdkey] = statistics.pstdev(values)
    return output


def base_env(gpu, *, role='policy'):
    return process_environment(gpu, role=role)


WORKER_ENV_EVIDENCE = (
    'CUDA_VISIBLE_DEVICES', 'DSR_LOCAL_STATE', 'TMPDIR', 'XDG_CACHE_HOME',
    'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME', 'TORCH_EXTENSIONS_DIR',
    'CUDA_CACHE_PATH', 'HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE',
    'TRANSFORMERS_CACHE', 'DSR_KIT_ARGS', 'DSR_KIT_SCRIPTING_CACHE_ROOT', 'EVAL_NUM', 'ROBODOJO_RUN_ID',
    'DSR_STRICT_NATIVE_LAYOUTS', 'DSR_EVAL_LAYOUT_IDS',
    'DSR_DISABLE_FAULT_RESTART', 'OMNI_KIT_ACCEPT_EULA',
)


def worker_environment_evidence(env):
    """Persist only execution/cache settings, never arbitrary inherited secrets."""
    return {key: env[key] for key in WORKER_ENV_EVIDENCE if key in env}


def worker_simulator_command(argv, env):
    """Override process-global Kit flags with this pair's isolated settings."""
    command = simulator_command(argv)
    clean = []
    index = 0
    while index < len(command):
        argument = command[index]
        if argument == '--kit_args':
            index += 2
            continue
        if not argument.startswith('--kit_args='):
            clean.append(argument)
        index += 1
    if env.get('DSR_KIT_ARGS'):
        clean.extend(['--kit_args', env['DSR_KIT_ARGS']])
    return clean


def kill_group(process):
    # Descendants may outlive a crashed group leader.
    if getattr(process, '_dsr_cleaned', False):
        return
    process._dsr_cleaned = True
    identity = Path(f'/proc/{process.pid}/stat')
    try:
        current_start = identity.read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        current_start = None
    original_start = getattr(process, '_dsr_start_time', current_start)
    if current_start is not None and current_start != original_start:
        return  # A former PID has been reused by a process this run does not own.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=15)


class FullRun:
    def __init__(self, args):
        self.args = args
        self.strict_native_layouts = (args.episodes is None
                                     and not getattr(args, 'layout_ids', None))
        self.out = Path(args.output_root).resolve() / args.run_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.state = evaluation_state(args)
        self.state.mkdir(parents=True, exist_ok=True)
        self.run_id = args.run_id
        self.records = inventory()
        self.policy_name = args.policy_name
        self.checkpoint = Path(args.checkpoint).resolve()
        if args.tasks:
            names = set(args.tasks.split(','))
            if names - {r['name'] for r in self.records}:
                raise ValueError('Unknown RoboDojo task name')
            self.records = [r for r in self.records if r['name'] in names]
        if args.episodes is not None:
            for record in self.records:
                record['episodes'] = min(record['episodes'], args.episodes)
        self.deferred_tasks = set(filter(None, getattr(args, 'defer_tasks', '').split(',')))
        if self.deferred_tasks - {r['name'] for r in self.records}:
            raise ValueError('Unknown deferred RoboDojo task name')
        self.seeds = args.seeds
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.phase = 'PREPARING'
        self.current_seed = None
        self.seed_round = 0
        self.processes = {}
        self.servers = {}
        self.server_tasks = {}
        self.server_candidates = {}
        self.ports = {}
        self.probe_bindings = {}
        probe_path = getattr(args, 'probe_bindings', None)
        if probe_path:
            probe = read_json(Path(probe_path).resolve())
            if not isinstance(probe, dict) or probe.get('schema') != 'dsr.low-sr-probe-bindings/v1':
                raise ValueError('Versioned low-SR probe bindings are required')
            self.probe_bindings = probe.get('bindings', {})
            if not isinstance(self.probe_bindings, dict):
                raise ValueError('Probe bindings must be a mapping')
        self.jobs = queue.Queue()
        self.rows = {}
        self.manifest = {}
        for seed in self.seeds:
            for record in self.records:
                task = record['name']
                key = f'{seed}/{task}'
                rid = f'{self.run_id}_s{seed}_{task}'
                path = ROBO/'eval_result/RoboDojo'/task/self.policy_name/'arx_x5'/f'{seed}_dsr_full_{self.run_id}'/rid
                self.rows[key] = {'task': task, 'seed': seed, 'status': 'PENDING', 'episodes_completed': 0,
                                  'expected_episodes': record['episodes'], 'result_path': str(path/'_result.json'),
                                  'robodojo_run_id': rid, 'directory': str(self.out/'tasks'/f'seed_{seed}'/task)}
                if task in self.deferred_tasks:
                    self.rows[key].update(status='DEFERRED',
                        defer_reason='User requested evaluation of runnable tasks before repairing this task')
                elif not getattr(args, 'sequential_seeds', False):
                    self.jobs.put(key)

    def register(self, process, role):
        try:
            process._dsr_start_time = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        except FileNotFoundError:
            process._dsr_start_time = None
        with self.lock:
            self.processes[process.pid] = (process, role)

    def write_status(self):
        with self.lock:
            rows = list(self.rows.values())
            data = task_accuracy(rows, run=self.run_id, status=self.phase,
                updated_at=now(), validate_native=validate_result,
                validate_retry=single_retry_metrics, require_agent=self.policy_name == 'DSRRSI',
                required_layouts={(r['task'], r['seed']): self.required_layout_ids(r) for r in rows})
            self.manifest.update(status=self.phase, updated_at=now())
            # Reuse the operational lock, outside results. Never replace its
            # inode: main() holds flock on it for the entire evaluation.
            checkpoint = dict(manifest=self.manifest,
                              status=dict(status=self.phase, results=self.rows))
            with (self.state/'run.lock').open('r+' if (self.state/'run.lock').exists() else 'w+') as stream:
                stream.write(json.dumps(checkpoint, indent=2, sort_keys=True) + '\n')
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
            write_task_accuracy(self.out, data)
            return data

    def update(self, key, **fields):
        with self.lock:
            self.rows[key].update(fields)
            self.write_status()

    def restore_progress(self, prior):
        # Reuse native episode records, including partial tasks. The simulator
        # consumes its own resume manifest to skip the completed layouts.
        for key, old in (prior or {}).get('results', {}).items():
            if key not in self.rows:
                continue
            row = self.rows[key]
            if row['status'] == 'DEFERRED':
                row['previous_status'] = old.get('status')
                row['previous_error'] = old.get('error')
            try:
                metrics = validate_result(read_json(row['result_path']),
                                          row['expected_episodes'], partial=True,
                                          expected_layout_ids=self.required_layout_ids(row))
            except (ValueError, TypeError):
                continue
            if 0 < metrics['episodes_completed'] < row['expected_episodes']:
                source = Path(row['result_path'])
                resume = read_json(source.parent.parent/('_resume_'+row['robodojo_run_id']+'.json'))
                raw = read_json(source)
                if (not resume or resume.get('run_id') != row['robodojo_run_id']
                        or resume.get('details') != raw['details']):
                    raise RuntimeError('Resume manifest differs from completed episode records: '+key)
            row.update(metrics)
            if (row['status'] != 'DEFERRED' and old.get('status') == 'COMPLETED'
                    and old.get('exit_code') == 0
                    and metrics['episodes_completed'] == row['expected_episodes']):
                row.update(status='COMPLETED', exit_code=0)

    def required_layout_ids(self, row):
        # The official initial sample is the first native episode count from
        # SeedManager's numerically ordered layouts, including on resume.
        return (range(row['expected_episodes'])
                if getattr(self, 'strict_native_layouts', False) else None)

    def worker_environment(self, index, role):
        """One local runtime per GPU pair; model caches may be shared offline."""
        if role not in {'policy', 'sim'}:
            raise ValueError('Worker role must be policy or sim')
        gpus = self.args.policy_gpus if role == 'policy' else self.args.env_gpus
        env = base_env(gpus[index], role=role)
        if getattr(self.args, 'probe_bindings', None) or getattr(self, 'formal_adaptive', False):
            env['DSR_DISABLE_FAULT_RESTART'] = '1'
        runtime = self.state / 'runtime' / f'worker-{index}'
        runtime.mkdir(parents=True, exist_ok=True)
        env['DSR_LOCAL_STATE'] = str(runtime)
        for key, suffix in (
                ('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'cache'),
                ('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                ('XDG_STATE_HOME', 'state'),
                ('TORCH_EXTENSIONS_DIR', 'cache/torch_extensions'),
                ('CUDA_CACHE_PATH', 'cache/cuda')):
            path = runtime / suffix
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        # HF_HOME and explicitly configured model cache directories are kept
        # from process_environment: the pinned offline weights are read-only.
        kit = runtime / 'kit'
        if any(character.isspace() for character in str(kit)):
            raise ValueError('Kit portable root cannot contain whitespace')
        (kit / 'DerivedDataCache').mkdir(parents=True, exist_ok=True)
        options = shlex.split(env.get('DSR_KIT_ARGS', ''))
        isolated = ('--portable-root', '--/UJITSO/datastore/localCachePath')
        retained = []
        position = 0
        while position < len(options):
            option = options[position]
            if option in isolated:
                if position + 1 >= len(options):
                    raise ValueError('Kit cache option is missing its value: ' + option)
                position += 2
                continue
            if not any(option.startswith(flag + '=') for flag in isolated):
                retained.append(option)
            position += 1
        retained.extend(['--portable-root', str(kit),
                         '--/UJITSO/datastore/localCachePath=' + str(kit / 'DerivedDataCache')])
        env['DSR_KIT_ARGS'] = shlex.join(retained)
        if role == 'sim':
            scripting = runtime / 'kit-scripting-home'
            if scripting.is_symlink():
                raise ValueError('Kit scripting cache must stay on the worker local-state filesystem')
            scripting.mkdir(parents=True, exist_ok=True)
            env['DSR_KIT_SCRIPTING_CACHE_ROOT'] = str(scripting)
        return env

    def start_server(self, worker, task):
        """Wait for GPU capacity; a transient OOM must not cancel other workers."""
        if not isinstance(task, str) or not task:
            raise ValueError('A policy server requires an exact task binding')
        while not self.stop.is_set():
            gpu = self.args.policy_gpus[worker]
            memory = subprocess.check_output([
                'nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True)
            free_mib = int(memory.strip())
            if free_mib < 27000:
                self.stop.wait(60)
                continue
            try:
                self._start_server_once(worker, task)
                self.server_tasks[worker] = task
                return
            except RuntimeError:
                recent = server_output(self.servers.get(worker))
                if 'CUDA out of memory' not in recent:
                    raise
                if getattr(self.args, 'probe_bindings', None) or getattr(self, 'formal_adaptive', False):
                    raise RuntimeError('First-attempt policy startup failed; OOM restart disabled')
                proc = self.servers.get(worker)
                if proc is not None:
                    kill_group(proc)
                    with self.lock:
                        self.processes.pop(proc.pid, None)
                self.stop.wait(60)
        raise InterruptedError('Stopped while waiting for policy GPU')

    def _start_server_once(self, worker, task):
        """Start a policy server bound to the exact task it will serve.

        PhysicalRSI admission is keyed by the trusted deployment task name.
        A worker may process several tasks, so the caller must restart the
        server when that binding changes rather than reusing a stale task.
        """
        if not isinstance(task, str) or not task:
            raise ValueError('A policy server requires an exact task binding')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.ports[worker] = port
        cmd = [str(POLICY_PYTHON), '-u', str(ROOT/'vendor/XPolicyLab/setup_policy_server.py'),
               '--config_path', str(ROOT/'vendor/XPolicyLab/policy'/self.policy_name/'deploy.yml'), '--overrides', 'host=127.0.0.1', f'port={port}',
               'bench_name=RoboDojo', f'task_name={task}', f'ckpt_name={self.checkpoint.name}',
               'env_cfg_type=arx_x5', 'seed=0', f'policy_name={self.policy_name}', 'action_type=joint', 'action_dim=14',
               f'model_path={self.checkpoint}', f'norm_stats_path={self.checkpoint/"norm_stats.json"}', 'liger_kernel=False']
        candidate = self.server_candidates.get(worker)
        if candidate:
            cmd.append(f'experiment_candidate={candidate}')
        server_env = self.worker_environment(worker, 'policy')
        if self.policy_name == 'DSRRSI' and self.args.agent_strategy == 'physicalrsi':
            server_env.update(DSR_AGENT_STRATEGY='physicalrsi',
                DSR_CODEX_HOME=str(Path(self.args.codex_home).resolve()),
                PHYSICALRSI_SKILL_CONFIG=str(self.state/'skill_library.json'),
                PHYSICALRSI_EVAL_OUTPUT=str(self.state/'physicalrsi'/f'worker_{worker}'),
                DSR_AGENT_CALL_ARCHIVE=str(self.out/'agent_call_archive'/f'worker_{worker}'),
                MAX_JOBS='4')
            if self.args.codex_proxy:
                server_env['DSR_CODEX_PROXY'] = self.args.codex_proxy
        proc = subprocess.Popen(cmd, cwd=POLICY, env=server_env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        capture_server_output(proc)
        self.servers[worker] = proc
        self.register(proc, f'policy_{worker}')
        deadline = time.monotonic() + 1800
        while not self.stop.is_set():
            if proc.poll() is not None:
                raise RuntimeError(f'Policy {worker} exited {proc.returncode}: {server_output(proc)[-2000:]}')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    return
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Policy startup exceeded 30 minutes')
                self.stop.wait(2)
        raise InterruptedError('Stopped during policy startup')

    def run_task(self, worker, key):
        row = dict(self.rows[key])
        task, seed = row['task'], row['seed']
        candidate = self.probe_bindings.get(f'{seed}/{task}')
        directory = Path(row['directory'])
        directory.mkdir(parents=True, exist_ok=True)
        source = Path(row['result_path'])
        source.parent.mkdir(parents=True, exist_ok=True)
        link = directory/'episodes'
        if not link.exists():
            link.symlink_to(source.parent, target_is_directory=True)
        # A run-scoped tree allows the unmodified vendor summarizer to exclude old smoke runs.
        mirror = self.out/'official_results'/task/self.policy_name/'arx_x5'/f'{seed}_dsr_full_{self.run_id}'/row['robodojo_run_id']
        mirror.parent.mkdir(parents=True, exist_ok=True)
        if not mirror.exists():
            mirror.symlink_to(source.parent, target_is_directory=True)
        # The trusted router reads deployment.task_name. Never run a task
        # against a server initialized for another exact configuration.
        bound_task = self.server_tasks.get(worker)
        bound_candidate = self.server_candidates.get(worker)
        server = self.servers.get(worker)
        if (bound_task != task or bound_candidate != candidate
                or server is None or server.poll() is not None):
            old_server = self.servers.get(worker)
            if old_server is not None and old_server.poll() is None:
                kill_group(old_server)
                with self.lock:
                    self.processes.pop(old_server.pid, None)
            self.server_tasks.pop(worker, None)
            self.server_candidates[worker] = candidate
            self.start_server(worker, task)
        if (self.server_tasks.get(worker) != task
                or self.server_candidates.get(worker) != candidate):
            raise RuntimeError('Policy server task binding differs from simulator task')
        port = self.ports[worker]
        argv = ['python','-u','src/eval_client/main.py','--task_name',task,'--env_cfg_type','arx_x5',
                '--num_envs','1','--enable_cameras','--device_id','0','--policy_name',self.policy_name,
                '--port',str(port),'--host','127.0.0.1','--protocol','ws','--policy_server_url',f'ws://127.0.0.1:{port}',
                '--additional_info',f'dsr_full_{self.run_id}','--seed',str(seed),'--headless']
        env = self.worker_environment(worker, 'sim')
        cmd = worker_simulator_command(argv, env)
        env.update(EVAL_NUM=str(self.args.episodes or 'native'), ROBODOJO_RUN_ID=row['robodojo_run_id'])
        if self.strict_native_layouts:
            env['DSR_STRICT_NATIVE_LAYOUTS'] = '1'
        else:
            env.pop('DSR_STRICT_NATIVE_LAYOUTS', None)
        if getattr(self.args, 'layout_ids', None):
            env['DSR_EVAL_LAYOUT_IDS'] = ','.join(map(str, self.args.layout_ids))
        else:
            env.pop('DSR_EVAL_LAYOUT_IDS', None)
        if self.policy_name == 'DSRRSI':
            agent_output = directory/'agent'
            env.update(
                DSR_ROOT=str(ROOT), DSR_AGENT_OUTPUT=str(agent_output),
                DSR_CODEX_HOME=str(Path(self.args.codex_home).resolve()),
                DSR_AGENT_MODE=self.args.agent_mode,
                DSR_AGENT_STRATEGY=self.args.agent_strategy,
                DSR_EPISODE_RETRY='1' if getattr(self.args, 'episode_retry', False) else '0',
            )
            if self.args.codex_proxy:
                env['DSR_CODEX_PROXY'] = self.args.codex_proxy
            if self.args.memory_root:
                memory = Path(self.args.memory_root)/task/'agent/experience_ledger.json'
                if not memory.is_file():
                    raise FileNotFoundError(memory)
                env['DSR_MEMORY'] = str(memory.resolve())
        self.update(key, status='RUNNING', worker=worker, policy_gpu=self.args.policy_gpus[worker],
                    environment_gpu=self.args.env_gpus[worker], started_at=now())
        last_error = 'No complete result'
        for attempt in range(1, self.args.attempts+1):
            if self.stop.is_set():
                self.update(key, status='INTERRUPTED')
                return
            start = time.monotonic()
            capture_diagnostics = getattr(self, 'formal_adaptive', False)
            proc = subprocess.Popen(cmd, cwd=ROBO, env=env,
                                    stdout=subprocess.PIPE if capture_diagnostics else subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
            if capture_diagnostics:
                capture_server_output(proc)
            self.register(proc, key)
            self.update(key, pid=proc.pid, attempt=attempt)
            timeout = False
            timeout_reason = None
            signature = progress_signature(source, directory)
            last_progress = time.monotonic()
            drain_active = getattr(self.args, 'drain_active_episode', False)
            try:
                while proc.poll() is None and (drain_active or not self.stop.is_set()):
                    if self.servers[worker].poll() is not None:
                        if not drain_active:
                            raise RuntimeError(f'Policy server {worker} died during {key}')
                        # Transport failure is reported by the simulator, which
                        # must retain the opportunity to flush its evidence.
                        self.update(key, monitor_error=f'Policy server {worker} died during {key}')
                    elapsed = time.monotonic()-start
                    if elapsed > self.args.task_timeout and not timeout:
                        timeout = True
                        timeout_reason = 'task wall-clock timeout'
                        if not drain_active:
                            break
                        self.update(key, status='DRAINING_ACTIVE_EPISODE', drain_warning=timeout_reason)
                    current_signature = progress_signature(source, directory)
                    if current_signature != signature:
                        signature, last_progress = current_signature, time.monotonic()
                    idle_seconds = time.monotonic() - last_progress
                    if idle_seconds > self.args.stall_timeout and not timeout_reason:
                        timeout_reason = f'no execution progress for {int(idle_seconds)} seconds'
                        self.update(key, status='DRAINING_ACTIVE_EPISODE' if drain_active else 'STALLED',
                                    drain_warning=timeout_reason)
                        if not drain_active:
                            timeout = True
                            break
                    self.update(key, progress_idle_seconds=round(idle_seconds, 1),
                                execution_progress=read_json(directory/'agent/progress.json'))
                    partial = read_json(source)
                    if partial:
                        try:
                            metrics = validate_result(partial, row['expected_episodes'], partial=True,
                                                      expected_layout_ids=self.required_layout_ids(row))
                            write_json(directory/'raw_result.json', partial)
                            self.update(key, **metrics)
                        except ValueError:
                            pass  # vendor writes non-atomically; read again on the next heartbeat
                    try:
                        check_run_storage(self.out, self.state,
                            output_gib=getattr(self.args, 'min_output_free_gib', 10),
                            state_gib=getattr(self.args, 'min_state_free_gib', 1))
                    except (OSError, RuntimeError, ValueError) as storage_error:
                        if not drain_active:
                            raise
                        # Stop new layouts on resource trouble, finish this one.
                        self.stop.set()
                        self.update(key, storage_monitor_error=str(storage_error))
                    if drain_active:
                        # Waiting on an already-set stop Event would busy-spin.
                        time.sleep(15)
                    else:
                        self.stop.wait(15)
            finally:
                if drain_active:
                    # A monitor exception also drains the one owned layout.
                    # Cleanup only after its native process has exited.
                    while proc.poll() is None:
                        try:
                            proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            pass
                kill_group(proc)
                if capture_diagnostics and (proc.returncode != 0 or timeout):
                    self.update(key, simulator_output_tail=server_output(proc)[-8192:])
                with self.lock:
                    self.processes.pop(proc.pid, None)
            if self.stop.is_set() and not drain_active:
                self.update(key, status='INTERRUPTED', exit_code=proc.returncode)
                return
            data = read_json(source)
            try:
                metrics = validate_result(data, row['expected_episodes'],
                                          expected_layout_ids=self.required_layout_ids(row))
                if proc.returncode != 0 or (timeout and not drain_active):
                    raise ValueError(f'Client did not finish cleanly: exit={proc.returncode}, timeout={timeout_reason}')
                if self.policy_name == 'DSRRSI':
                    events_path = agent_output/'episode_events.jsonl'
                    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
                    by_layout = {int(event['layout_id']): event for event in events}
                    if (candidate or getattr(self, 'formal_adaptive', False)) and len(by_layout) != len(events):
                        raise ValueError('First-attempt run has duplicate layouts; restarts cannot be counted')
                    for detail in data['details'].values():
                        event = by_layout.get(detail['layout_id'])
                        if (event is None or event['first_success'] != detail['success']
                                or not math.isclose(event['first_score'], detail['score'], abs_tol=1e-8)):
                            raise ValueError('Native score is missing or inconsistent with agent episode trace')
                        if candidate:
                            validate_probe_episode(event, task, seed, candidate,
                                read_json(self.state/'skill_library.json'), agent_output)
                        elif getattr(self, 'formal_adaptive', False):
                            validate_formal_episode(event, task, seed,
                                read_json(self.state/'skill_library.json'), agent_output, detail)
                write_json(directory/'raw_result.json', data)
                self.update(key, status='COMPLETED', exit_code=0, finished_at=now(),
                            drain_warning=timeout_reason, **metrics)
                return
            except (ValueError, TypeError, KeyError, OSError) as exc:
                last_error = timeout_reason or str(exc)
                self.update(key, status='RETRYING', error=last_error, exit_code=proc.returncode)
                # Drop any pending request left by a killed simulator. A fresh
                # policy server reloads the same frozen weights for the retry.
                if attempt < self.args.attempts and not self.stop.is_set():
                    old_server = self.servers[worker]
                    kill_group(old_server)
                    with self.lock:
                        self.processes.pop(old_server.pid, None)
                    self.server_candidates[worker] = candidate
                    self.start_server(worker, task)
                    new_port = self.ports[worker]
                    # Rebuild only the connection flags; retain task/layout IDs.
                    for flag, value in (('--port', str(new_port)),
                                        ('--policy_server_url', f'ws://127.0.0.1:{new_port}')):
                        argv[argv.index(flag)+1] = value
                    cmd = worker_simulator_command(argv, env)
                    self.update(key, status='RUNNING')
        self.update(key, status='FAILED', error=last_error, finished_at=now())

    def worker(self, index):
        try:
            while not self.stop.is_set():
                try:
                    key = self.jobs.get_nowait()
                except queue.Empty:
                    return
                try:
                    if self.rows[key]['status'] != 'COMPLETED':
                        self.run_task(index, key)
                finally:
                    self.jobs.task_done()
        except BaseException:
            self.stop.set()
            raise

    def run_workers(self):
        with ThreadPoolExecutor(max_workers=len(self.args.policy_gpus)) as pool:
            futures = [pool.submit(self.worker, i) for i in range(len(self.args.policy_gpus))]
            try:
                while not all(f.done() for f in futures):
                    self.write_status()
                    wait_futures(futures, timeout=10)
                for future in futures:
                    future.result()
            except BaseException:
                self.stop.set()
                raise

    def run_scheduled_workers(self):
        if not getattr(self.args, 'sequential_seeds', False):
            self.run_workers()
            return
        # A failed or deferred task is not a completed seed. Revisit only its
        # unfinished layouts before permitting the next seed to start.
        for seed in self.seeds:
            self.current_seed, self.seed_round = seed, 0
            while not self.stop.is_set():
                pending = [key for key, row in self.rows.items()
                           if row['seed'] == seed and row['status'] != 'COMPLETED']
                if not pending:
                    break
                for key in pending:
                    if self.rows[key]['status'] != 'DEFERRED':
                        self.jobs.put(key)
                if not self.jobs.empty():
                    self.seed_round += 1
                    self.phase = 'RUNNING'
                    self.write_status()
                    self.run_workers()
                if self.stop.is_set():
                    return
                remaining = [key for key in pending if self.rows[key]['status'] != 'COMPLETED']
                if remaining:
                    self.phase = 'WAITING_FOR_SEED'
                    self.manifest['seed_barrier'] = {'seed': seed, 'unfinished_tasks': remaining,
                        'retry_after_seconds': self.args.seed_retry_delay}
                    self.write_status()
                    self.stop.wait(self.args.seed_retry_delay)
            if self.stop.is_set():
                return
            self.manifest.pop('seed_barrier', None)
        self.current_seed = None

    def run(self):
        if self.args.codex_home:
            os.environ["DSR_CODEX_HOME"] = str(Path(self.args.codex_home).expanduser().resolve())
        preflight(agent=self.policy_name == "DSRRSI")
        verify_state_lock(self.state)
        check_run_storage(self.out, self.state,
            output_gib=getattr(self.args, 'min_output_free_gib', 10),
            state_gib=getattr(self.args, 'min_state_free_gib', 1))
        (self.state / 'physicalrsi').mkdir(exist_ok=True)
        saved = read_evaluation_checkpoint(self.out, self.state.parent)
        prior, prior_manifest = saved['status'], saved['manifest']
        if (self.out/'task_accuracy.json').exists() and not prior_manifest:
            raise RuntimeError('Existing results require their original operational checkpoint for verified resume')
        if prior_manifest and not self.args.resume:
            raise RuntimeError('Run already exists. Use --resume to validate and continue it.')
        budget = int(getattr(self.args, 'run_timeout', 14400))
        if budget < 1:
            raise ValueError('run_timeout must be positive')
        budget_timer = threading.Timer(budget, self.stop.set)
        budget_timer.daemon = True
        budget_timer.start()
        probe_bindings_sha256 = None
        readiness_sha256 = None
        source_ref = None
        readiness = None
        self.formal_adaptive = False
        if self.policy_name == 'DSRRSI' and self.args.agent_strategy == 'physicalrsi':
            from configure_physicalrsi_dm05 import configure
            library = self.state/'skill_library.json'
            if self.args.resume and not library.exists() and (self.out/'skill_library.json').is_file():
                library.write_bytes((self.out/'skill_library.json').read_bytes())
            source_ref = prepare_skill_library(library, getattr(self.args, 'skill_library', None))
            if not library.exists():
                configure(library,self.out/'code_skills',self.checkpoint)
            config = read_json(library)
            if not config.get('program_registry'):
                raise RuntimeError('Strict PhysicalRSI evaluation needs the official executable operation library')
            probe_bindings_sha256 = validate_probe_run(config,
                getattr(self.args, 'probe_bindings', None), library, self.rows, self.args)
            from dsr.robodojo.formal_evidence import formal_strategy_library, validate_formal_run
            self.formal_adaptive = formal_strategy_library(config)
            validate_formal_run(config, self.rows, self.args)
            if getattr(self.args, 'allow_unverified_low_sr_measurement', False):
                from dsr.robodojo.low_sr_readiness import audit
                readiness = audit([library])
                existing_review = (prior_manifest or {}).get('low_sr_readiness')
                if existing_review is not None and existing_review != readiness:
                    raise ValueError('Run capability audit differs from the recorded measurement review')
                payload = json.dumps(readiness, indent=2, sort_keys=True) + '\n'
                readiness_sha256 = hashlib.sha256(payload.encode()).hexdigest()
        memory_digest = None
        if self.args.memory_root:
            memory_root = Path(self.args.memory_root).resolve()
            source_manifest = read_evaluation_checkpoint(
                memory_root.parents[1], self.state.parent)['manifest']
            if not source_manifest and (memory_root.parents[1]/'task_accuracy.json').exists():
                raise ValueError('Memory source requires its original operational checkpoint')
            if source_manifest and self.args.agent_mode == 'frozen':
                if set(source_manifest.get('seeds', [])) & set(self.seeds):
                    raise ValueError('Frozen evaluation seeds overlap the experience training seeds')
            digest = hashlib.sha256()
            for record in self.records:
                path = memory_root/record['name']/'agent/experience_ledger.json'
                digest.update(record['name'].encode())
                digest.update(path.read_bytes())
            memory_digest = digest.hexdigest()
        verification = (verify('sim') if self.checkpoint == CHECKPOINT.resolve()
                        else {**validate_training(self.checkpoint, 1), 'status': 'VERIFIED',
                              'sha256': hashlib.sha256((self.checkpoint/'training_config.json').read_bytes()).hexdigest()})
        if verification['status'] != 'VERIFIED':
            raise RuntimeError('Simulation checkpoint has not passed SHA-256 verification')
        for seed in self.seeds:
            layouts = ROBO/'Assets/Eval_Layout/RoboDojo/arx_x5'/str(seed)
            for record in self.records:
                if len(list(layouts.glob(record['name']+'_[0-9]*.json'))) < record['episodes']:
                    raise RuntimeError(f'Missing layouts for seed {seed}: {record["name"]}')
        if prior_manifest:
            if (prior_manifest.get('seeds') != self.seeds
                    or prior_manifest.get('checkpoint_sha256') != verification['sha256']
                    or prior_manifest.get('policy_name', 'OpenDM') != self.policy_name
                    or prior_manifest.get('agent_mode', 'frozen') != self.args.agent_mode
                    or prior_manifest.get('agent_strategy', 'failure_adapt') != self.args.agent_strategy
                    or prior_manifest.get('episode_retry', False) != getattr(self.args, 'episode_retry', False)
                    or prior_manifest.get('memory_sha256') != memory_digest
                    or prior_manifest.get('skill_library_source') != getattr(self.args, 'skill_library', None)
                    or prior_manifest.get('probe_bindings_sha256') != probe_bindings_sha256
                    or prior_manifest.get('allow_unverified_low_sr_measurement', False) != getattr(self.args, 'allow_unverified_low_sr_measurement', False)
                    or prior_manifest.get('low_sr_readiness_sha256') != readiness_sha256
                    or (self.policy_name == 'DSRRSI' and self.args.agent_strategy == 'physicalrsi'
                        and prior_manifest.get('skill_library_sha256') != hashlib.sha256(library.read_bytes()).hexdigest())
                    or prior_manifest.get('episode_limit') != self.args.episodes
                    or prior_manifest.get('layout_ids') != getattr(self.args, 'layout_ids', None)
                    or prior_manifest.get('drain_active_episode', False) != getattr(self.args, 'drain_active_episode', False)
                    or prior_manifest.get('task_names', [r['name'] for r in self.records]) != [r['name'] for r in self.records]):
                raise RuntimeError('Cannot resume with a different protocol/checkpoint')
            self.restore_progress(prior)
        self.manifest = {'run_id':self.run_id, 'started_at':(prior_manifest or {}).get('started_at', now()), 'seeds':self.seeds,
                         'checkpoint':str(self.checkpoint),'checkpoint_sha256':verification['sha256'],
                         'policy_name':self.policy_name, 'agent_mode':self.args.agent_mode,
                         'agent_strategy':self.args.agent_strategy,
                         'episode_retry':getattr(self.args, 'episode_retry', False),
                         'native_fault_restarts_disabled':bool(getattr(self.args, 'probe_bindings', None) or self.formal_adaptive),
                         'formal_adaptive_strategy':self.formal_adaptive,
                         'drain_active_episode':getattr(self.args, 'drain_active_episode', False),
                         'memory_sha256':memory_digest,
                         'skill_library_source':getattr(self.args, 'skill_library', None),
                         'probe_bindings':getattr(self.args, 'probe_bindings', None),
                         'probe_bindings_sha256':probe_bindings_sha256,
                         'allow_unverified_low_sr_measurement':getattr(self.args, 'allow_unverified_low_sr_measurement', False),
                         'low_sr_readiness_sha256':readiness_sha256,
                         'measurement_authorization':({
                             'mode':'explicit_unverified_low_sr_measurement',
                             'rationale':'User-authorized isolated measurement of low-success candidates; failed capability audit is retained and promotion remains disabled.',
                             'capability_gate_overridden':True,
                         } if readiness_sha256 else None),
                         'candidate_qualification':'UNVERIFIED_MEASUREMENT' if readiness_sha256 else None,
                         'task_names':[r['name'] for r in self.records], 'episode_limit':self.args.episodes,
                         'layout_ids':getattr(self.args, 'layout_ids', None),
                         'strict_native_layouts':self.strict_native_layouts,
                         'native_layout_coverage':('exact_initial_range_no_replacement'
                                                  if self.strict_native_layouts else None),
                         'task_configurations':len(self.records),
                         'canonical_tasks':len(vendor_module('summarize_result').ALL_TASKS),
                         'policy_gpus':self.args.policy_gpus,'environment_gpus':self.args.env_gpus,
                         'scope':('FULL_NATIVE_ROBODOJO_THREE_SEED' if self.seeds == [0,1,2]
                                  and not self.args.tasks and not self.args.episodes else
                                  'FULL_TASK_ROBODOJO_SELECTED_SEEDS' if not self.args.tasks and not self.args.episodes
                                  else 'NATIVE_ROBODOJO_SUBSET'),
                         'eval_num':'native', 'coding_agent_config':str(AGENT_CONFIG),
                         'coding_agent_role':'launcher audit, separate CLI transcript',
                         'coding_agent_audit':self.args.agent_audit, 'robot_supervisor_enabled':False,
                         'adaptation_enabled':False, 'scope_note':'Real DM05 executor benchmark; no mock or RSI adaptation.',
                         'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.manifest.update(task_timeout_seconds=self.args.task_timeout,
                             checkpoint_verification=verification,
                             skill_library_source_receipt=source_ref,
                             low_sr_readiness=readiness,
                             state_root=str(self.state),
                             worker_runtime_root=str(self.state / 'runtime'),
                             worker_runtime_directory_pattern='worker-{index}',
                             output_root=str(self.out),
                             run_timeout_seconds=budget,
                             stall_timeout_seconds=self.args.stall_timeout,
                             attempts_per_task=self.args.attempts,
                             seed_scheduling='sequential' if self.args.sequential_seeds else 'shared_queue',
                             seed_retry_delay_seconds=self.args.seed_retry_delay,
                             deferred_tasks=sorted(self.deferred_tasks),
                             resumed_at=now() if prior_manifest else None)
        if self.policy_name == 'DSRRSI':
            self.manifest.update(
                coding_agent_config=str(Path(self.args.codex_home)/'config.toml'),
                coding_agent_role='robot supervisor: failure diagnosis and bounded repair',
                robot_supervisor_enabled=True, adaptation_enabled=self.args.agent_mode == 'adapt',
                scope_note='DM0.5 + System2; first-pass scores; paired repair rollouts logged separately',
                memory_root=self.args.memory_root,
            )
            if self.args.agent_strategy == 'physicalrsi':
                sources = ['src/dsr/system1/physicalrsi_runtime.py','src/dsr/system2/physicalrsi_codex.py',
                           'src/dsr/robodojo/physicalrsi_eval.py','tools/configure_physicalrsi_dm05.py',
                           'vendor/RoboDojo/src/eval_client/main.py',
                           'vendor/PhysicalRSI/PhysicalRSI_baselines/robodojo/skill_model.py',
                           'vendor/PhysicalRSI/PhysicalRSI_baselines/robodojo/agent_planner.py',
                           'vendor/PhysicalRSI/PhysicalRSI_baselines/robodojo/evaluation.py']
                if getattr(self.args, 'probe_bindings', None):
                    sources += ['src/dsr/robodojo/task_policy_probe.py',
                                'src/dsr/robodojo/task_policy_routing.py',
                                'src/dsr/robodojo/stage_evidence.py']
                self.manifest.update(
                    coding_agent_role='episode-start memory-conditioned skill/operation selection',
                    adaptation_enabled=False,agent_calls_per_episode=1,
                    scope_note='Pinned PhysicalRSI frozen skill runtime; DM0.5 VLA and Codex substitutions; no failure-triggered repairs',
                    skill_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    registered_operations=len(config['program_registry']),
                    protocol_source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                            for name in sources},
                    upstream_commit=config['model_substitution']['upstream_commit'])
                if getattr(self.args, 'probe_bindings', None):
                    self.manifest.update(agent_calls_per_episode=None,
                        max_agent_plans_per_episode=2, max_pre_action_selection_corrections=1,
                        scope_note='Isolated frozen-per-episode portfolio development measurement; '
                                   'staged configurations require an agent-selected stage operation; '
                                   'one pre-action correction allowed, one native attempt, no failure restart.')
                if getattr(self.args, 'episode_retry', False):
                    sources += ['src/dsr/system2/episode_retry.py', 'src/dsr/robodojo/episode_retry.py']
                    self.manifest.update(
                        coding_agent_role='episode-start selection; one full failure-conditioned replan',
                        agent_calls_per_episode=None, max_agent_plans_per_episode=2,
                        max_failure_restarts=1,
                        retry_protocol='episode_start_single_retry_same_layout/v1',
                        scope_note='Frozen registered executors; unchanged first pass; one same-layout failure restart',
                        protocol_source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                                for name in sources})
        self.phase = 'RUNNING'
        self.write_status()
        def request_stop(signum, frame):
            self.stop.set()
        signal.signal(signal.SIGTERM,request_stop)
        signal.signal(signal.SIGINT,request_stop)
        error = None
        try:
            self.run_scheduled_workers()
        except BaseException as exc:
            self.stop.set()
            error = str(exc)
            traceback.print_exc()
        finally:
            self.stop.set()
            budget_timer.cancel()
            for proc, role in list(self.processes.values()):
                kill_group(proc)
            # A global resource failure can bypass run_task's ordinary stop
            # branch. Never leave dead clients labelled RUNNING in the report.
            for row in self.rows.values():
                if row['status'] in {'RUNNING', 'RETRYING', 'STALLED'}:
                    row.update(status='INTERRUPTED', finished_at=now(),
                               error=error or 'Run stopped before task completion')
        complete = all(r['status']=='COMPLETED' for r in self.rows.values())
        self.phase = 'COMPLETED' if complete else 'INCOMPLETE'
        if error:
            self.manifest['error'] = error
        result_status = self.write_status()
        env = os.environ.copy()
        env['ROBODOJO_EVAL_ROOT'] = str(self.out/'official_results')
        if (self.out/'official_results').exists():
            result = subprocess.run([sys.executable,str(ROBO/'scripts/internal/summarize_result.py')],
                                    env=env,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,timeout=600)
            self.manifest['official_summary_exit_code'] = result.returncode
            self.write_status()
        return 0 if result_status['status'] == 'COMPLETED' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_full'))
    parser.add_argument('--output-root', default=str(local_state_root()/'experiments/full_robodojo'))
    parser.add_argument('--state-root', help='Lock-capable local directory for active memory and run locks; evidence may use a separate shared output-root')
    parser.add_argument('--min-output-free-gib', type=float, default=10)
    parser.add_argument('--min-state-free-gib', type=float, default=1)
    parser.add_argument('--policy-gpus', required=True)
    parser.add_argument('--env-gpus', required=True)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sequential-seeds', action='store_true',
                        help='Finish every task in one seed before starting the next; resume failed tasks in place')
    parser.add_argument('--seed-retry-delay', type=int, default=300,
                        help='Wait between passes over unfinished tasks in the current seed')
    parser.add_argument('--defer-tasks', default='',
                        help='Comma-separated tasks to leave pending repair; retained in full benchmark coverage')
    parser.add_argument('--attempts', type=int, default=2)
    parser.add_argument('--task-timeout', type=int, default=86400)
    parser.add_argument('--run-timeout', type=int, default=14400,
                        help='Bound total execution to this many seconds, preserving incomplete results for resume')
    parser.add_argument('--stall-timeout', type=int, default=5400,
                        help='Record a no-progress warning after this many seconds; isolated drains may continue')
    parser.add_argument('--drain-active-episode', action='store_true',
                        help='For single-layout probes: stop/deadline/stall never kills the active native episode')
    parser.add_argument('--agent-audit', default=None)
    parser.add_argument('--policy-name', choices=['OpenDM', 'DSRRSI'], default='OpenDM')
    parser.add_argument('--checkpoint', default=str(CHECKPOINT))
    parser.add_argument('--agent-mode', choices=['adapt', 'frozen'], default='frozen')
    parser.add_argument('--agent-strategy',
                        choices=['failure_adapt', 'physicalrsi'], default='physicalrsi')
    parser.add_argument('--episode-retry', action='store_true',
                        help='One failure-only same-layout restart through the full episode-start Agent path')
    parser.add_argument('--codex-home', default=os.environ.get('DSR_CODEX_HOME'))
    parser.add_argument('--codex-proxy', default=os.environ.get('DSR_CODEX_PROXY'))
    parser.add_argument('--memory-root')
    parser.add_argument('--skill-library', help='Frozen skill library copied into this isolated experiment')
    parser.add_argument('--probe-bindings', help='Versioned seed/task to candidate bindings for an isolated probe')
    parser.add_argument('--allow-unverified-low-sr-measurement', action='store_true',
                        help='Explicitly measure isolated low-SR candidates despite the failed capability audit; records every blocker and never promotes candidates')
    parser.add_argument('--tasks', help='Comma-separated subset; omitted means the full 54 configurations')
    parser.add_argument('--episodes', type=int, help='Explicit validation cap, never used in a full run')
    parser.add_argument('--layout-ids', help='Explicit native layout IDs for a preregistered subset; requires --episodes to match')
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.min_output_free_gib, args.min_state_free_gib)):
        parser.error('Storage reserves must be positive finite GiB values')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):
        parser.error('run-id must be a simple directory name')
    for name in ('policy_gpus','env_gpus','seeds'):
        setattr(args,name,[int(x) for x in getattr(args,name).split(',')])
    if (len(args.policy_gpus) != len(args.env_gpus) or not args.policy_gpus
            or len(set(args.policy_gpus+args.env_gpus)) != 2*len(args.policy_gpus)):
        parser.error('Each worker needs one disjoint policy/simulator GPU pair')
    # Worker count controls concurrency, never benchmark coverage. A complete
    # run still contains all native tasks/layouts even on fewer GPU pairs.
    if args.episodes is not None and args.episodes < 1:
        parser.error('episodes must be positive')
    if args.layout_ids is not None:
        try:
            args.layout_ids = [int(value) for value in args.layout_ids.split(',')]
        except ValueError:
            parser.error('layout-ids must be comma-separated integers')
        if (not args.layout_ids or min(args.layout_ids) < 0
                or len(set(args.layout_ids)) != len(args.layout_ids)
                or args.episodes != len(args.layout_ids)):
            parser.error('layout-ids must be unique, nonnegative and match episodes')
    if args.episode_retry and (args.policy_name != 'DSRRSI' or args.agent_strategy != 'physicalrsi'):
        parser.error('--episode-retry requires DSRRSI and the episode-start strategy')
    if args.probe_bindings and os.environ.get('OMNI_KIT_ACCEPT_EULA', '').lower() not in {'yes', 'y', '1'}:
        parser.error('Native probe needs the already authorized host EULA environment; source deployment/l20/env.l20.sh before launch')
    if args.allow_unverified_low_sr_measurement and not args.probe_bindings:
        parser.error('Unverified measurement mode requires versioned isolated probe bindings')
    if args.drain_active_episode and (not args.probe_bindings or args.episodes != 1
            or not args.layout_ids or args.attempts != 1 or args.episode_retry):
        parser.error('Episode draining requires a single-layout, single-attempt isolated probe without retry')
    if args.policy_name == 'DSRRSI' and args.agent_strategy == 'physicalrsi' and args.agent_mode != 'frozen':
        parser.error('PhysicalRSI formal evaluation uses a frozen skill library; System2 evolution is separate')
    if args.policy_name == 'DSRRSI' and args.agent_strategy == 'failure_adapt' and args.agent_mode == 'frozen' and not args.memory_root:
        parser.error('Frozen DSRRSI evaluation requires --memory-root from an adapted training run')
    if len(set(args.seeds)) != len(args.seeds) or any(s not in (0,1,2) for s in args.seeds):
        parser.error('seeds must be unique values from 0,1,2')
    if (not args.seeds or args.attempts < 1 or args.task_timeout < 1 or args.run_timeout < 1
            or args.stall_timeout < 1 or args.seed_retry_delay < 1):
        parser.error('At least one seed and positive attempts/timeout are required')
    out = Path(args.output_root)/args.run_id
    out.mkdir(parents=True,exist_ok=True)
    state = evaluation_state(args)
    verify_state_lock(state)
    # Native crashes can otherwise create tens of GiB of core dumps. Preserve
    # exception/log evidence and disable only this launcher's descendant cores.
    import resource
    _, core_hard = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(resource.RLIMIT_CORE, (0, core_hard))
    with (state/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return FullRun(args).run()


if __name__ == '__main__':
    raise SystemExit(main())
