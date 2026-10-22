"""The existing v6 task_accuracy.json contract, backed by native episodes.

This module writes one result file. Runtime checkpoints and diagnostic logs do
not belong to the public evaluation directory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def read_evaluation_checkpoint(root, state_root=None):
    """Read operational metadata without requiring public result sidecars."""
    from dsr.runtime_config import local_state_root
    root = Path(root).resolve()
    state_root = (Path(state_root) if state_root is not None
                  else local_state_root()/'evaluation-state')
    path = state_root/root.name/'run.lock'
    checkpoint = load_json(path)
    # main() opens an empty lock before a new run has its first checkpoint.
    if path.is_file() and path.stat().st_size:
        manifest = checkpoint.get('manifest') if isinstance(checkpoint, dict) else None
        if (not isinstance(manifest, dict) or manifest.get('run_id') != root.name
                or not isinstance(manifest.get('output_root'), str)
                or Path(manifest['output_root']).resolve() != root
                or not isinstance(checkpoint.get('status'), dict)):
            raise ValueError('Operational checkpoint does not match this evaluation output')
        return checkpoint
    return dict(manifest=load_json(root/'manifest.json'),
                status=load_json(root/'status.json'))


def task_accuracy(rows, *, run, status, updated_at, validate_native,
                  validate_retry, require_agent=False, required_layouts=None):
    """Keep the v6 keys/order/units; never derive scores from success rates.

    Retry counters are null when historical telemetry is unknown. Invalid or
    missing evidence is listed in coverage_issues and cannot qualify a task as
    completed. Provisional totals count only verified native first attempts.
    """
    completed, issues = [], []
    totals = dict(episodes=0, first_successes=0, final_successes=0,
                  first_success_rate=None, final_success_rate=None,
                  recovered_failures=0, retry_eligible=0, retry_requests=0,
                  retry_planner_starts=0, retry_attempts=0, completed_retries=0)
    counters = ('retry_requests', 'retry_planner_starts', 'retry_attempts',
                'completed_retries')
    expected = sum(row['expected_episodes'] for row in rows)
    for row in rows:
        task, seed = row['task'], row['seed']
        directory = Path(row['directory'])
        source = Path(row['result_path'])
        # Native writes may be in progress. Never substitute stale mirrored
        # data for an existing but unreadable native result.
        if not source.exists():
            source = directory/'raw_result.json'
        if not source.exists():
            continue
        native = load_json(source)
        issue_start = len(issues)

        def issue(message):
            issues.append(dict(task=task, seed=seed, issue=message))

        try:
            validate_native(native, row['expected_episodes'], partial=True,
                            expected_layout_ids=(required_layouts or {}).get((task, seed)))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            issue('Invalid native episodes: ' + str(exc))
            continue
        layouts = {detail['layout_id']: detail for detail in native['details'].values()}
        events = {}
        try:
            with (directory/'agent/episode_events.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue  # An unfinished append is not a completed episode.
                    if (isinstance(event, dict) and event.get('task') == task
                            and type(event.get('seed')) is int and event['seed'] == seed
                            and type(event.get('layout_id')) is int):
                        events[event['layout_id']] = event
        except FileNotFoundError:
            pass
        first = final = recovered = 0
        before, after = [], []
        for layout, detail in layouts.items():
            first_success, first_score = detail['success'], detail['score']
            final_success, final_score = first_success, first_score
            event = events.get(layout)
            if event is not None and (
                    event.get('first_success') is not first_success
                    or type(event.get('first_score')) not in (int, float)
                    or not math.isclose(event['first_score'], first_score, abs_tol=1e-8)):
                issue(f'Agent/native first-attempt mismatch at layout {layout}')
                event = None
            if event is None and require_agent:
                issue(f'Missing matching Agent event at layout {layout}')
                for name in counters:
                    totals[name] = None
            if event is not None and event.get('protocol') in {
                    'single_retry_same_layout/v1', 'episode_start_single_retry_same_layout/v1'}:
                metrics = validate_retry([event], protocol=event['protocol'])
                if metrics['invalid_records'] or metrics['episodes'] != 1:
                    issue(f'Invalid retry telemetry at layout {layout}')
                    for name in counters:
                        totals[name] = None
                else:
                    for name in counters:
                        value = metrics[name]
                        totals[name] = (totals[name] + value
                                        if totals[name] is not None and value is not None else None)
                    if metrics['completed_retries']:
                        # Use the actual completed retry score, including a
                        # failed retry whose partial score decreased.
                        path = Path(event.get('report') or '').resolve()
                        report = (load_json(path) if path.is_relative_to(
                            (directory/'agent/episodes').resolve()) else None)
                        retry = report.get('retry_attempt') if isinstance(report, dict) else None
                        if retry is None and event['protocol'] == 'single_retry_same_layout/v1':
                            attempts = report.get('attempts', []) if isinstance(report, dict) else []
                            if isinstance(attempts, list) and len(attempts) == 2:
                                retry = attempts[1]
                        if (not isinstance(retry, dict) or retry.get('task') != task
                                or retry.get('seed') != seed or retry.get('layout_id') != layout
                                or retry.get('status') != 'COMPLETED'
                                or retry.get('success') is not event['retry_success']
                                or type(retry.get('score')) not in (int, float)
                                or not math.isfinite(retry['score']) or not 0 <= retry['score'] <= 1):
                            issue(f'Missing or inconsistent retry report at layout {layout}')
                        else:
                            final_success, final_score = retry['success'], retry['score']
            elif event is not None and (event.get('retry_attempted')
                    or event.get('final_success', first_success) is not first_success):
                issue(f'Unrecognized retry protocol at layout {layout}')
            first += int(first_success)
            final += int(final_success)
            recovered += int(not first_success and final_success)
            before.append(first_score)
            after.append(final_score)
        count = len(layouts)
        totals['episodes'] += count
        totals['first_successes'] += first
        totals['final_successes'] += final
        totals['recovered_failures'] += recovered
        totals['retry_eligible'] += count - first
        if (row['status'] == 'COMPLETED' and count == row['expected_episodes']
                and count and len(issues) == issue_start):
            completed.append(dict(task=task, seed=seed, episodes=count,
                first_success_rate=first/count, final_success_rate=final/count,
                recovered_failures=recovered, score_before=100*sum(before)/count,
                score_after=100*sum(after)/count))
    count = totals['episodes']
    if count:
        totals['first_success_rate'] = totals['first_successes']/count
        totals['final_success_rate'] = totals['final_successes']/count
    if status == 'COMPLETED' and (count != expected or len(completed) != len(rows) or issues):
        status = 'INCOMPLETE'
    return dict(updated_at=updated_at, run=run, status=status,
                expected_episodes=expected, episodes_completed=count,
                completed=completed, coverage_issues=issues,
                episode_weighted_provisional=totals)


def write_task_accuracy(root, data):
    """Atomic two-space JSON with insertion order and no permanent sidecars."""
    path = Path(root)/'task_accuracy.json'
    temporary = path.with_suffix('.json.tmp')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                        allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
