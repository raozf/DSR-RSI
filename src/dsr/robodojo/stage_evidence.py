"""Episode-owned, durable evidence for a measured Code + Code boundary."""
import hashlib
import json
import math
from pathlib import Path


def _save_once(path, payload):
    try:
        with path.open('xb') as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError('Recorded stage evidence changed: ' + str(path))


def _controller_pair(controllers):
    if (not isinstance(controllers, list) or len(controllers) != 2
            or any(not isinstance(c, str) or not c for c in controllers)
            or controllers[0] == controllers[1]):
        raise ValueError('Stage evidence requires two distinct registered controllers')


def _open_grippers(value):
    if not isinstance(value, dict) or set(value) != {'left', 'right'}:
        return False
    return all(isinstance(v, list) and len(v) == 1 and type(v[0]) in (int, float)
               and math.isfinite(v[0]) and .95 <= v[0] <= 1 for v in value.values())


def inspect_trace(rows, controllers, *, successful, native_steps):
    """A command proposal is not an accepted handoff; never count it twice."""
    _controller_pair(controllers)
    if type(successful) is not bool or type(native_steps) is not int or native_steps < 1:
        raise ValueError('Stage evidence needs a native first-attempt outcome')
    if (not rows or any(not isinstance(r, dict) or not isinstance(r.get('channel'), str)
                        or not r['channel'] for r in rows)
            or len({r['channel'] for r in rows}) != 1):
        raise ValueError('Stage trace must belong to one executor channel')
    for row in rows:
        if any(type(row.get(k)) is not int or row[k] < 1 for k in ('sequence', 'steps')):
            raise ValueError('Stage trace lacks physical sequence/step accounting')
        if row['steps'] > native_steps:
            raise ValueError('Executor trace exceeds the completed native budget')
        if 'observation_accepted' in row and not isinstance(row['observation_accepted'], dict):
            raise ValueError('Stage observation receipt must be a mapping')
    if any(b['sequence'] < a['sequence'] or b['steps'] < a['steps']
           for a, b in zip(rows, rows[1:])):
        raise ValueError('Stage trace moves backwards in physical time')
    accepted = [(i, r) for i, r in enumerate(rows) if 'handoff' in r]
    if len(accepted) > 1:
        raise ValueError('Stage trace accepted more than one handoff')
    resumed = False
    if accepted:
        index, row = accepted[0]
        command = row['handoff']
        if (not isinstance(command, dict) or command.get('op') != 'handoff'
                or command.get('from_controller') != controllers[0]
                or command.get('to_controller') != controllers[1]):
            raise ValueError('Accepted handoff has the wrong controller pair')
        receipt = command.get('receipt')
        if not isinstance(receipt, dict):
            raise ValueError('Accepted handoff lacks a valid state/budget receipt')
        consumed, total, remaining = (receipt.get(k) for k in
                                     ('consumed_steps', 'total_step_limit', 'remaining_steps'))
        if (any(type(v) is not int for v in (consumed, total, remaining))
                or not 0 < consumed < total or remaining != total-consumed
                or consumed > native_steps or type(receipt.get('sequence')) is not int
                or receipt.get('sequence') != row['sequence']
                or consumed != row['steps'] or receipt.get('channel') != row['channel']
                or receipt.get('holding') != {} or receipt.get('native_success') is not None
                or receipt.get('observations') != 'three_fresh_rgb_and_measured_robot_states'
                or not isinstance(receipt.get('protected'), list) or not receipt['protected']
                or any(not isinstance(p, str) or not p for p in receipt['protected'])
                or len(set(receipt['protected'])) != len(receipt['protected'])):
            raise ValueError('Accepted handoff lacks a valid state/budget receipt')
        frames = [r for r in rows[:index] if 'observation_accepted' in r][-3:]
        if (len(frames) != 3 or len({r['sequence'] for r in frames}) != 3
                or len({r['steps'] for r in frames}) != 3
                or frames[-1]['sequence'] != row['sequence']
                or any(r['observation_accepted'].get('controller') != controllers[0]
                       or not _open_grippers(r['observation_accepted'].get('grippers')) for r in frames)):
            raise ValueError('Handoff lacks three fresh measured open-gripper frames')
        resumed = any(r.get('observation_accepted', {}).get('controller') == controllers[1]
                      and r['sequence'] > row['sequence'] and r['steps'] > row['steps']
                      for r in rows[index+1:])
    if successful and (len(accepted) != 1 or not resumed):
        raise ValueError('Successful staged probe needs one accepted handoff and a fresh stage-B observation')
    return dict(observed=bool(accepted), count=len(accepted), stage_b_resumed=resumed)


def validate_stage_handoff(route, report, source, report_path, controllers):
    """Archive one trace correlated by owned source, wall interval and channel.

    Missing traces or failure before stage B remain native failures. They cannot
    qualify a successful composition. Legacy traces have no native episode ID;
    this explicitly records correlation instead of claiming direct ID binding.
    """
    composition = route.get('selected_stage_composition')
    if not controllers:
        if composition is not None:
            raise ValueError('Route claims an unregistered stage composition')
        return None
    _controller_pair(controllers)
    if (not isinstance(composition, dict) or composition.get('controllers') != controllers
            or composition.get('max_handoffs') != 1
            or composition.get('shared_physical_budget') is not True
            or composition.get('reset_environment') is not False):
        raise ValueError('Stage route differs from the registered execution contract')
    report_path = Path(report_path)
    evidence = report_path.parent/'evidence'
    manifest = json.loads((evidence/'manifest.json').read_text())
    identity = {k: report.get(k) for k in ('task', 'seed', 'layout_id', 'episode_id')}
    if manifest.get('identity') != identity or manifest.get('attempt') != 1:
        raise ValueError('Stage evidence belongs to a different native episode')
    first = report.get('first_attempt') or {}
    if (type(first.get('success')) is not bool or type(first.get('steps')) is not int
            or first['steps'] < 1):
        raise ValueError('Stage evidence needs the native first-attempt outcome')
    audit_path, trace_path = evidence/'stage-handoff.json', evidence/'staged-executor.jsonl'
    # Validate the durable copy on subsequent reads; runtime scratch can expire.
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if (audit.get('schema') != 'dsr.stage-handoff-evidence/v1'
                or audit.get('identity') != identity or audit.get('controllers') != controllers
                or audit.get('required') is not True or audit.get('native_success') is not first['success']):
            raise ValueError('Archived stage evidence identity changed')
        if audit.get('trace'):
            payload = trace_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != audit['trace']['sha256']:
                raise ValueError('Archived stage executor trace changed')
            result = inspect_trace([json.loads(line) for line in payload.splitlines()], controllers,
                                   successful=first['success'], native_steps=first['steps'])
            if audit['trace'].get('channel') != json.loads(payload.splitlines()[0])['channel']:
                raise ValueError('Archived executor channel changed')
            if any(audit.get(k) != v for k, v in result.items()):
                raise ValueError('Archived stage boundary differs from its executor trace')
        elif first['success'] or any(audit.get(k) != v for k, v in
                    dict(observed=False, count=0, stage_b_resumed=False).items()):
            raise ValueError('Successful staged probe is missing its executor trace')
        return audit
    started = manifest.get('wall_started')
    if type(started) not in (float, int) or not math.isfinite(started):
        raise ValueError('Stage manifest lacks an episode wall interval')
    root_value = (report.get('binding') or {}).get('evidence_root')
    if not root_value or Path(source).name != source:
        raise ValueError('Stage executor lacks an owned evidence root')
    log_root = Path(root_value).resolve().parent/'probe-code'/source
    logs = sorted(p for p in log_root.glob('**/*.jsonl') if p.is_file()
                  and started <= p.stat().st_mtime <= report_path.stat().st_mtime)
    if len(logs) > 1:
        raise ValueError('Multiple executor traces overlap this staged episode')
    audit = dict(schema='dsr.stage-handoff-evidence/v1', identity=identity,
                 controllers=controllers, required=True, observed=False, count=0,
                 stage_b_resumed=False, trace=None,
                 correlation='owned controller source and native episode wall interval; one executor channel',
                 native_success=first['success'])
    if logs:
        payload = logs[0].read_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
        audit.update(inspect_trace(rows, controllers, successful=first['success'], native_steps=first['steps']))
        _save_once(trace_path, payload)
        audit['trace'] = dict(path=str(trace_path), sha256=hashlib.sha256(payload).hexdigest(),
                             source_path=str(logs[0]), channel=rows[0]['channel'])
    elif first['success']:
        raise ValueError('Successful staged probe is missing its executor trace')
    else:
        audit['unavailable_reason'] = 'No unique executor trace in the owned completed episode interval'
    _save_once(audit_path, (json.dumps(audit, indent=2, sort_keys=True)+'\n').encode())
    return audit
