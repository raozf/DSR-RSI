"""Durable same-episode strategy evidence, separate from legacy staged evidence."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from dsr.harness.strategy import DECISION_SCHEMA, PROTOCOL, StrategyLimits
from dsr.robodojo.stage_evidence import _save_once


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def inspect_calls(evidence, rows, *, successful):
    requests=[r['strategy']['request'] for r in rows
              if r.get('strategy',{}).get('kind')=='consultation']
    decisions={r['strategy']['call']:r['strategy']['decision'] for r in rows
               if r.get('strategy',{}).get('kind')=='decision'}
    expected={f'call-{n:03d}' for n in range(1,len(requests)+1)}
    if any(p.name not in expected or p.is_symlink() or not p.is_dir()
           for p in (evidence/'strategy-agent').glob('call-*')):
        raise ValueError('Strategy agent call lacks a physical consultation')
    incomplete=[]
    for number,request in enumerate(requests,1):
        call=evidence/'strategy-agent'/f'call-{number:03d}'
        required=['context.json','invocation.json','status.json']
        if any(not (call/name).is_file() for name in required):
            incomplete.append(number);continue
        context=json.loads((call/'context.json').read_text())
        invocation=json.loads((call/'invocation.json').read_text())
        status=json.loads((call/'status.json').read_text())
        if (context.get('execution')!=request or invocation.get('protocol')!=PROTOCOL
                or invocation.get('environment_reset') is not False
                or invocation.get('physical_actions')!=0
                or invocation.get('timeout_seconds')!=request['timeout_seconds']):
            raise ValueError('Strategy call audit differs from physical consultation')
        cameras=context.get('observation',{}).get('vision',{})
        if not cameras:raise ValueError('Strategy consultation lacks RGB evidence')
        for camera in cameras.values():
            name=camera['image']
            if Path(name).name!=name or digest((call/name).read_bytes())!=camera['sha256']:
                raise ValueError('Strategy consultation RGB changed')
        if number in decisions:
            if (status.get('status')!='ANSWERED' or not (call/'answer.json').is_file()
                    or json.loads((call/'answer.json').read_text())!=decisions[number]):
                raise ValueError('Accepted strategy answer differs from agent audit')
    if successful and (not requests or incomplete):
        raise ValueError('Successful strategy lacks complete agent call evidence')
    return dict(calls=len(requests),complete=not incomplete,incomplete=incomplete)


def inspect_trace(rows, settings, *, native_steps):
    import jsonschema
    if settings.get('protocol') != PROTOCOL or set(settings) != {'protocol','limits'}:
        raise ValueError('Frozen strategy settings required')
    limits=StrategyLimits(**settings['limits'])
    if (not rows or type(native_steps) is not int or native_steps<1
            or len({r.get('channel') for r in rows})!=1 or not rows[0].get('channel')):
        raise ValueError('Strategy trace requires one physical channel')
    for row in rows:
        if any(type(row.get(k)) is not int or not 1<=row[k] for k in ('sequence','steps')):
            raise ValueError('Strategy trace lacks physical sequence/steps')
        if row['steps']>native_steps:
            raise ValueError('Strategy trace exceeds native steps')
    if any(b['steps']<a['steps'] or b['sequence']<a['sequence'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Strategy trace moves backwards')
    calls=0; completed=[]; blocked=[]; decisions=[]; rejected=0
    pending=[]; used=set(); receipts={}; capabilities={}; mode=None
    for row in rows:
        event=row.get('strategy')
        if not isinstance(event,dict):continue
        kind=event.get('kind')
        if row.get('native_success') is not None or event.get('native_success') is not None:
            raise ValueError('Strategy cannot declare native success')
        if kind=='consultation':
            request=event['request'];calls+=1
            if (pending or request.get('protocol')!=PROTOCOL or request.get('calls_used')!=calls-1
                    or request.get('physical_steps')!=row['steps']
                    or calls>limits.max_agent_calls):
                raise ValueError('Strategy consultation accounting changed')
            capabilities={c['name']:c for c in request['capabilities']}
        elif kind=='decision_rejected':
            if event.get('call')!=calls:raise ValueError('Rejected decision call changed')
            rejected+=1
        elif kind=='decision':
            decision=event['decision'];jsonschema.validate(decision,DECISION_SCHEMA)
            if event.get('call')!=calls or mode is not None and decision['strategy']!='adaptive':
                raise ValueError('Strategy decision call/mode changed')
            mode=decision['strategy'];pending=deepcopy(decision['steps'])
            if decision['action']=='stop' and pending:raise ValueError('Stop contains executable tail')
            if mode=='direct' and decision['action']=='execute' and (len(pending)!=1 or pending[0]['executor']!='whole_policy'):
                raise ValueError('Direct strategy changed complete policy contract')
            for step in pending:
                cap=capabilities.get(step['capability'],{})
                if step['id'] in used or cap.get('executor')!=step['executor']:
                    raise ValueError('Strategy replayed ID or unregistered capability')
                used.add(step['id'])
            decisions.append(dict(call=calls,sequence=row['sequence'],steps=row['steps'],decision=decision))
        elif kind in ('segment_completed','segment_blocked'):
            step=event.get('step') if kind=='segment_completed' else event.get('error',{}).get('step')
            if not pending or step!=pending.pop(0):
                raise ValueError('Strategy segment differs from accepted plan')
            if len(completed)+len(blocked)>=limits.max_segments:
                raise ValueError('Strategy exceeded segment budget')
            cap=capabilities[step['capability']]
            if cap.get('input_contract') and receipts.get(step['context'])!=cap['input_contract']:
                raise ValueError('Strategy continuation lost its verified receipt')
            if kind=='segment_completed':
                receipt=step['id'] if cap.get('output_contract') else None
                if event.get('receipt')!=receipt:raise ValueError('Strategy receipt changed')
                if receipt:receipts[receipt]=cap['output_contract']
                completed.append(dict(step=step,steps=row['steps']))
            else:
                if event['error'].get('abandoned_tail')!=pending:
                    raise ValueError('Strategy abandoned tail changed')
                blocked.append(dict(step=step,steps=row['steps'],reason=event['error'].get('reason')))
                pending=[]
    return dict(agent_calls=calls,decisions=decisions,rejected_decisions=rejected,
                completed_segments=completed,blocked_segments=blocked,receipts=receipts,
                active_plan_tail=pending,mode=mode,
                accounting='One physical episode; consultations, environment retries and source revisions are separate.')


def validate_strategy(route, report, source, report_path, settings):
    selected=route.get('selected_execution_strategy')
    if not settings:
        if selected is not None:raise ValueError('Unregistered strategy route')
        return None
    if selected!=settings:raise ValueError('Strategy route differs from frozen registry')
    report_path=Path(report_path);evidence=report_path.parent/'evidence'
    manifest=json.loads((evidence/'manifest.json').read_text())
    identity={k:report.get(k) for k in ('task','seed','layout_id','episode_id')}
    if manifest.get('identity')!=identity or manifest.get('attempt')!=1:
        raise ValueError('Strategy evidence episode changed')
    first=report.get('first_attempt') or {}
    if type(first.get('success')) is not bool:raise ValueError('Native first-attempt outcome required')
    audit_path=evidence/'execution-strategy.json';trace_path=evidence/'strategy-executor.jsonl'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        if (audit.get('schema')!='dsr.execution-strategy-evidence/v1'
                or audit.get('identity')!=identity or audit.get('settings')!=settings
                or audit.get('native_success') is not first['success']):
            raise ValueError('Archived strategy identity changed')
        for ref in audit['files']:
            path=(evidence/ref['path']).resolve(strict=True)
            if not path.is_relative_to(evidence.resolve()) or digest(path.read_bytes())!=ref['sha256']:
                raise ValueError('Archived strategy file changed')
        if audit.get('trace'):
            payload=trace_path.read_bytes()
            if digest(payload)!=audit['trace']['sha256']:
                raise ValueError('Archived strategy trace changed')
            rows=[json.loads(line) for line in payload.splitlines()]
            summary=inspect_trace(rows,settings,native_steps=first['steps'])
            if summary!=audit['execution']:raise ValueError('Archived strategy execution changed')
            if inspect_calls(evidence,rows,successful=first['success'])!=audit['agent_audit']:
                raise ValueError('Archived strategy call evidence changed')
        elif first['success']:raise ValueError('Successful strategy has no executor evidence')
        return audit
    started=manifest.get('wall_started')
    if type(started) not in (int,float) or not math.isfinite(started):
        raise ValueError('Strategy evidence lacks wall interval')
    root=(report.get('binding') or {}).get('evidence_root')
    if not root or Path(source).name!=source:raise ValueError('Strategy source lacks owned root')
    directory = ('strategy-code' if route.get('schema') == 'dsr.task-policy-admission/v1'
                 else 'probe-code')
    log_root=Path(root).resolve().parent/directory/source
    logs=sorted(p for p in log_root.glob('**/env*.jsonl') if p.is_file()
                and started<=p.stat().st_mtime<=report_path.stat().st_mtime)
    if len(logs)>1:raise ValueError('Multiple strategy executor channels overlap episode')
    audit=dict(schema='dsr.execution-strategy-evidence/v1',identity=identity,settings=settings,
               native_success=first['success'],trace=None,files=[],execution=None,
               correlation='Owned controller source and native episode wall interval; one executor channel. No direct native episode ID in worker trace.')
    def archive(target,payload):
        target.parent.mkdir(parents=True,exist_ok=True);_save_once(target,payload)
        audit['files'].append(dict(path=str(target.relative_to(evidence)),sha256=digest(payload)))
    if logs:
        payload=logs[0].read_bytes();rows=[json.loads(line) for line in payload.splitlines()]
        audit['execution']=inspect_trace(rows,settings,native_steps=first['steps'])
        archive(trace_path,payload)
        audit['trace']=dict(path=trace_path.name,sha256=digest(payload),source_path=str(logs[0]),channel=rows[0]['channel'])
        agent_root=logs[0].with_suffix('.agent')
        expected_calls={f'call-{n:03d}' for n in range(1,audit['execution']['agent_calls']+1)}
        for call in sorted(agent_root.glob('call-*')):
            if call.name not in expected_calls or call.is_symlink() or not call.is_dir():
                raise ValueError('Strategy agent call lacks a physical consultation')
            for path in sorted(call.iterdir()):
                if path.is_symlink() or not path.is_file():raise ValueError('Invalid strategy audit file')
                archive(evidence/'strategy-agent'/call.name/path.name,path.read_bytes())
        audit['agent_audit']=inspect_calls(evidence,rows,successful=first['success'])
    elif first['success']:
        raise ValueError('Successful strategy has no executor evidence')
    else:audit['unavailable_reason']='No executor trace in completed native wall interval'
    _save_once(audit_path,(json.dumps(audit,indent=2,sort_keys=True)+'\n').encode())
    return audit
