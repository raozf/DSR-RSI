"""Codex transport for the unchanged, pinned PhysicalRSI AgentPlanner.

Only the model/API transport is substituted. The official method constructs
the observation projection, prompts, capability catalogue and plan validation.
"""
import base64
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import SimpleNamespace
import urllib.error
import urllib.request
import uuid

from dsr.system2.codex_backend import write_json
from dsr.system2.episode_retry import (RETRY_PROPERTIES, RETRY_INSTRUCTIONS,
    bind_retry_evidence_catalog, validate_retry_decision)
from dsr.system2.retry_transport import (assert_retry_payload_budget,
    command_without_images, prepare_retry_image_transport,
    retry_format_correction_prompt, retry_format_correction_schema,
    RetryFormatBudgetError, validate_format_correction)

from dsr.paths import repository_root, physicalrsi_root
ROOT = repository_root()
UPSTREAM = physicalrsi_root()/'PhysicalRSI_baselines/robodojo/agent_planner.py'

class AgentPlanner:
    def __init__(self, *, endpoint=None, model=None):
        self.home = Path(os.environ['DSR_CODEX_HOME']).resolve()
        if not (self.home/'config.toml').is_file():
            raise FileNotFoundError(self.home/'config.toml')
        self.executable = shutil.which('codex')
        if not self.executable: raise FileNotFoundError('codex')
        self.output = (Path(os.environ['DSR_AGENT_CALL_ARCHIVE'])
                       if os.environ.get('DSR_AGENT_CALL_ARCHIVE') else
                       Path(os.environ['PHYSICALRSI_EVAL_OUTPUT'])/'agent_calls')
        self.timeout = int(os.environ.get('DSR_AGENT_TIMEOUT', '600'))
        self.retry_window = int(os.environ.get('DSR_AGENT_RETRY_WINDOW', '3600'))
        self.retry_interval = int(os.environ.get('DSR_AGENT_RETRY_INTERVAL', '60'))
        if min(self.timeout, self.retry_window, self.retry_interval) <= 0:
            raise ValueError('Agent timeouts and retry intervals must be positive')
        # A private module namespace prevents transport replacement from
        # affecting any network operation or other planner in the process.
        spec = importlib.util.spec_from_file_location(
            'PhysicalRSI_baselines.robodojo._dsr_planner_'+uuid.uuid4().hex, UPSTREAM)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.urllib = SimpleNamespace(
            request=SimpleNamespace(Request=urllib.request.Request, urlopen=self._request),
            error=urllib.error)
        self.impl = object.__new__(module.AgentPlanner)
        self.impl.api_key = 'local-cli-no-api-secret'
        self.impl.endpoint = 'http://local-codex.invalid/chat/completions'
        self.impl.model = model or 'codex-configured-model'
        self._retry_context = None
        self._retry_images = []
        self._retry_image_urls = []
        self.last_retry_decision = None

    def set_retry_context(self, context, images=(), image_urls=()):
        """Provide one failed-episode handoff to the next episode-start plan.

        The normal PhysicalRSI planner has no retry argument.  Keeping this
        handoff on the transport preserves its original first-attempt call
        signature while allowing the retry to use the same planner and
        capability catalogue.
        """
        if not isinstance(context, dict):
            raise TypeError('retry context must be a mapping')
        self._retry_context = json.loads(json.dumps(context, allow_nan=False))
        self._retry_images = list(images)
        self._retry_image_urls = list(image_urls)

    def plan(self, **kwargs):
        retry_context = getattr(self, '_retry_context', None)
        self.last_retry_decision = None
        if retry_context is not None:
            memory = kwargs.get('memory')
            memory = json.loads(json.dumps(memory or {}, allow_nan=False))
            memory['dsr_retry_context'] = retry_context
            kwargs['memory'] = memory
            # The upstream planner exposes this field in its serialized
            # context. It is the original plan for comparison, not an order
            # to repeat it.
            kwargs['previous_plan'] = retry_context.get('original_plan')
        try:
            result=self.impl.plan(**kwargs)
            if retry_context is not None and self.last_retry_decision is not None:
                observation = kwargs.get('observation')
                if isinstance(observation, dict):
                    base = observation.get('instruction', kwargs.get('instruction', ''))
                    observation['instruction'] = base + '\n' + self.last_retry_decision['instruction_suffix']
            write_json(self.last_directory/'validated_plan.json',result)
            status=json.loads((self.last_directory/'status.json').read_text())
            write_json(self.last_directory/'status.json',dict(status,status='COMPLETED'))
            return result
        except BaseException as exc:
            if hasattr(self,'last_directory'):
                write_json(self.last_directory/'validation_error.json',{'error':str(exc)})
            raise
        finally:
            # A retry context belongs to exactly one new episode-start plan.
            self._retry_context = None
            self._retry_images = []
            self._retry_image_urls = []

    def _invoke_with_retries(self, command, prompt, env, directory):
        """Wait for transient service outages without advancing the episode.

        Each attempt uses the same observation, prompt, model, and schema.
        Retry only failed transport, never an invalid plan or forbidden tool use.
        """
        started = time.monotonic()
        deadline = started + self.retry_window
        # Validation feedback may request one corrected answer in this same
        # audit directory; never overwrite the preceding transport evidence.
        existing = list((directory/'transport_attempts').glob('[0-9][0-9][0-9]'))
        attempt = max((int(path.name) for path in existing), default=0)
        while True:
            attempt += 1
            attempt_dir = directory/'transport_attempts'/f'{attempt:03d}'
            attempt_dir.mkdir(parents=True)
            argv = list(command)
            argv[argv.index('-o') + 1] = str(attempt_dir/'answer.json')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Agent service unavailable for retry window: '+str(directory))
            write_json(directory/'status.json', dict(status='REQUESTING',
                transport_attempts=attempt, retry_window_seconds=self.retry_window,
                elapsed_seconds=time.monotonic()-started))
            timed_out = False
            with (attempt_dir/'events.jsonl').open('w') as out, (attempt_dir/'stderr.log').open('w') as err:
                proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=out,
                                        stderr=err, env=env, text=True)
                try:
                    proc.communicate(prompt, timeout=min(self.timeout, remaining))
                except BaseException as exc:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                    if not isinstance(exc, subprocess.TimeoutExpired):
                        raise
                    timed_out = True
            # Keep conventional paths for readers; preserve every attempt below.
            for name in ('events.jsonl', 'stderr.log'):
                shutil.copyfile(attempt_dir/name, directory/name)
            events_text = (attempt_dir/'events.jsonl').read_text()
            for line in events_text.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('item', {}).get('type') in {
                        'command_execution', 'file_change', 'mcp_tool_call', 'web_search'}:
                    raise RuntimeError('Planner invoked a tool outside the observation boundary')
            answer_path = attempt_dir/'answer.json'
            empty_answer = answer_path.is_file() and not answer_path.read_text().strip()
            success = (not timed_out and proc.returncode == 0
                       and answer_path.is_file() and not empty_answer)
            write_json(attempt_dir/'status.json', dict(exit_code=proc.returncode,
                timed_out=timed_out, empty_answer=empty_answer, answer_received=success,
                elapsed_seconds=time.monotonic()-started))
            if success:
                shutil.copyfile(attempt_dir/'answer.json', directory/'answer.json')
                return attempt
            transient = timed_out or empty_answer or bool(re.search(
                r'(?:unexpected status\s+|last status:\s*|HTTP\s+)(?:408|429|500|502|503|504)\b|'
                r'connection (?:reset|refused|closed)|error sending request|'
                r'stream disconnected|timed out|temporary failure in name resolution',
                events_text + '\n' + (attempt_dir/'stderr.log').read_text(), flags=re.IGNORECASE))
            remaining = deadline - time.monotonic()
            if not transient or remaining <= 0:
                raise RuntimeError('Codex planner failed after '+str(attempt)+
                                   ' transport attempts: '+str(directory))
            delay = min(self.retry_interval, remaining)
            write_json(directory/'status.json', dict(status='WAITING_FOR_AGENT',
                transport_attempts=attempt, retry_window_seconds=self.retry_window,
                elapsed_seconds=time.monotonic()-started, retry_in_seconds=delay))
            time.sleep(delay)

    def _request(self, request, timeout=None):
        body = json.loads(request.data)
        directory = self.output/uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        self.last_directory=directory
        messages = body['messages']
        text_parts = [part['text'] for part in messages[1]['content'] if part['type']=='text']
        context = json.loads(text_parts[0])
        retry_context = None
        policy_memory = context.get('task_specific_policy_memory')
        if isinstance(policy_memory, dict):
            retry_context = policy_memory.get('dsr_retry_context')
        if isinstance(retry_context, dict) and (self._retry_images or getattr(self, '_retry_image_urls', [])):
            # Attach terminal/start RGB evidence to the same planner request.
            # The labels are inserted into the structured context so the model
            # can associate each image with its evidence role.
            retry_context = dict(retry_context)
            labels = list(retry_context.get('image_labels', []))
            retry_context['attached_image_labels'] = labels
            policy_memory = dict(policy_memory)
            policy_memory['dsr_retry_context'] = retry_context
            context = dict(context, task_specific_policy_memory=policy_memory)
            messages[1]['content'][0]['text'] = json.dumps(context, allow_nan=False)
            image_urls = list(getattr(self, '_retry_image_urls', []))
            for image in self._retry_images:
                stream = io.BytesIO()
                image.save(stream, format='PNG') if hasattr(image, 'save') else self._write_png(stream, image)
                image_urls.append('data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode())
            for image_url in image_urls:
                if not isinstance(image_url, str) or not image_url.startswith('data:image/png;base64,'):
                    raise ValueError('Retry evidence images must be inline PNG data URLs')
                messages[1]['content'].append({
                    'type': 'image_url',
                    'image_url': {'url': image_url},
                })
        properties = {
            'composition': {'type':'string','enum':list(context['available_compositions'])},
            'rationale': {'type':'string'},
        }
        if 'operation_program' in context:
            properties['operations']={'type':'array','items':{'type':'string'}}
        if isinstance(retry_context, dict):
            properties.update(RETRY_PROPERTIES)
        schema={'type':'object','additionalProperties':False,'properties':properties,
                'required':list(properties)}
        if isinstance(retry_context, dict):
            try:
                schema = bind_retry_evidence_catalog(schema, retry_context.get('evidence_catalog', []))
            except ValueError as exc:
                write_json(directory/'status.json', dict(status='FAILED',
                    phase='retry_evidence_catalog_validation', reason='invalid_retry_evidence_catalog',
                    error_type=type(exc).__name__, error=str(exc), transport_attempts=0,
                    physical_actions=0, physical_execution_started=False))
                write_json(directory/'validation_error.json', dict(error=str(exc),
                    phase='retry_evidence_catalog_validation'))
                raise
        write_json(directory/'schema.json', schema)
        if isinstance(retry_context, dict):
            write_json(directory/'evidence_catalog_binding.json', dict(
                protocol='dsr.retry_evidence_catalog_binding/v1',
                references=sorted(entry['ref'] for entry in retry_context['evidence_catalog']),
                fields=['failure_analysis.cause_candidates[].evidence_refs[]',
                        'repeat_strategy.evidence_refs[]'],
                schema_sha256=sha256((directory/'schema.json').read_bytes()).hexdigest()))
        write_json(directory/'context.json', context)
        (directory/'system_prompt.txt').write_text(messages[0]['content']+'\n')
        images=[]
        for part in messages[1]['content']:
            if part['type']!='image_url':continue
            url=part['image_url']['url']
            if not url.startswith('data:image/png;base64,'):
                raise ValueError('Official planner must supply inline PNG observations')
            path=directory/f'image_{len(images)}.png'
            path.write_bytes(base64.b64decode(url.split(',',1)[1]));images.append(path)
        system_prompt = messages[0]['content']
        if isinstance(retry_context, dict):
            system_prompt = RETRY_INSTRUCTIONS + '\n\n' + system_prompt
            text_parts = [json.dumps(context, allow_nan=False)]
            (directory/'system_prompt.txt').write_text(system_prompt+'\n')
        prompt=(system_prompt+'\nReturn the required JSON using only the supplied '
                'observation evidence. Do not invoke tools, inspect files, or modify code.\n'
                +'\n'.join(text_parts))
        if isinstance(retry_context, dict):
            # Preserve the original observations and all audit PNGs. Only
            # failure-history transport files may be downsampled, with exact
            # source/reference bindings and a measured encoded-byte budget.
            preflight_started = time.monotonic()
            try:
                images, transport_metadata = prepare_retry_image_transport(
                    images, retry_context.get('image_labels', []), prompt, schema, directory)
                retry_context = dict(retry_context, image_transport=transport_metadata)
                policy_memory = dict(policy_memory, dsr_retry_context=retry_context)
                context = dict(context, task_specific_policy_memory=policy_memory)
                text_parts = [json.dumps(context, allow_nan=False)]
                prompt=(system_prompt+'\nReturn the required JSON using only the supplied '
                        'observation evidence. Do not invoke tools, inspect files, or modify code.\n'
                        +'\n'.join(text_parts))
                measured_bytes = assert_retry_payload_budget(prompt, schema, images)
                write_json(directory/'image_transport.json', dict(transport_metadata,
                    encoded_request_bytes=measured_bytes,
                    prompt_sha256=sha256(prompt.encode()).hexdigest()))
                write_json(directory/'context.json', context)
            except Exception as exc:
                write_json(directory/'status.json', dict(status='FAILED',
                    phase='retry_payload_preflight', reason='retry_evidence_payload_rejected',
                    error_type=type(exc).__name__, error=str(exc), transport_attempts=0,
                    physical_actions=0, physical_execution_started=False,
                    elapsed_seconds=time.monotonic()-preflight_started))
                write_json(directory/'validation_error.json', dict(error=str(exc),
                    phase='retry_payload_preflight'))
                raise
        (directory/'prompt.txt').write_text(prompt+'\n')
        command=[self.executable,'-a','never','exec','--skip-git-repo-check','--ephemeral',
                 '-s','read-only','-C',str(directory),'--json','--output-schema',
                 str(directory/'schema.json'),'-o',str(directory/'answer.json')]
        for path in images:command.extend(['--image',str(path)])
        command.append('-')
        env=os.environ.copy();env['CODEX_HOME']=str(self.home)
        for key in ('LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH'):env.pop(key,None)
        proxy=env.get('DSR_CODEX_PROXY')
        if proxy:
            for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
                env[key]=proxy
        started=time.monotonic()
        write_json(directory/'invocation.json',dict(command=command,
            phase='failure_retry_episode_start' if retry_context is not None else 'episode_start',
            role='physicalrsi_memory_conditioned_skill_composition',status='RUNNING',
            config_home=str(self.home),official_prompt=True,
            retry_window_seconds=self.retry_window))
        try:
            # A syntactically incomplete answer or an over-budget instruction
            # can be corrected before any action. Other contract violations
            # remain hard failures; no unvalidated plan reaches the robot.
            validation_attempts = 0
            current_prompt = prompt
            current_command = command
            current_images = images
            current_schema = schema
            format_source = None
            while True:
                validation_attempts += 1
                if isinstance(retry_context, dict):
                    assert_retry_payload_budget(current_prompt, current_schema, current_images)
                attempts = self._invoke_with_retries(current_command, current_prompt, env, directory)
                for line in (directory/'events.jsonl').read_text().splitlines():
                    if not line.strip():
                        continue
                    event=json.loads(line);item=event.get('item',{})
                    if item.get('type') in {'command_execution','file_change','mcp_tool_call','web_search'}:
                        raise RuntimeError('Planner invoked a tool outside the observation boundary')
                try:
                    answer=json.loads((directory/'answer.json').read_text())
                    if isinstance(retry_context, dict):
                        if format_source is not None:
                            validate_format_correction(answer, format_source, current_schema)
                        route, guidance = validate_retry_decision(answer, retry_context,
                            context['available_compositions'], context.get('operation_program'))
                        self.last_retry_decision = dict(answer, instruction_suffix=guidance)
                        write_json(directory/'retry_decision.json', self.last_retry_decision)
                        answer = route
                    break
                except ValueError as exc:
                    correctable = (isinstance(exc, json.JSONDecodeError)
                                   or 'Compiled retry instruction exceeds' in str(exc))
                    if not correctable or validation_attempts >= 2:
                        raise
                    rejected = (directory/'answer.json').read_text()
                    write_json(directory/f'rejected_answer_{validation_attempts}.json',
                               dict(error=str(exc), answer=rejected))
                    # The answer is untrusted evidence, and the original
                    # task, observation and registered capabilities stay fixed.
                    if (isinstance(retry_context, dict)
                            and not isinstance(exc, json.JSONDecodeError)):
                        # Length repair has an already parsed decision. Do not
                        # re-send a long trajectory and all RGB merely to
                        # shorten action/check wording. The diagnosis, route,
                        # stages and constraints are checked for conservation.
                        format_source = answer
                        try:
                            current_schema, formatting_budget = retry_format_correction_schema(answer, schema)
                        except RetryFormatBudgetError as budget_error:
                            write_json(directory/f'correction_transport_{validation_attempts}.json', dict(
                                mode='retry_plan_wording_only', status='REJECTED', image_count=0,
                                reason='retry_format_infeasible', formatting_budget=budget_error.budget,
                                correction_transport_attempts=0, physical_actions=0))
                            raise
                        schema_path = directory/f'correction_schema_{validation_attempts}.json'
                        write_json(schema_path, current_schema)
                        current_prompt = retry_format_correction_prompt(answer, retry_context, exc,
                                                                        formatting_budget)
                        current_command = command_without_images(command)
                        current_command[current_command.index('--output-schema') + 1] = str(schema_path)
                        current_images = []
                        write_json(directory/f'correction_transport_{validation_attempts}.json', dict(
                            mode='retry_plan_wording_only', image_count=0,
                            command=current_command, schema_file=schema_path.name,
                            schema_sha256=sha256(schema_path.read_bytes()).hexdigest(),
                            formatting_budget=formatting_budget,
                            prompt_sha256=sha256(current_prompt.encode()).hexdigest(),
                            original_prompt_sha256=sha256(prompt.encode()).hexdigest(),
                            original_image_labels=retry_context.get('image_labels', []),
                            encoded_request_bytes=assert_retry_payload_budget(current_prompt, current_schema, [])))
                    else:
                        current_prompt = prompt + '\nValidation feedback (data):\n' + json.dumps({
                            'error': str(exc), 'rejected_answer': rejected,
                            'instruction': 'Return a corrected complete JSON answer. Keep the same task and evidence. '
                                           'Use concise steps; the entire rendered retry guidance must fit 1200 characters.'})
                    (directory/f'correction_prompt_{validation_attempts}.txt').write_text(current_prompt)
            write_json(directory/'status.json',dict(status='ANSWER_RECEIVED',
                elapsed_seconds=time.monotonic()-started,exit_code=0,
                transport_attempts=attempts, validation_attempts=validation_attempts))
            # Upstream AgentPlanner validates the plan after consuming this
            # exact Chat Completions response shape. Do not add fallbacks here.
            return io.BytesIO(json.dumps({'choices':[{'message':{'content':json.dumps(answer)}}]}).encode())
        except BaseException as exc:
            details = {}
            if isinstance(exc, RetryFormatBudgetError):
                details = dict(phase='retry_format_budget', reason='retry_format_infeasible',
                    error_type=type(exc).__name__, transport_attempts=attempts,
                    correction_transport_attempts=0, physical_actions=0,
                    physical_execution_started=False, formatting_budget=exc.budget)
            elif format_source is not None and isinstance(exc, ValueError):
                details = dict(phase='retry_format_validation', reason='retry_format_invalid',
                    error_type=type(exc).__name__, transport_attempts=attempts,
                    physical_actions=0, physical_execution_started=False)
            elif isinstance(exc, ValueError) and str(exc) in {
                    'Failure analysis references unavailable evidence',
                    'Repeated strategy references unavailable evidence'}:
                details = dict(phase='retry_decision_validation',
                    reason='planner_evidence_reference_validation_failure',
                    error_type=type(exc).__name__, transport_attempts=attempts,
                    physical_actions=0, physical_execution_started=False)
            write_json(directory/'status.json',dict(status='FAILED',
                elapsed_seconds=time.monotonic()-started,error=str(exc), **details))
            raise

    @staticmethod
    def _write_png(stream, image):
        from PIL import Image
        import numpy as np
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError('Retry evidence images must be RGB or RGBA arrays')
        Image.fromarray(array.astype('uint8')).convert('RGB').save(stream, format='PNG')
