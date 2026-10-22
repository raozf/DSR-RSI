"""Bounded multimodal decisions for the same physical episode.

Self-contained so an isolated specialist build can freeze this transport too.
No reset/retry or source-repair protocol is accepted here.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
from PIL import Image

if __package__ == 'dsr.system2':
    from dsr.harness.strategy import DECISION_SCHEMA, INSTRUCTIONS, PROTOCOL
else:
    from .strategy_runtime import DECISION_SCHEMA, INSTRUCTIONS, PROTOCOL


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def numbers(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Unserializable agent context: ' + type(value).__name__)


class StrategyAgent:
    def __init__(self, output, config_home=None):
        self.output = Path(output).resolve()
        self.home = Path(config_home or os.environ['DSR_CODEX_HOME']).resolve()
        if not (self.home / 'config.toml').is_file():
            raise FileNotFoundError(self.home / 'config.toml')
        self.executable = shutil.which('codex')
        if self.executable is None:
            raise FileNotFoundError('codex CLI is not on PATH')
        self.calls = 0

    def decide(self, request, observation):
        if request.get('protocol') != PROTOCOL:
            raise ValueError('Same-episode strategy protocol required')
        timeout = request['timeout_seconds']
        if type(timeout) not in (int, float) or not 0 < timeout <= 600:
            raise ValueError('Bounded strategy call timeout required')
        self.calls += 1
        directory = self.output / f'call-{self.calls:03d}'
        directory.mkdir(parents=True, exist_ok=False)
        # A whitelist projection, including cameras, also applies when this
        # transport is used outside the native worker's clean_batch boundary.
        state_fields = {a+s for a in ('left','right') for s in (
            '_arm_joint_state','_ee_joint_state','_ee_pose','_tcp_pose','_delta_ee_pose')}
        visible = dict(state={k: deepcopy(v) for k, v in observation['state'].items() if k in state_fields},
                       vision={})
        images = []
        for camera in ('cam_head','cam_left_wrist','cam_right_wrist'):
            if camera not in observation['vision']:
                continue
            source = observation['vision'][camera]
            rgb = np.asarray(source['color'])
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] not in (3,4):
                raise ValueError('Strategy cameras require uint8 RGB or RGBA')
            path = directory / f'{camera}.png'
            Image.fromarray(rgb).convert('RGB').save(path)
            images.append(path)
            visible['vision'][camera] = {k: deepcopy(source[k]) for k in (
                'intrinsic_matrix','extrinsic_matrix','shape') if k in source}
            visible['vision'][camera].update(image=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if not images:
            raise ValueError('Current RGB required for strategy decisions')
        context = json.loads(json.dumps(dict(execution=request, observation=visible),
                                       default=numbers, allow_nan=False))
        write(directory / 'context.json', context)
        write(directory / 'schema.json', DECISION_SCHEMA)
        prompt = INSTRUCTIONS + '\n' + json.dumps(context, allow_nan=False)
        (directory / 'prompt.txt').write_text(prompt + '\n')
        command = [self.executable, '-a', 'never', 'exec', '--skip-git-repo-check', '--ephemeral',
                   '-s', 'read-only', '-C', str(directory), '--json', '--output-schema',
                   str(directory / 'schema.json'), '-o', str(directory / 'answer.json')]
        for path in images:
            command.extend(['--image', str(path)])
        command.append('-')
        env = os.environ.copy()
        env['CODEX_HOME'] = str(self.home)
        for key in ('LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH'):
            env.pop(key, None)
        if env.get('DSR_CODEX_PROXY'):
            for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
                env[key] = env['DSR_CODEX_PROXY']
        started = time.monotonic()
        process = None
        write(directory / 'invocation.json', dict(protocol=PROTOCOL, command=command,
              timeout_seconds=timeout, environment_reset=False, physical_actions=0))
        try:
            with (directory / 'events.jsonl').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout,
                    stderr=stderr, env=env, text=True, start_new_session=True)
                try:
                    process.communicate(prompt, timeout=timeout)
                except BaseException:
                    # Kill the entire CLI process group, including a stalled
                    # transport child. There is no hidden transport retry loop.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    raise
            if process.returncode:
                raise RuntimeError('Strategy agent exited with code ' + str(process.returncode))
            for line in (directory / 'events.jsonl').read_text().splitlines():
                event = json.loads(line)
                if event.get('item', {}).get('type') in {
                        'command_execution','file_change','mcp_tool_call','web_search'}:
                    raise RuntimeError('Strategy agent used tools outside supplied evidence')
            answer = json.loads((directory / 'answer.json').read_text())
            # Registry, context and state contracts are checked by Strategy;
            # malformed decisions return there as bounded validator feedback.
            write(directory / 'status.json', dict(status='ANSWERED', seconds=time.monotonic()-started))
            return answer
        except Exception as error:
            write(directory / 'status.json', dict(status='REJECTED', seconds=time.monotonic()-started,
                  error=type(error).__name__ + ': ' + str(error),
                  exit_code=process.returncode if process else None))
            if isinstance(error, subprocess.TimeoutExpired):
                raise TimeoutError('Strategy agent exceeded call deadline') from error
            if isinstance(error, OSError):
                raise RuntimeError('Strategy agent transport failed: ' + str(error)) from error
            raise
