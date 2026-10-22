"""Fail closed on missing or modified frozen sources and dependency identities."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(cfg):
    root = Path(__file__).parent
    manifest = root / 'package.json'
    if sha(manifest) != cfg['specialist_package_sha256']:
        raise ValueError('specialist package identity changed')
    data = json.loads(manifest.read_text())
    if data['schema'] != 'dsr.native-specialists/v1' or cfg['task_name'] not in data['tasks']:
        raise ValueError('specialist task not frozen')
    if 'execution_contracts' in data:
        contract = data['execution_contracts'].get(cfg['task_name'])
        if contract is None or cfg.get('stage_composition', []) != contract['stage_composition']:
            raise ValueError('specialist runtime stage composition differs from frozen contract')
        if cfg.get('execution_strategy') != contract.get('execution_strategy'):
            raise ValueError('specialist execution strategy differs from frozen contract')
    elif cfg.get('execution_strategy'):
        raise ValueError('Agent strategy requires a frozen execution contract')
    for relative, expected in data['files'].items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError('specialist source changed: ' + relative)
    for path, expected in data['dependencies'].items():
        if sha(path) != expected:
            raise ValueError('specialist dependency changed: ' + path)
    return root
