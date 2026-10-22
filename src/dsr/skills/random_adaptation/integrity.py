"""Fail before controller initialization if a frozen candidate package changed."""
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_package(cfg):
    root=Path(__file__).parent.resolve()
    manifest=root/'package.json'
    if sha(manifest)!=cfg['random_package_sha256']:raise ValueError('random package manifest changed')
    package=json.loads(manifest.read_text())
    if package.get('schema')!='dsr.random-controller-package/v1':
        raise ValueError('unsupported random package schema')
    files=package.get('files')
    task=cfg.get('task_name')
    controller=package.get('controllers',{}).get(task)
    if not isinstance(files,dict) or not isinstance(controller,dict):
        raise ValueError('random controller is not registered in the frozen package')
    if controller.get('revision')!=cfg.get('random_controller_revision'):
        raise ValueError('random controller revision mismatch')
    if cfg.get('random_controller_module')!=controller.get('module'):
        raise ValueError('random controller module mismatch')
    for relative,digest in files.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path)!=digest:
            raise ValueError('random package source changed: '+relative)
    for source,digest in controller.get('dependencies',{}).items():
        path=Path(source)
        if not path.is_absolute() or not path.is_file() or sha(path)!=digest:
            raise ValueError('random controller dependency changed: '+source)
    return root
