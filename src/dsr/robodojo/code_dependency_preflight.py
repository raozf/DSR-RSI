"""Preflight vendored Code Policy dependencies without starting a simulator.

The check runs the exact interpreter selected by ``DSR_CODE_PYTHON`` (or the
current Python), adds only descriptor-declared paths plus pinned iopath, and
reports missing imports. It never installs or removes packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


VENDORED_IMPORTS = {
    "iopath": "dependencies/iopath/5aa7e4c36e6c68f0",
    "ftfy": "dependencies/ftfy/61862e9c43ef746f",
}
DEPENDENCY_IDENTITIES = {
    "iopath": ("iopath/common/file_io.py", "fd20391c8ebebacf9acbc1336f0643433dd5668410e55c0d05e066d51cfd541e"),
}


def vendored_import_paths(assets):
    assets = Path(assets)
    result = []
    for name, relative in VENDORED_IMPORTS.items():
        root = assets / relative
        if not (root / name / "__init__.py").is_file():
            raise FileNotFoundError("Pinned dependency is missing: " + str(root / name))
        if name in DEPENDENCY_IDENTITIES:
            relative_file, expected = DEPENDENCY_IDENTITIES[name]
            if hashlib.sha256((root / relative_file).read_bytes()).hexdigest() != expected:
                raise ValueError("Vendored dependency differs from the pinned revision: " + name)
        result.append(root)
    return result


def _expand(value, assets: Path):
    if isinstance(value, str):
        return value.replace("{assets}", str(assets)).replace("{root}", str(assets))
    if isinstance(value, list):
        return [_expand(item, assets) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, assets) for key, item in value.items()}
    return value


def descriptor_probe(descriptor: str | Path, assets: str | Path, *, python: str | Path | None = None) -> dict:
    descriptor = Path(descriptor).resolve(strict=True)
    assets = Path(assets).resolve(strict=True)
    config = _expand(json.loads(descriptor.read_text(encoding="utf-8")), assets)
    interpreter = Path(python or os.environ.get("DSR_CODE_PYTHON", sys.executable)).expanduser().absolute()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise FileNotFoundError("Code interpreter is not executable: " + str(interpreter))
    paths = [Path(path) for path in config.get("pythonpath", [])]
    # SAM3's model_builder imports iopath before the policy's own module runs.
    if any("/dependencies/sam3/" in str(path) for path in paths):
        paths = vendored_import_paths(assets) + paths
    probe = "import importlib, json, sys, traceback\n" \
            "mods = json.loads(sys.argv[1])\n" \
            "out = {}\n" \
            "for name in mods:\n" \
            "    try:\n" \
            "        module = importlib.import_module(name)\n" \
            "        out[name] = {'origin': getattr(module, '__file__', None), 'error': None}\n" \
            "    except Exception as error:\n" \
            "        out[name] = {'origin': None, 'error': f'{type(error).__name__}: {error}'}\n" \
            "print(json.dumps(out, sort_keys=True))\n"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    required = ["torch"]
    if any("/dependencies/sam3/" in str(path) for path in paths):
        required.extend(["iopath.common.file_io", "ftfy", "sam3.model.sam3_image_processor", "sam3.model_builder"])
    result = subprocess.run([str(interpreter), "-c", probe, json.dumps(required)],
                            env=env, capture_output=True, text=True, check=False, timeout=180)
    if result.returncode:
        raise RuntimeError("Dependency probe interpreter failed: " + result.stderr.strip())
    found = json.loads(result.stdout)
    missing = [name for name, result in found.items() if result["origin"] is None]
    return {"schema": "dsr.code-dependency-preflight/v1", "descriptor": str(descriptor),
            "interpreter": str(interpreter), "required": required, "resolved": found,
            "missing": missing, "status": "ready" if not missing else "missing_dependency",
            "installation_performed": False}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(descriptor_probe(args.descriptor, args.assets, python=args.python), indent=2))


if __name__ == "__main__":
    main()
