"""Isolate the pinned Kit scripting cache without changing HOME or vendor files.

Kit prepends its own fast importer, so a normal meta-path finder cannot reliably
intercept this extension. Its relative ``from .omni_cache import OmniCache``
still uses Python's import function. A temporary wrapper changes only that
module's local Path binding, before the extension calls initialize().
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib

from dsr.runtime_config import independent_path, local_state_root


MODULE = "omni.kit.scripting.scripts.loader.omni_cache"
EXTENSION_VERSION = "107.3.2"
SOURCE_SHA256 = "32d3da8c0778e72b1421c3212425e6990db1034488bd2ddcc4efde2b4828d000"
SOURCE_SUFFIX = Path("omni/kit/scripting/scripts/loader/omni_cache.py")


class KitScriptingCacheIsolation:
    """One-shot startup shim, with a receipt and a verified native cache path."""

    def __init__(self, root=None):
        configured = root if root is not None else os.environ.get("DSR_KIT_SCRIPTING_CACHE_ROOT")
        self.root = None
        self.receipt = {"schema": "dsr.kit-scripting-cache-isolation/v1", "pid": os.getpid(),
                        "module": MODULE, "expected_version": EXTENSION_VERSION,
                        "expected_source_sha256": SOURCE_SHA256}
        if configured is not None:
            self.root = independent_path(configured, purpose="DSR_KIT_SCRIPTING_CACHE_ROOT")
            state = local_state_root()
            if self.root == state or not self.root.is_relative_to(state):
                raise ValueError("Kit scripting cache root must be a child of DSR_LOCAL_STATE")
            self.root.mkdir(parents=True, exist_ok=True)
            self.receipt["cache_root"] = str(self.root)
        self._home_env = os.environ.get("HOME")
        self._real_home = Path.home()
        self._original_import = None
        self._wrapper = None
        self._patched = False
        self._patching = False

    @property
    def cache_path(self):
        return self.root / ".nvidia-omniverse" / "pycache"

    def _save(self, status, **fields):
        self.receipt.update(status=status, **fields)
        temporary = self.root / f".isolation-{os.getpid()}.tmp"
        temporary.write_text(json.dumps(self.receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.root / "isolation.json")

    def _restore(self):
        if self._wrapper is not None:
            if builtins.__import__ is self._wrapper:
                builtins.__import__ = self._original_import
            elif builtins.__import__ is not self._original_import:
                raise RuntimeError("Kit replaced the cache import wrapper; restoration cannot be verified")

    def _patch(self, module):
        if self._patching or self._patched:
            return
        self._patching = True
        try:
            source = Path(module.__file__).resolve()
            if module.__name__ != MODULE or tuple(source.parts[-6:]) != SOURCE_SUFFIX.parts:
                raise RuntimeError("Unexpected Kit scripting cache module identity")
            extension = source.parents[5]
            config = tomllib.loads((extension / "config/extension.toml").read_text())
            version = config.get("package", {}).get("version")
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            if version != EXTENSION_VERSION or checksum != SOURCE_SHA256:
                raise RuntimeError("Unreviewed Kit scripting cache version or source fingerprint")
            if module.Path is not Path or module.OmniCache._path is not None:
                raise RuntimeError("Kit scripting cache was modified or initialized before isolation")
            root = self.root

            class ScriptingCachePath(type(Path())):
                @classmethod
                def home(cls):
                    return cls(root)

            module.Path = ScriptingCachePath
            self._patched = True
            self._restore()
            self._save("PATCHED", source=str(source), source_sha256=checksum,
                       extension_version=version, expected_cache_path=str(self.cache_path))
        finally:
            self._patching = False

    def __enter__(self):
        if self.root is None:
            return self
        self._original_import = builtins.__import__
        module = sys.modules.get(MODULE)
        if module is not None:
            self._patch(module)
            return self

        def importing(name, globals=None, locals=None, fromlist=(), level=0):
            result = self._original_import(name, globals, locals, fromlist, level)
            target = sys.modules.get(MODULE)
            if target is not None and hasattr(target, "OmniCache") and hasattr(target, "Path"):
                self._patch(target)
            return result

        self._wrapper = importing
        builtins.__import__ = importing
        try:
            self._save("ARMED")
        except BaseException:
            self._restore()
            raise
        return self

    def verify_initialized(self):
        if self.root is None:
            return
        module = sys.modules.get(MODULE)
        if not self._patched or module is None:
            raise RuntimeError("Kit scripting cache isolation did not intercept the pinned extension")
        actual = module.OmniCache._path
        if actual is None or Path(actual).resolve() != self.cache_path or not Path(actual).is_dir():
            raise RuntimeError("Kit scripting cache did not initialize at the isolated path")
        if os.environ.get("HOME") != self._home_env or Path.home() != self._real_home:
            raise RuntimeError("Global HOME or pathlib.Path.home changed during Kit startup")
        self._restore()
        self._save("VERIFIED", actual_cache_path=str(actual), home_environment_unchanged=True,
                   pathlib_home_unchanged=True, import_wrapper_restored=True)

    def __exit__(self, kind, error, traceback):
        if self.root is None:
            return False
        try:
            if error is None:
                self.verify_initialized()
            else:
                self._save("FAILED", error=f"{kind.__name__}: {error}")
        except Exception as exc:
            self._save("FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._restore()
        return False


def isolate_kit_scripting_cache():
    """Activate only when the simulator launcher supplies an isolated root."""
    return KitScriptingCacheIsolation()
