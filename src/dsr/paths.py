"""Explicit roots; no search of an older project or implicit environment activation."""
import os
from pathlib import Path
import sys

def repository_root():
    root = Path(os.environ.get("DSR_ROOT", Path(__file__).resolve().parents[2])).resolve()
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError("DSR source bundle missing; set DSR_ROOT to an independently installed DSR checkout")
    return root

def physicalrsi_root():
    return repository_root() / "vendor" / "PhysicalRSI"

def bootstrap_policy():
    vendor = repository_root() / "vendor"
    if not (vendor / "XPolicyLab" / "process_data.py").is_file():
        raise FileNotFoundError("Pinned DSR XPolicyLab source bundle is missing")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    return vendor

def bootstrap():
    root = physicalrsi_root()
    if not (root / "PhysicalRSI_core").is_dir():
        raise FileNotFoundError("Pinned PhysicalRSI source bundle is missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Public configuration uses DSR_. Upstream wire names stay at this boundary.
    for key, value in tuple(os.environ.items()):
        if key.startswith("DSR_PHYSICALRSI_"):
            os.environ[key[4:]] = value
    return root
