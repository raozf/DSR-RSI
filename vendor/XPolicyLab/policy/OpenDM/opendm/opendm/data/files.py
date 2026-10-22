"""Local file I/O used by DSR's offline OpenDM deployment.

The upstream ``megfile`` import eagerly loads cloud and SSH dependencies even
for local checkpoints. DSR's default inference and local training paths need
only these four file operations. Reject remote URLs explicitly so unsupported
cloud storage cannot silently be interpreted as a local path.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from urllib.parse import urlparse


def _local_path(value: str | Path) -> Path:
    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(f"Remote storage requires an explicit adapter: {parsed.scheme}")
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError("Remote file authorities are unsupported")
        return Path(parsed.path)
    return Path(text)


def smart_open(value: str | Path, mode: str = "r"):
    path = _local_path(value)
    if any(flag in mode for flag in ("w", "a", "x")):
        path.parent.mkdir(parents=True, exist_ok=True)
    return path.open(mode)


def smart_exists(value: str | Path) -> bool:
    return _local_path(value).exists()


def smart_isdir(value: str | Path) -> bool:
    return _local_path(value).is_dir()


def smart_glob(pattern: str) -> list[str]:
    return sorted(glob(str(_local_path(pattern)), recursive=True))
