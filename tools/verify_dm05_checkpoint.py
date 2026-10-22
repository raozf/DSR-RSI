"""Verify released ModelScope DM05 weights; sparse files are never accepted."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from dsr.paths import repository_root
from dsr.runtime_config import checkpoint_root
ROOT = repository_root()
RELEASES = {
    "base": ("DM05-MEM", 11658431104, "cbbdf7b8d874225d49d789390c87720c62ce0332f39a47531a4d1266606ec09a"),
    "sim": ("DM05-MEM-Robodojo-Sim", 23316655136, "db5f19ac29bd049ceee542db37263a470c15a6c33a589a446e274c5ac5cbf3ba"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(kind, rehash=False):
    name, size, expected = RELEASES[kind]
    directory = checkpoint_root() / name
    weight = directory / "model.safetensors"
    stamp = ROOT / "runtime/cache/checkpoint_verification" / (name + ".json")
    record = {"repo": "Dexmal/" + name, "path": str(weight),
              "source": "ModelScope", "expected_bytes": size,
              "sha256_expected": expected, "status": "INCOMPLETE"}
    if not weight.is_file():
        partial = directory / "model.safetensors.downloading"
        record["actual_bytes"] = 0
        if partial.exists():
            record["partial_allocated_bytes"] = partial.stat().st_blocks * 512
            record["note"] = "Allocated disk space is an estimate, not verified downloaded bytes."
        return record
    stat = weight.stat()
    record.update(actual_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if stat.st_size != size or weight.with_suffix(weight.suffix + ".aria2").exists():
        return record
    if not rehash and stamp.exists():
        cached = json.loads(stamp.read_text())
        if (cached.get("mtime_ns") == stat.st_mtime_ns and
                cached.get("actual_bytes") == size and cached.get("sha256") == expected):
            return cached
    actual = sha256(weight)
    if weight.stat().st_mtime_ns != stat.st_mtime_ns:
        record["status"] = "CHANGED_DURING_VERIFICATION"
        return record
    record.update(sha256=actual, verified_at=datetime.now(timezone.utc).isoformat(),
                  status="VERIFIED" if actual == expected else "HASH_MISMATCH")
    if record["status"] == "VERIFIED":
        stamp.parent.mkdir(parents=True, exist_ok=True)
        temporary = stamp.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(stamp)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["base", "sim", "all"])
    parser.add_argument("--rehash", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [verify(k, args.rehash) for k in (RELEASES if args.kind == "all" else [args.kind])]
    payload = {"checked_at": datetime.now(timezone.utc).isoformat(), "models": records}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(rendered)
        tmp.replace(args.output)
    print(rendered, end="")
    return 0 if all(r["status"] == "VERIFIED" for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
