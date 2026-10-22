#!/usr/bin/env python3
"""Restore only missing IsaacLab 0.54.x dependency files from an audited tag."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/RoboDojo/third_party/IsaacLab"
PREFIX = "source/isaaclab/isaaclab/"
REPOSITORY = "isaac-sim/IsaacLab"
TAG = "v2.3.2"
COMMIT = "37ddf626871758333d6ed89cf64ad702aef127d0"
MAX_FILES = 40
MAX_BYTES = 512 * 1024
MAX_FILE_BYTES = 128 * 1024


def github_json(path: str):
    url = f"https://api.github.com/repos/{REPOSITORY}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "DSR-IsaacLab-dependency-repair"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def git_blob_sha(data: bytes):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve()
    if report_path.exists():
        parser.error(f"Refusing to overwrite an earlier restoration report: {report_path}")
    if not (VENDOR / "LICENSE").is_file():
        parser.error("Vendored IsaacLab license is missing")

    tag = github_json(f"git/ref/tags/{TAG}")
    if tag["object"]["sha"] != COMMIT:
        raise ValueError("The audited upstream tag no longer points to the expected commit")
    tree = github_json(f"git/trees/{TAG}?recursive=1")
    if tree.get("truncated"):
        raise ValueError("Upstream tree listing is truncated")
    entries = [item for item in tree["tree"] if item["type"] == "blob"
               and item["path"].startswith(PREFIX)]
    missing = [item for item in entries if not (VENDOR / item["path"]).exists()]
    if len(missing) > MAX_FILES or sum(item["size"] for item in missing) > MAX_BYTES:
        raise ValueError("Missing-file count or byte budget exceeds the audited repair scope")
    report = {"schema": "dsr.isaaclab-source-restoration/v1", "repository": REPOSITORY,
              "tag": TAG, "commit": COMMIT, "source_prefix": PREFIX,
              "upstream_files": len(entries), "missing_initially": len(missing),
              "restored": [], "status": "IN_PROGRESS"}

    def save():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)

    save()
    try:
        for item in missing:
            path = item["path"]
            destination = VENDOR / path
            if destination.resolve().is_relative_to(VENDOR.resolve()) is False:
                raise ValueError(f"Invalid upstream path: {path}")
            if item["size"] > MAX_FILE_BYTES:
                raise ValueError(f"Individual file exceeds repair budget: {path}")
            remote = github_json("contents/" + urllib.parse.quote(path, safe="/") + f"?ref={COMMIT}")
            data = base64.b64decode(remote["content"], validate=False)
            if len(data) != item["size"] or git_blob_sha(data) != item["sha"]:
                raise ValueError(f"Upstream content/hash mismatch: {path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(data)
            report["restored"].append({"path": path, "bytes": len(data), "git_blob_sha": item["sha"]})
            save()
    except Exception as exc:
        report.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    report["status"] = "COMPLETE"
    save()
    print(json.dumps({"status": report["status"], "restored": len(report["restored"]),
                      "report": str(report_path)}))


if __name__ == "__main__":
    main()
