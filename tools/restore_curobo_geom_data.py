#!/usr/bin/env python3
"""Restore cuRobo geometry modules matching the vendored four-argument SDF API."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/RoboDojo/third_party/curobo"
COMMIT = "457b658dbb773bf0ea2215831a1946285563fd60"
PREFIX = "curobo/_src/geom/data"
FILES = ("__init__.py", "data_cuboid.py", "data_mesh.py",
         "data_voxel.py", "data_scene.py", "helper_pose.py")
MAX_TOTAL_BYTES = 160 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--replace-restoration", type=Path,
                        help="Replace only unchanged files listed in a previous restoration report")
    args = parser.parse_args()
    report_path = args.report.resolve()
    if report_path.exists():
        parser.error(f"Refusing to overwrite prior report: {report_path}")
    if not (VENDOR / "LICENSE").is_file():
        parser.error("Vendored cuRobo license is missing")
    previous = {}
    if args.replace_restoration:
        prior = json.loads(args.replace_restoration.read_text())
        if prior.get("schema") != "dsr.curobo-source-restoration/v1" or prior.get("status") != "COMPLETE":
            parser.error("Previous restoration must be complete and use the expected schema")
        previous = {entry["path"]: entry for entry in prior["files"]}
    report = {"schema": "dsr.curobo-source-restoration/v1", "repository": "NVlabs/curobo",
              "commit": COMMIT, "source_prefix": PREFIX, "files": [], "status": "IN_PROGRESS"}

    def save():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)

    save()
    used = 0
    pending = []
    try:
        for name in FILES:
            relative = f"{PREFIX}/{name}"
            destination = VENDOR / relative
            if destination.exists():
                expected = previous.get(relative, {}).get("sha256")
                if expected != hashlib.sha256(destination.read_bytes()).hexdigest():
                    raise FileExistsError(f"Refusing to replace source outside the unchanged restoration: {destination}")
            url = f"https://api.github.com/repos/NVlabs/curobo/contents/{relative}?ref={COMMIT}"
            request = urllib.request.Request(url, headers={"User-Agent": "DSR-cuRobo-dependency-repair"})
            with urllib.request.urlopen(request, timeout=30) as response:
                entry = json.loads(response.read(MAX_TOTAL_BYTES * 2))
            data = base64.b64decode(entry["content"])
            git_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if entry["path"] != relative or git_sha != entry["sha"]:
                raise ValueError(f"Git blob verification failed: {relative}")
            used += len(data)
            if used > MAX_TOTAL_BYTES or b"SPDX-License-Identifier: Apache-2.0" not in data[:300]:
                raise ValueError(f"Invalid source size or license header: {relative}")
            pending.append((relative, destination, data, git_sha))
        for relative, destination, data, git_sha in pending:
            backup = None
            if destination.exists():
                # Recheck after download, then preserve the exact previous bytes.
                old = destination.read_bytes()
                if hashlib.sha256(old).hexdigest() != previous[relative]["sha256"]:
                    raise RuntimeError(f"Source changed during download: {destination}")
                backup = report_path.parent / (report_path.stem + "-backup") / Path(relative).name
                backup.parent.mkdir(parents=True, exist_ok=True)
                with backup.open("xb") as output:
                    output.write(old)
                temporary = destination.with_suffix(".py.dsr-restored")
                with temporary.open("xb") as output:
                    output.write(data)
                temporary.replace(destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as output:
                    output.write(data)
            report["files"].append({"path": relative, "bytes": len(data),
                                    "git_blob_sha1": git_sha,
                                    "backup": str(backup) if backup else None,
                                    "sha256": hashlib.sha256(data).hexdigest()})
            save()
    except Exception as exc:
        report.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    report["status"] = "COMPLETE"
    save()
    print(json.dumps({"status": report["status"], "restored": len(report["files"]),
                      "report": str(report_path)}))


if __name__ == "__main__":
    main()
