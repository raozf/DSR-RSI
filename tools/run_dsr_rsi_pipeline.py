"""Durable native adaptation -> verified trajectory LoRA -> held-out evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from full_robodojo_agent_eval import ROOT, write_json, kill_group


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--codex-home", default=os.environ.get("DSR_CODEX_HOME"))
    parser.add_argument("--codex-proxy", default=os.environ.get("DSR_CODEX_PROXY"))
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_id.replace("_", "").replace("-", "").isalnum() or args.steps < 1:
        parser.error("Invalid run-id or steps")
    if not args.codex_home:
        parser.error("Set DSR_CODEX_HOME or --codex-home to an authorized Agent configuration")
    out = ROOT / "results/dsr_rsi_pipeline" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        argv = sys.argv[1:] if argv is None else argv
        command = [sys.executable, str(Path(__file__).resolve()),
                   *[v for v in argv if v != "--detach"]]
        with (out / "pipeline.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"pid": process.pid, "status_path": str(out / "status.json")}, indent=2))
        return 0
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = out / "status.json"
        state = json.loads(status_path.read_text()) if status_path.exists() else {
            "run_id": args.run_id, "stages": {}, "train_seed": 0, "test_seeds": [1, 2],
            "scope": "54 task configurations; 2100 adaptation layouts; 4200 held-out test layouts",
        }
        if state["stages"] and not args.resume:
            raise ValueError("Pipeline exists; use --resume")
        state.update(pid=os.getpid(), status="RUNNING", steps=args.steps)
        write_json(out / "config.json", vars(args))
        active = None
        stopped_signum = None

        def save(**values):
            state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
            write_json(status_path, state)

        def stop(signum, _frame):
            nonlocal stopped_signum
            stopped_signum = signum
            if active is not None:
                try:
                    os.killpg(active.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        eval_python = sys.executable
        full_data = ROOT / "vendor/XPolicyLab/policy/OpenDM/data/RoboDojo-robodojo-mem-arx_x5-joint-full"
        legacy_names = (out / "evaluations/adapt/manifest.json").exists()
        adapt_id = "adapt" if legacy_names else args.run_id + "_adapt"
        test_id = "test" if legacy_names else args.run_id + "_test"
        collection = out / "evaluations" / adapt_id
        trained = ROOT / f"vendor/XPolicyLab/policy/OpenDM/checkpoints/RoboDojo-{args.run_id}-arx_x5-joint-0"
        common_eval = [eval_python, str(ROOT / "tools/eval_dsr_rsi.py"), "--output-root", str(out / "evaluations"),
                       "--policy-gpus", "2,3,4,5", "--env-gpus", "6,7,0,1", "--codex-home", args.codex_home]
        if args.codex_proxy:
            common_eval += ["--codex-proxy", args.codex_proxy]
        stages = [
        ("adapt", [*common_eval, "--run-id", adapt_id, "--agent-mode", "adapt", "--agent-strategy", "failure_adapt", "--seeds", "0"]),
            ("train", [eval_python, str(ROOT / "tools/train_dsr_rsi.py"), "--agent-run", str(collection),
                       "--run-id", args.run_id, "--base-data", str(full_data), "--steps", str(args.steps)]),
            ("test", [*common_eval, "--run-id", test_id, "--agent-mode", "frozen", "--agent-strategy", "failure_adapt", "--seeds", "1,2",
                      "--checkpoint", str(trained), "--memory-root", str(collection / "tasks/seed_0")]),
        ]
        save()
        try:
            if not (full_data / "dataset.json").is_file():
                raise FileNotFoundError("Convert the full RoboDojo LeRobot dataset before launching this pipeline")
            for name, command in stages:
                if state["stages"].get(name, {}).get("status") == "COMPLETED":
                    continue
                previous = state["stages"].get(name)
                if args.resume and previous:
                    command.append("--resume")
                stage = {"status": "RUNNING", "command": command}
                state["stages"][name] = stage
                save(current_stage=name)
                with (out / f"{name}.log").open("a") as log:
                    active = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                              stdin=subprocess.DEVNULL, start_new_session=True)
                    stage["pid"] = active.pid
                    save()
                    stopping_at = None
                    while active.poll() is None:
                        if stopped_signum:
                            stopping_at = stopping_at or time.monotonic()
                            if time.monotonic() - stopping_at > 90:
                                kill_group(active)
                        time.sleep(1)
                    returncode = active.returncode
                active = None
                if stopped_signum:
                    stage.update(status="INTERRUPTED", exit_code=returncode)
                    save(status="STOPPED")
                    return 128 + stopped_signum
                stage.update(status="COMPLETED" if returncode == 0 else "FAILED", exit_code=returncode)
                save()
                if returncode:
                    save(status="FAILED")
                    return returncode
            save(status="COMPLETED", checkpoint=str(trained), evaluation=str(out / "evaluations" / test_id / "task_accuracy.json"))
            return 0
        except BaseException as exc:
            if state.get("status") != "STOPPED":
                save(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
