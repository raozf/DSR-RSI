"""Bounded eight-GPU RoboDojo smoke test, using four independent DM05 workers.

This orchestrates existing eval_dm05_local.py; it does not change the benchmark,
launch unbounded retries, or stop processes that it did not create. GPU locks
coordinate DSR launchers; an external scheduler can still race a resource check.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dsr.runtime_config import checkpoint_root, interpreter, local_state_root, preflight, verify_state_lock

PAIRS = ((0, 1, 0), (2, 3, 1), (4, 5, 2), (6, 7, 0))
TERMINAL = {"COMPLETED", "FAILED", "TIMED_OUT", "RESOURCE_TIMEOUT", "INTERRUPTED"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def query_gpus():
    """Fail closed if either memory or compute-process inventory is unavailable."""
    def query(kind, fields):
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=15,
        )
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))

    gpus = {}
    by_uuid = {}
    for row in query("gpu", "index,uuid,memory.free,memory.total"):
        if len(row) != 4:
            raise ValueError(f"Unexpected nvidia-smi GPU record: {row!r}")
        index = int(row[0])
        gpus[index] = {"index": index, "uuid": row[1].strip(),
                       "free_mib": int(row[2]), "total_mib": int(row[3]), "compute_pids": []}
        by_uuid[row[1].strip()] = index
    for row in query("compute-apps", "gpu_uuid,pid,used_gpu_memory"):
        if len(row) != 3 or row[0].strip() not in by_uuid:
            raise ValueError(f"Unexpected nvidia-smi process record: {row!r}")
        gpus[by_uuid[row[0].strip()]]["compute_pids"].append(int(row[1]))
    return gpus


def resource_blockers(pair, snapshot, policy_free_mib, sim_free_mib, allowed_processes=None):
    reasons = []
    allowed_processes = allowed_processes or {}
    for role, threshold in (("policy", policy_free_mib), ("sim", sim_free_mib)):
        index = pair[f"{role}_gpu"]
        gpu = snapshot.get(index)
        if gpu is None:
            reasons.append(f"GPU {index} ({role}) missing from inventory")
            continue
        unexpected = sorted(set(gpu["compute_pids"]) - set(allowed_processes.get(index, ())))
        if unexpected:
            reasons.append(f"GPU {index} ({role}) has unapproved compute PIDs {unexpected}")
        if gpu["free_mib"] < threshold:
            reasons.append(f"GPU {index} ({role}) free {gpu['free_mib']} MiB < {threshold} MiB")
    return reasons


def process_identity(pid):
    try:
        # comm can include whitespace or parentheses; fields after the last ')'
        # begin at state (field 3). starttime is field 22, ppid is field 4.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "ppid": int(fields[1]), "starttime": fields[19], "state": fields[0]}
    except (OSError, IndexError, ValueError):
        return None


def collect_descendants(parent_identity, known):
    identities = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            identity = process_identity(int(entry.name))
            if identity:
                identities[identity["pid"]] = identity
    parent_pid = parent_identity["pid"]
    parents = ({parent_pid} if identities.get(parent_pid, {}).get("starttime") == parent_identity["starttime"] else set())
    # Existing descendants may have been reparented after their parent exited.
    parents.update(pid for pid, identity in known.items()
                   if identities.get(pid, {}).get("starttime") == identity["starttime"])
    changed = True
    while changed:
        changed = False
        for pid, identity in identities.items():
            if identity["ppid"] in parents and pid not in parents:
                parents.add(pid)
                known[pid] = identity
                changed = True
    return known


def signal_owned(identity, sig):
    current = process_identity(identity["pid"])
    if current and current["starttime"] == identity["starttime"] and current.get("state") != "Z":
        try:
            os.kill(identity["pid"], sig)
        except ProcessLookupError:
            pass


def stop_worker(worker, grace_seconds=20):
    """Let the evaluator run its cleanup, then stop only verified descendants."""
    process = worker["process"]
    owned = collect_descendants(worker["identity"], worker["owned"])
    if process.poll() is None:
        signal_owned(worker["identity"], signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        collect_descendants(worker["identity"], owned)
        alive = [item for item in owned.values()
                 if (current := process_identity(item["pid"]))
                 and current["starttime"] == item["starttime"] and current.get("state") != "Z"]
        if process.poll() is not None and not alive:
            break
        if process.poll() is not None:
            for identity in alive:
                signal_owned(identity, signal.SIGTERM)
        time.sleep(0.2)
    for identity in [*owned.values(), worker["identity"]]:
        signal_owned(identity, signal.SIGKILL)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    worker["log"].close()
    for lock in worker["locks"]:
        lock.close()


def acquire_gpu_locks(directory, indices):
    locks = []
    try:
        for index in sorted(indices):
            stream = (directory / f"gpu-{index}.lock").open("a")
            locks.append(stream)
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return locks
    except BlockingIOError:
        for stream in locks:
            stream.close()
        return None
    except BaseException:
        for stream in locks:
            stream.close()
        raise


def worker_spec(pair, args):
    env = os.environ.copy()
    env.update(DSR_EVAL_OUT=pair["output"], DSR_EVAL_TASK="align_blocks",
               DSR_EVAL_SEED=str(pair["seed"]), DSR_EVAL_NUM="1",
               DSR_POLICY_GPU=str(pair["policy_gpu"]), DSR_SIM_GPU=str(pair["sim_gpu"]),
               DSR_POLICY_START_TIMEOUT=str(args.policy_start_timeout),
               DSR_EVAL_TIMEOUT=str(args.eval_timeout), DSR_DRY_RUN="0",
               PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT / "vendor/PhysicalRSI"))),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    if "local_state" in pair:
        state = Path(pair["local_state"])
        env["DSR_LOCAL_STATE"] = str(state)
        for key, suffix in (("TMPDIR", "tmp"), ("HF_HOME", "cache/huggingface"),
                            ("XDG_CACHE_HOME", "cache"), ("TORCH_EXTENSIONS_DIR", "cache/torch_extensions"),
                            ("CUDA_CACHE_PATH", "cache/cuda")):
            # Explicit user cache settings are retained; defaults are isolated
            # so concurrent native workers do not share compilation/Kit state.
            env.setdefault(key, str(state / suffix))
        # Kit portable paths are independent of XDG_CACHE_HOME. Its default
        # datastore lives beside the installed package, on the shared volume
        # which cannot provide the locks required by its cache.
        kit_root = state / "kit"
        if any(char.isspace() for char in str(kit_root)):
            raise ValueError("Kit portable root cannot contain whitespace")
        kit_options = env.get("DSR_KIT_ARGS", "")
        if "--portable-root" not in kit_options:
            kit_options += f" --portable-root {kit_root}"
        defaults = {
            "/UJITSO/datastore/localCachePath": str(kit_root / "DerivedDataCache"),
        }
        for key, value in defaults.items():
            if f"--{key}=" not in kit_options:
                kit_options += f" --{key}={value}"
        env["DSR_KIT_ARGS"] = kit_options.strip()
    # Interpreter paths and local state are inherited, preserving venv symlinks.
    command = [sys.executable, "-u", str(ROOT / "tools/eval_dm05_local.py")]
    return command, env


def result_status(returncode, metrics):
    if returncode != 0:
        return "FAILED", f"Evaluator exited with {returncode}"
    if metrics.get("status") != "COMPLETED":
        return "FAILED", f"No completed episode metrics: {metrics.get('status', 'missing')}"
    if metrics.get("episodes") != 1:
        return "FAILED", "Expected exactly one native episode"
    return "COMPLETED", None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Launch guarded GPU workers")
    mode.add_argument("--preflight-only", action="store_true", help="Check runtime and GPU availability, no workers")
    mode.add_argument("--dry-run", action="store_true", help="Record commands and resource state, no workers (default)")
    parser.add_argument("--output", type=Path, help="Fresh output directory (must not exist)")
    parser.add_argument("--max-wait", type=int, default=900)
    parser.add_argument("--policy-start-timeout", type=int, default=600)
    parser.add_argument("--eval-timeout", type=int, default=600)
    parser.add_argument("--worker-overhead-timeout", type=int, default=300,
                        help="Additional time for checkpoint verification and teardown")
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument("--policy-free-mib", type=int, default=34000)
    parser.add_argument("--sim-free-mib", type=int, default=20000)
    parser.add_argument("--allow-existing-processes", action="store_true",
                        help="Explicitly permit sharing GPUs with existing compute PIDs if free-memory guards pass")
    args = parser.parse_args(argv)
    for name in ("max_wait", "policy_start_timeout", "eval_timeout", "worker_overhead_timeout"):
        if not 0 < getattr(args, name) <= 86400:
            parser.error(f"--{name.replace('_', '-')} must be in [1, 86400] seconds")
    if not 0.1 <= args.poll_interval <= 60:
        parser.error("--poll-interval must be in [0.1, 60] seconds")
    if min(args.policy_free_mib, args.sim_free_mib) < 1:
        parser.error("GPU memory thresholds must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = (args.output or ROOT / "results" / f"robodojo_eight_gpu_{run_id}").absolute()
    output.mkdir(parents=True, exist_ok=False)
    pairs = [dict(id=f"pair-{number}", policy_gpu=policy, sim_gpu=sim, seed=seed,
                  output=str(output / f"pair-{number}"), status="QUEUED",
                  local_state=str(local_state_root() / "eightgpu" / run_id / f"pair-{number}"))
             for number, (policy, sim, seed) in enumerate(PAIRS)]
    run_budget = args.policy_start_timeout + args.eval_timeout + args.worker_overhead_timeout
    report = dict(run_id=run_id, status="PREPARING", created_at=utc_now(),
                  mode="execute" if args.execute else "preflight" if args.preflight_only else "dry_run",
                  task="align_blocks", episodes_per_pair=1, physical_gpu_ids=list(range(8)),
                  scope="Four DM05 native RoboDojo smoke episodes; not a full benchmark or RSI evaluation",
                  is_research_result=False, recovery="off", pairs=pairs,
                  max_wait_seconds=args.max_wait, worker_timeout_seconds=run_budget,
                  policy_start_timeout_seconds=args.policy_start_timeout,
                  eval_timeout_seconds=args.eval_timeout,
                  policy_free_mib=args.policy_free_mib, sim_free_mib=args.sim_free_mib,
                  allow_existing_processes=args.allow_existing_processes,
                  resource_guard=("Sufficient free memory; explicitly permit preexisting compute PIDs" if args.allow_existing_processes
                                  else "No compute PIDs and sufficient free memory on both GPUs before launch"),
                  resource_guard_limitation="Local locks do not reserve resources against external schedulers")
    path = output / "status.json"
    workers = {}

    def save():
        report["updated_at"] = utc_now()
        save_json(path, report)

    def snapshot():
        try:
            value = query_gpus()
            report["gpu_snapshot"] = value
            report.pop("resource_query_error", None)
            return value
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            report["resource_query_error"] = f"{type(exc).__name__}: {exc}"
            return {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    save()
    try:
        report["runtime"] = {"policy_python": str(interpreter("policy", required=False)),
                             "sim_python": str(interpreter("sim", required=False)),
                             "local_state": str(local_state_root()),
                             "checkpoint": os.environ.get("DSR_MODEL_PATH", str(checkpoint_root() / "DM05-MEM-Robodojo-Sim"))}
        if args.execute or args.preflight_only:
            report["preflight"] = preflight()
        initial = snapshot()
        allowed_processes = ({index: list(gpu["compute_pids"]) for index, gpu in initial.items()}
                             if args.allow_existing_processes else {})
        report["allowed_existing_processes"] = allowed_processes
        for pair in pairs:
            pair["resource_blockers"] = resource_blockers(pair, initial, args.policy_free_mib, args.sim_free_mib, allowed_processes)
            command, env = worker_spec(pair, args)
            pair["command"] = command
            pair["configuration"] = {key: env[key] for key in (
                "DSR_EVAL_TASK", "DSR_EVAL_SEED", "DSR_EVAL_NUM", "DSR_POLICY_GPU", "DSR_SIM_GPU",
                "DSR_POLICY_START_TIMEOUT", "DSR_EVAL_TIMEOUT", "DSR_EVAL_OUT", "DSR_DRY_RUN",
                "DSR_LOCAL_STATE", "TMPDIR", "HF_HOME", "XDG_CACHE_HOME", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH")}
        if not args.execute:
            report["status"] = "PREFLIGHT" if args.preflight_only else "DRY_RUN"
            report["ready_pairs"] = sum(not pair["resource_blockers"] for pair in pairs)
            save()
            print(json.dumps(report, indent=2))
            return 0 if not args.preflight_only or report["ready_pairs"] == len(pairs) else 2
        locks_dir = local_state_root() / "evaluation_locks" / "gpus"
        verify_state_lock(locks_dir)
        deadline = time.monotonic() + args.max_wait
        report["status"] = "RUNNING"
        report["launcher_pid"] = os.getpid()
        while any(pair["status"] not in TERMINAL for pair in pairs):
            current = time.monotonic()
            gpu_state = snapshot()
            for pair in pairs:
                worker = workers.get(pair["id"])
                if worker:
                    collect_descendants(worker["identity"], worker["owned"])
                    code = worker["process"].poll()
                    if code is None and current < worker["deadline"]:
                        continue
                    metrics_path = Path(pair["output"]) / "metrics.json"
                    try:
                        metrics = json.loads(metrics_path.read_text())
                    except (OSError, ValueError):
                        metrics = {}
                    if code is None:
                        pair.update(status="TIMED_OUT", reason="Worker exceeded total bounded runtime")
                    else:
                        state, reason = result_status(code, metrics)
                        pair.update(status=state, reason=reason, returncode=code)
                    pair.update(finished_at=utc_now(), metrics=metrics,
                                elapsed_seconds=round(current - worker["started"], 3))
                    stop_worker(worker)
                    del workers[pair["id"]]
                elif pair["status"] == "QUEUED":
                    if current >= deadline:
                        pair.update(status="RESOURCE_TIMEOUT", finished_at=utc_now(),
                                    reason="GPU pair remained unavailable within queue budget")
                        continue
                    pair["resource_blockers"] = resource_blockers(pair, gpu_state, args.policy_free_mib, args.sim_free_mib, allowed_processes)
                    if pair["resource_blockers"]:
                        continue
                    locks = acquire_gpu_locks(locks_dir, (pair["policy_gpu"], pair["sim_gpu"]))
                    if locks is None:
                        pair["resource_blockers"] = ["Another DSR launcher holds a GPU lock"]
                        continue
                    # Recheck after locking, immediately before process creation.
                    launch_snapshot = snapshot()
                    pair["resource_blockers"] = resource_blockers(pair, launch_snapshot, args.policy_free_mib, args.sim_free_mib, allowed_processes)
                    if pair["resource_blockers"]:
                        for stream in locks:
                            stream.close()
                        continue
                    pair["gpu_snapshot_at_launch"] = {index: launch_snapshot[index] for index in (pair["policy_gpu"], pair["sim_gpu"])}
                    child_out = Path(pair["output"])
                    child_out.mkdir()
                    log = (child_out / "launcher.log").open("w")
                    command, env = worker_spec(pair, args)
                    try:
                        process = subprocess.Popen(command, env=env, cwd=ROOT,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close()
                        for stream in locks:
                            stream.close()
                        raise
                    identity = process_identity(process.pid)
                    if identity is None:
                        # An immediately exited child is still reaped normally.
                        identity = {"pid": process.pid, "ppid": os.getpid(), "starttime": "unavailable"}
                    started = time.monotonic()
                    workers[pair["id"]] = dict(process=process, identity=identity, owned={}, log=log,
                                               locks=locks, started=started, deadline=started + run_budget)
                    pair.update(status="RUNNING", pid=process.pid, started_at=utc_now())
            save()
            if any(pair["status"] not in TERMINAL for pair in pairs):
                time.sleep(args.poll_interval)
        report["status"] = "COMPLETED" if all(pair["status"] == "COMPLETED" for pair in pairs) else "INCOMPLETE"
        report["completed_pairs"] = sum(pair["status"] == "COMPLETED" for pair in pairs)
        report["finished_at"] = utc_now()
        save()
        print(json.dumps({"status": report["status"], "report": str(path), "completed_pairs": report["completed_pairs"]}))
        return 0 if report["status"] == "COMPLETED" else 2
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                      reason=f"{type(exc).__name__}: {exc}", finished_at=utc_now())
        for pair in pairs:
            if pair["status"] not in TERMINAL:
                pair.update(status=report["status"], reason=report["reason"])
        save()
        print(f"{report['status']}: {exc}; report: {path}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        for worker in workers.values():
            stop_worker(worker)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
