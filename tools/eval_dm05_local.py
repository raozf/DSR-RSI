"""Evaluate DM05 in RoboDojo with separate project-local policy and Isaac environments."""

from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import time

from check_dm05_training import validate as validate_training
from verify_dm05_checkpoint import verify

from dsr.paths import repository_root
from dsr.runtime_config import checkpoint_root, interpreter, local_state_root, simulator_command, process_environment, preflight, verify_state_lock
ROOT = repository_root()
POLICY = ROOT / "vendor/XPolicyLab/policy/OpenDM"


def validate_metrics(result, episodes):
    if int(result["eval_time"]) != episodes or len(result.get("details", {})) != episodes:
        raise ValueError("Evaluation did not produce the requested number of episode records")
    sr, score = float(result["success_rate"]), float(result["score"])
    # RoboDojo stores SR as a fraction and score in percent.
    if not (math.isfinite(sr) and 0 <= sr <= 1 and math.isfinite(score) and 0 <= score <= 100):
        raise ValueError("Invalid RoboDojo metric range")
    return {"episodes": episodes, "success_rate": sr, "success_rate_percent": 100 * sr,
            "score": score, "success_rate_unit": "fraction", "score_unit": "percent"}


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
    except ProcessLookupError:
        pass


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main():
    out = Path(os.environ.get("DSR_EVAL_OUT", ROOT / "results/dm05_eval_smoke")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"Preserve historical evaluation evidence; choose a fresh DSR_EVAL_OUT: {out}")
    locks = local_state_root() / "evaluation_locks"
    verify_state_lock(locks)
    key = hashlib.sha256(str(out).encode()).hexdigest()
    with (locks / f"dm05-{key}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "metrics.json").exists():
            raise FileExistsError(f"Evaluation evidence already exists: {out}")
        return run(out)


def run(out):
    model = Path(os.environ.get("DSR_MODEL_PATH", checkpoint_root() / "DM05-MEM-Robodojo-Sim")).resolve()
    task = os.environ.get("DSR_EVAL_TASK", "build_tower")
    seed = int(os.environ.get("DSR_EVAL_SEED", "0"))
    episodes = int(os.environ.get("DSR_EVAL_NUM", "1"))
    if episodes < 1:
        raise ValueError("EVAL_NUM must be positive")
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    policy_gpu = os.environ.get("DSR_POLICY_GPU")
    sim_gpu = os.environ.get("DSR_SIM_GPU")
    if not policy_gpu or not sim_gpu or not policy_gpu.isdecimal() or not sim_gpu.isdecimal():
        raise ValueError("Declare DSR_POLICY_GPU and DSR_SIM_GPU explicitly after checking shared GPU workloads")
    start_timeout = int(os.environ.get("DSR_POLICY_START_TIMEOUT", "600"))
    eval_timeout = int(os.environ.get("DSR_EVAL_TIMEOUT", "600"))
    if start_timeout <= 0 or eval_timeout <= 0:
        raise ValueError("Policy and simulator time budgets must be positive")
    dry_run = os.environ.get("DSR_DRY_RUN", "1") == "1"
    manifest = {"status": "PREPARING", "run_id": run_id, "task": task, "seed": seed,
                "checkpoint": str(model), "requested_episodes": episodes,
                "policy_gpu": policy_gpu, "simulation_gpu": sim_gpu,
                "policy_start_timeout_seconds": start_timeout, "evaluation_timeout_seconds": eval_timeout,
                "scope": "DM05 executor simulation smoke; not an RSI benchmark",
                "is_research_result": False}

    def save(status, **fields):
        manifest.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
        temporary = out / "metrics.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(out / "metrics.json")

    save("PREPARING")
    try:
        if not dry_run:
            if model == (checkpoint_root() / "DM05-MEM-Robodojo-Sim").resolve():
                record = verify("sim")
                if record["status"] != "VERIFIED":
                    save("BLOCKED", reason="Released simulation checkpoint is not SHA-256 verified")
                    return 2
            else:
                record = validate_training(model, int(os.environ.get("DSR_TRAIN_STEPS", "10")))
            (out / "checkpoint_verification.json").write_text(json.dumps(record, indent=2) + "\n")

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        server = [str(interpreter("policy", required=not dry_run)), "-u",
                  str(ROOT / "vendor/XPolicyLab/setup_policy_server.py"),
                  "--config_path", str(POLICY / "deploy.yml"), "--overrides",
                  "host=127.0.0.1", f"port={port}", "bench_name=RoboDojo", f"task_name={task}",
                  f"ckpt_name={model.name}", "env_cfg_type=arx_x5", f"seed={seed}",
                  "policy_name=OpenDM", "action_type=joint", "action_dim=14",
                  f"model_path={model}", f"norm_stats_path={model / 'norm_stats.json'}",
                  "liger_kernel=False"]
        client_args = ["python", "-u", "src/eval_client/main.py", "--task_name", task,
                       "--env_cfg_type", "arx_x5", "--num_envs", "1", "--enable_cameras",
                       "--device_id", "0", "--policy_name", "OpenDM", "--port", str(port),
                       "--host", "127.0.0.1", "--protocol", "ws", "--policy_server_url",
                       f"ws://127.0.0.1:{port}", "--additional_info", "dsr_smoke",
                       "--seed", str(seed), "--headless"]
        client = simulator_command(client_args, required=not dry_run)
        manifest.update(server_command=server, client_command=client)
        (out / "commands.json").write_text(json.dumps({"server": server, "client": client}, indent=2) + "\n")
        if dry_run:
            save("DRY_RUN", reason="Commands generated; no model inference or simulation episode executed")
            print(json.dumps(manifest, indent=2))
            return 0
        preflight()
        env = process_environment(policy_gpu)
        env.update(NO_ALBUMENTATIONS_UPDATE="1", TOKENIZERS_PARALLELISM="false",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled",
                   PYTHONNOUSERSITE="1",
                   PYTHONPATH=env["PYTHONPATH"],
                   CUDA_VISIBLE_DEVICES=policy_gpu, OMP_NUM_THREADS="4")
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            env.pop(key, None)
        env.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
        signal.signal(signal.SIGTERM, interrupted)
        with ExitStack() as resources:
            log = resources.enter_context((out / "policy_server.log").open("w"))
            server_process = subprocess.Popen(server, cwd=POLICY, env=env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
            resources.callback(stop, server_process)
            save("LOADING_POLICY", policy_pid=server_process.pid)
            deadline = time.monotonic() + start_timeout
            while True:
                if server_process.poll() is not None:
                    raise RuntimeError(f"Policy server exited with {server_process.returncode}; see policy_server.log")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Policy server did not become ready")
                    time.sleep(2)
            sim_env = process_environment(sim_gpu, role="sim")
            sim_env.update(EVAL_NUM=str(episodes), ROBODOJO_RUN_ID=run_id,
                           ROBODOJO_FATAL_RESTART_COUNT="3", ROBODOJO_MAX_BASH_RETRIES="0")
            sim_log = resources.enter_context((out / "simulator.log").open("w"))
            client_process = subprocess.Popen(client, cwd=ROOT/"vendor/RoboDojo", env=sim_env, stdout=sim_log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
            resources.callback(stop, client_process)
            save("EVALUATING", simulator_pid=client_process.pid)
            deadline = time.monotonic() + eval_timeout
            while client_process.poll() is None:
                if server_process.poll() is not None:
                    raise RuntimeError("Policy server exited during evaluation")
                if time.monotonic() >= deadline:
                    raise TimeoutError("RoboDojo evaluation exceeded EVAL_TIMEOUT")
                time.sleep(2)
            if client_process.returncode:
                raise RuntimeError(f"Simulator exited with {client_process.returncode}; see simulator.log")
        result_path = (ROOT / "vendor/RoboDojo/eval_result/RoboDojo" / task / "OpenDM/arx_x5"
                       / f"{seed}_dsr_smoke" / run_id / "_result.json")
        result = json.loads(result_path.read_text())
        metrics = validate_metrics(result, episodes)
        shutil.copyfile(result_path, out / "raw_result.json")
        save("COMPLETED", result_source=str(result_path), **metrics)
        print(json.dumps(manifest, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        save("FAILED", reason=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
