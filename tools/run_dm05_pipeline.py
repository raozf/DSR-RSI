"""Resume ModelScope downloads, train DM05, and evaluate real checkpoints.

This is an engineering smoke pipeline. It does not run the mock RSI loop or
claim benchmark-quality results. Existing aria2 transfers are adopted in place.
"""

from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import sys

from verify_dm05_checkpoint import RELEASES, verify

from dsr.paths import repository_root
from dsr.runtime_config import checkpoint_root
ROOT = repository_root()
OUT = ROOT / "results/dm05_pipeline"
POLICY = ROOT / "vendor/XPolicyLab/policy/OpenDM"


def now():
    return datetime.now(timezone.utc).isoformat()


def existing_downloader(name):
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
            if (args and Path(os.fsdecode(args[0])).name == "aria2c"
                    and (process / "cwd").resolve() == ROOT
                    and any(("/" + name + "/repo?").encode() in arg for arg in args)):
                return int(process.name)
        except (OSError, ValueError):
            continue
    return None


def gpu_available(ids, minimum_mib):
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
        free = {int(row.split(",")[0]): int(row.split(",")[1]) for row in result.splitlines()}
        return all(free.get(int(gpu), 0) >= minimum_mib for gpu in ids.split(",")), free
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-failed", action="store_true", help="Retry interrupted/failed stages, preserving their history")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args.retry_failed)


def run(retry_failed=False):
    path = OUT / "status.json"
    state = json.loads(path.read_text()) if path.exists() else {"created_at": now(), "stages": {}}
    for name, stage in list(state["stages"].items()):
        if stage["status"] == "RUNNING":
            stage.update(status="INTERRUPTED", reason="Previous pipeline exited before recording completion")
        if retry_failed and stage["status"] != "COMPLETED":
            state.setdefault("stage_history", []).append({"stage": name, **stage})
            del state["stages"][name]
    state.update(pid=os.getpid(), status="RUNNING", is_research_result=False,
                 scope="DM05 executor: 10 optimizer steps, trained-checkpoint and released-checkpoint simulation smoke")
    owned_downloads = {}

    def save(**fields):
        state.update(updated_at=now(), **fields)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)

    def launch_download(kind):
        name, _, expected = RELEASES[kind]
        if kind in owned_downloads:
            process, log = owned_downloads[kind]
            code = process.poll()
            if code is None:
                return process.pid
            log.close()
            del owned_downloads[kind]
            if code == 0:
                partial = checkpoint_root() / name / "model.safetensors.downloading"
                if partial.exists():
                    partial.replace(partial.with_name("model.safetensors"))
                return None
        active = existing_downloader(name)
        if active:
            return active
        # An adopted downloader's shell may have just promoted the verified file.
        if (checkpoint_root() / name / "model.safetensors").exists():
            return None
        if not os.environ.get("https_proxy"):
            raise RuntimeError("Set the authorized HTTPS proxy before starting the pipeline")
        directory = checkpoint_root() / name
        command = ["aria2c", "--https-proxy=" + os.environ["https_proxy"],
                   "--http-proxy=" + os.environ.get("http_proxy", os.environ["https_proxy"]),
                   "-x16", "-s16", "-k1M", "--file-allocation=none", "--continue=true",
                   "--auto-file-renaming=false", "--check-integrity=true", "--checksum=sha-256=" + expected,
                   "--timeout=60", "--connect-timeout=20", "--max-tries=8", "--retry-wait=5",
                   "--summary-interval=60", "--show-console-readout=false", "--console-log-level=warn",
                   "-d", str(directory), "-o", "model.safetensors.downloading",
                   f"https://www.modelscope.cn/api/v1/models/Dexmal/{name}/repo?Revision=master&FilePath=model.safetensors"]
        log = (OUT / f"download_{kind}.log").open("a")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned_downloads[kind] = (process, log)
        return process.pid

    def execute(stage, script, env, result_file):
        state["stages"][stage] = {"status": "RUNNING", "started_at": now(), "result_file": str(result_file)}
        save(status="RUNNING", active_stage=stage)
        print(f"{now()} Starting {stage}", flush=True)
        try:
            with (OUT / f"{stage}.log").open("a") as log:
                tool = "train_dm05_local.py" if script.startswith("train") else "eval_dm05_local.py"
                result = subprocess.run([sys.executable, str(ROOT / "tools" / tool)], cwd=ROOT,
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
            record = json.loads(result_file.read_text())
            status = "COMPLETED" if result.returncode == 0 and record.get("status") == "COMPLETED" else "FAILED"
            state["stages"][stage].update(status=status, exit_code=result.returncode,
                                         finished_at=now(), reason=record.get("reason"))
        except Exception as exc:
            state["stages"][stage].update(status="FAILED", finished_at=now(), reason=str(exc))
        save(active_stage=None)
        print(f"{now()} {stage}: {state['stages'][stage]['status']}", flush=True)

    env = os.environ.copy()
    env.update(DSR_DRY_RUN="0", DSR_TRAIN_STEPS="10", DSR_EVAL_NUM="1", DSR_EVAL_TASK="build_tower")
    gpu_ids = env.get("DSR_TRAIN_GPU_IDS", "4,5,6,7")
    lora = env.get("DSR_TRAIN_LORA", "0") == "1"
    name = "dm05-lora-smoke" if lora else "dm05-smoke"
    trained = POLICY / f"checkpoints/RoboDojo-{name}-arx_x5-joint-0"
    train_out = Path(env.get("DSR_TRAIN_OUT", ROOT / "results/dm05_training_smoke"))
    eval_out = Path(env.get("DSR_TRAINED_EVAL_OUT", ROOT / "results/dm05_trained_eval_smoke"))
    save(training_adaptation="lora" if lora else "full", training_checkpoint=str(trained))
    save(active_stage=None)
    deadline = time.monotonic() + int(os.environ.get("DSR_PIPELINE_TIMEOUT", "14400"))
    while time.monotonic() < deadline:
        models = {}
        for kind in RELEASES:
            # Reuse cached verification only if size and mtime are unchanged.
            record = verify(kind)
            if record["status"] == "INCOMPLETE":
                record["downloader_pid"] = launch_download(kind)
            elif record["status"] not in ("VERIFIED", "INCOMPLETE"):
                save(status="BLOCKED", reason=f"Checkpoint {kind}: {record['status']}")
                return 2
            models[kind] = record
        save(models=models, pending_models=[kind for kind, record in models.items() if record["status"] != "VERIFIED"])
        stages = state["stages"]
        can_train, free = gpu_available(gpu_ids, int(env.get("DSR_TRAIN_MIN_FREE_MIB", "48000")))
        can_eval, _ = gpu_available(env.get("DSR_POLICY_GPU", "4"), 36000)
        can_sim, _ = gpu_available(env.get("DSR_SIM_GPU", "6"), 12000)
        save(gpu_free_mib=free)
        if "training" not in stages and models["base"]["status"] == "VERIFIED" and can_train:
            execute("training", "train_dm05_smoke.sh", env, train_out / "training_manifest.json")
            continue
        if (stages.get("training", {}).get("status") == "COMPLETED"
                and "trained_evaluation" not in stages and can_eval and can_sim):
            eval_env = {**env, "DSR_MODEL_PATH": str(trained), "DSR_EVAL_OUT": str(eval_out)}
            execute("trained_evaluation", "eval_dm05_smoke.sh", eval_env,
                    eval_out / "metrics.json")
            continue
        if "released_evaluation" not in stages and models["sim"]["status"] == "VERIFIED" and can_eval and can_sim:
            eval_env = {**env, "DSR_MODEL_PATH": str(checkpoint_root() / "DM05-MEM-Robodojo-Sim"),
                        "DSR_EVAL_OUT": str(ROOT / "results/dm05_eval_smoke")}
            execute("released_evaluation", "eval_dm05_smoke.sh", eval_env,
                    ROOT / "results/dm05_eval_smoke/metrics.json")
            continue
        if stages.get("training", {}).get("status") in ("FAILED", "INTERRUPTED"):
            stages.setdefault("trained_evaluation", {"status": "BLOCKED", "reason": "Training failed"})
        expected = ("training", "trained_evaluation", "released_evaluation")
        if all(name in stages for name in expected):
            complete = all(stages[name]["status"] == "COMPLETED" for name in expected)
            save(status="COMPLETED" if complete else "FAILED", finished_at=now())
            return 0 if complete else 1
        pending = [name for name, record in models.items() if record["status"] != "VERIFIED"]
        save(status="WAITING_FOR_DOWNLOADS" if pending else "WAITING_FOR_GPU_MEMORY", pending_models=pending)
        time.sleep(30)
    save(status="INCOMPLETE", reason="Configured DSR_PIPELINE_TIMEOUT reached")
    for process, log in owned_downloads.values():
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        log.close()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
