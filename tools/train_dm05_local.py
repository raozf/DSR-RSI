"""Run the real OpenDM training recipe using the project-local environment."""

import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
from datetime import datetime, timezone

from check_dm05_training import validate
from verify_dm05_checkpoint import verify

from dsr.paths import repository_root
from dsr.runtime_config import interpreter, process_environment, checkpoint_root
ROOT = repository_root()
POLICY = ROOT / "vendor/XPolicyLab/policy/OpenDM"


def main():
    out = Path(os.environ.get("DSR_TRAIN_OUT", ROOT / "results/dm05_training_smoke")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(out)


def run(out):
    steps = int(os.environ.get("DSR_TRAIN_STEPS", "10"))
    seed = int(os.environ.get("DSR_TRAIN_SEED", "0"))
    gpu_ids = os.environ.get("DSR_TRAIN_GPU_IDS", "4,5,6,7")
    world_size = len(gpu_ids.split(","))
    dry_run = os.environ.get("DSR_DRY_RUN", "1") == "1"
    data = Path(os.environ.get("DSR_TRAIN_DATA_ROOT", POLICY / "data/RoboDojo-robodojo-mem-arx_x5-joint-smoke"))
    lora = os.environ.get("DSR_TRAIN_LORA", "0") == "1"
    checkpoint_name = os.environ.get(
        "TRAIN_CHECKPOINT_NAME", "dm05-lora-smoke" if lora else "dm05-smoke"
    )
    checkpoint = POLICY / f"checkpoints/RoboDojo-{checkpoint_name}-arx_x5-joint-{seed}"
    manifest = {"status": "PREPARING", "started_at": datetime.now(timezone.utc).isoformat(),
                "requested_steps": steps, "seed": seed, "gpu_ids": gpu_ids,
                "adaptation": "lora" if lora else "full",
                "dataset": str(data), "checkpoint": str(checkpoint),
                "scope": os.environ.get("DSR_TRAIN_SCOPE", "DM05 executor training smoke; not an RSI benchmark"),
                "is_research_result": False}

    def save(status, **fields):
        manifest.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
        temporary = out / "training_manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(out / "training_manifest.json")

    command = ["bash", str(POLICY / "train.sh"), "RoboDojo", checkpoint_name, "arx_x5", "joint",
               str(seed), gpu_ids, "--data-root", str(data), "--per-device-batch", "1",
               "--global-batch", os.environ.get("DSR_TRAIN_GLOBAL_BATCH", str(world_size)), "--steps", str(steps),
               "--save-steps", os.environ.get("DSR_TRAIN_SAVE_STEPS", str(steps)), "--workers", "1", "--sdpa"]
    base_model = os.environ.get("DSR_TRAIN_BASE_MODEL", str(checkpoint_root()/"DM05-MEM"))
    norm_stats_root = os.environ.get("DSR_TRAIN_NORM_STATS_ROOT", str(checkpoint_root()/"robodojo-norm"))
    if base_model:
        command.extend(["--base-model", base_model])
    if norm_stats_root:
        command.extend(["--norm-stats-root", norm_stats_root])
    if lora:
        command.append("--lora")
    if dry_run:
        command.append("--dry-run")
    (out / "command.txt").write_text(shlex.join(command) + "\n")
    save("PREPARING", command=command)
    try:
        if dry_run:
            save("DRY_RUN", reason="Commands generated; no trainer or optimizer executed")
            print(json.dumps(manifest, indent=2))
            return 0
        if not dry_run:
            record = verify(os.environ.get("DSR_TRAIN_VERIFY_RELEASE", "base"))
            (out / "checkpoint_verification.json").write_text(json.dumps(record, indent=2) + "\n")
            if record["status"] != "VERIFIED":
                save("BLOCKED", reason="ModelScope base checkpoint is not fully downloaded and SHA-256 verified")
                return 2
        env = process_environment(gpu_ids)
        env.update(PATH=str(interpreter("train", required=not dry_run).parent) + os.pathsep + env["PATH"],
                   NO_ALBUMENTATIONS_UPDATE="1", TOKENIZERS_PARALLELISM="false",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled",
                   OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "4"),
                   TMPDIR=str(ROOT / "runtime/tmp"), HF_HOME=str(ROOT / "runtime/cache/huggingface"))
        save("DRY_RUN_RUNNING" if dry_run else "TRAINING")
        with (out / "stdout.log").open("w") as log:
            result = subprocess.run(command, cwd=POLICY, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            save("FAILED", exit_code=result.returncode, reason="Training command failed; see stdout.log")
            return result.returncode
        if dry_run:
            save("DRY_RUN", reason="Recipe resolved only; no optimizer step executed")
        else:
            save("COMPLETED", validation=validate(checkpoint, steps))
        print(json.dumps(manifest, indent=2))
        return 0
    except Exception as exc:
        save("FAILED", reason=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
