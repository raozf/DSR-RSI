"""Train DM0.5 LoRA on demonstrations plus verified native agent experience."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

from build_agent_training_dataset import build

from dsr.paths import repository_root
from dsr.runtime_config import interpreter, process_environment
from dsr.robodojo.accuracy_output import load_json, read_evaluation_checkpoint
ROOT = repository_root()
POLICY = ROOT / "vendor/XPolicyLab/policy/OpenDM"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-run", type=Path, required=True, help="Completed native adapt run root")
    parser.add_argument("--state-root", type=Path, help="Operational state root used by the agent evaluation")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("dsrrsi_%Y%m%dT%H%M%S"))
    parser.add_argument("--base-data", type=Path, default=POLICY / "data/RoboDojo-robodojo-mem-arx_x5-joint-full")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) or args.steps < 1:
        parser.error("run-id must be a simple directory name and steps must be positive")
    gpus = [int(x) for x in args.gpus.split(",")]
    if len(set(gpus)) != len(gpus) or args.global_batch % len(gpus):
        parser.error("GPU IDs must be unique; global batch must be divisible by GPU count")
    agent_run = args.agent_run.resolve()
    checkpoint = read_evaluation_checkpoint(agent_run, args.state_root)
    manifest, status = checkpoint['manifest'] or {}, checkpoint['status'] or {}
    accuracy = load_json(agent_run / "task_accuracy.json")
    if (manifest.get("policy_name") != "DSRRSI" or manifest.get("agent_mode") != "adapt"
            or status.get("status") != "COMPLETED"
            or ((agent_run / "task_accuracy.json").exists() and (
                not isinstance(accuracy, dict) or accuracy.get('status') != 'COMPLETED'
                or accuracy.get('run') != agent_run.name or accuracy.get('coverage_issues')))):
        raise ValueError("Training requires a completed native DSRRSI adapt run")
    base_manifest = json.loads((args.base_data / "dataset.json").read_text())
    if base_manifest.get("selected_task_indices") is not None:
        raise ValueError("Full training cannot use the smoke task subset")
    output = ROOT / "results/dsr_rsi_training" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "dataset"
    if not (args.resume and (dataset / "dataset.json").is_file()):
        build(args.base_data, agent_run, dataset)
    dataset_manifest = json.loads((dataset / "dataset.json").read_text())
    if dataset_manifest["agent_experience_root"] != str(agent_run):
        raise ValueError("Cannot resume training with different agent experience")
    env = process_environment(args.gpus)
    env.update(
        DSR_TRAIN_OUT=str(output), DSR_TRAIN_CHECKPOINT_NAME=args.run_id,
        DSR_TRAIN_DATA_ROOT=str(dataset), DSR_TRAIN_STEPS=str(args.steps),
        DSR_TRAIN_SAVE_STEPS=str(min(1000, args.steps)), DSR_TRAIN_GLOBAL_BATCH=str(args.global_batch),
        DSR_TRAIN_SEED=str(args.seed), DSR_TRAIN_GPU_IDS=args.gpus,
        DSR_TRAIN_LORA="1", DSR_DRY_RUN="1" if args.dry_run else "0",
        DSR_TRAIN_SCOPE="DM0.5 LoRA on released demonstrations + verified native DSR-RSI repairs",
    )
    record = {
        "agent_run": str(agent_run), "adaptation_seeds": manifest["seeds"],
        "checkpoint": str(POLICY / f"checkpoints/RoboDojo-{args.run_id}-arx_x5-joint-{args.seed}"),
        "memory_root": str(agent_run / "tasks" / f"seed_{manifest['seeds'][0]}"),
        "verified_repair_episodes": dataset_manifest["verified_repair_episodes"],
        "coding_model_weights_trained": False,
        "policy_weights_trained": "LoRA; completion is recorded in training_manifest.json",
    }
    (output / "agent_training.json").write_text(json.dumps(record, indent=2) + "\n")
    command = [str(interpreter("train", required=not args.dry_run)), str(ROOT / "tools/train_dm05_local.py")]
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
