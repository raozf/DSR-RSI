"""Validate a completed training run before allowing its checkpoint into evaluation."""

import argparse
import json
import math
from pathlib import Path


def validate(directory, minimum_steps=10):
    directory = Path(directory)
    state = json.loads((directory / "trainer_state.json").read_text())
    config = json.loads((directory / "training_config.json").read_text())
    step = int(state["global_step"])
    if step < minimum_steps or config["global_step"] != step:
        raise ValueError(f"Incomplete or inconsistent optimizer step: {step}")
    losses = [row[key] for row in state["log_history"]
              for key in ("loss", "train_loss") if key in row]
    if not losses or not all(math.isfinite(float(loss)) for loss in losses):
        raise ValueError("Training has no finite recorded loss, or contains nonfinite loss")
    for name in ("norm_stats.json", "tokenizer.json", "processor_config.json"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    adapter = (directory / "adapter_config.json").is_file()
    if config.get("experiment", {}).get("use_lora") and not adapter:
        raise ValueError("LoRA training did not produce adapter_config.json")
    if adapter:
        adapter_config = json.loads((directory / "adapter_config.json").read_text())
        base = adapter_config.get("base_model_name_or_path")
        if not base or not Path(base).is_dir():
            raise ValueError("LoRA checkpoint must reference an available local base model")
    if not adapter and not (directory / "config.json").is_file():
        raise FileNotFoundError("config.json or adapter_config.json")
    index = directory / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif adapter and (directory / "adapter_model.safetensors").exists():
        names = ["adapter_model.safetensors"]
    else:
        names = ["model.safetensors"]
    from safetensors import safe_open
    files = []
    for name in names:
        path = directory / name
        # Validate the same safetensors header/offsets without loading PyTorch
        # or materializing tensors. This check belongs in the CPU orchestrator.
        with safe_open(str(path), framework="numpy") as weights:
            tensors = len(list(weights.keys()))
        if not tensors:
            raise ValueError(f"Empty weights: {path}")
        files.append({"name": name, "bytes": path.stat().st_size, "tensors": tensors})
    return {"checkpoint": str(directory.resolve()), "global_step": step,
            "adapter": adapter,
            "recorded_losses": losses, "weights": files,
            "dataset": config["dataset"], "global_batch_size": config["global_batch_size"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--minimum-steps", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(validate(args.checkpoint, args.minimum_steps), indent=2))
