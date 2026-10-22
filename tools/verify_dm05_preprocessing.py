"""Compare inference preprocessing to the retained upstream Albumentations pipeline."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import numpy as np
import cv2
import albumentations


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NoAugmentationPipeline()


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--observation", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
directory = root / "vendor/XPolicyLab/policy/OpenDM/opendm/opendm/data"
reference = load(directory / "augmentations.py", "original_augmentations")
adapted = load(directory / "inference_images.py", "adapted_images")
rng = np.random.default_rng(20261011)
inputs = {
    str(shape): rng.integers(0, 256, shape, dtype=np.uint8)
    for shape in [(1, 1, 3), (3, 8, 3), (8, 3, 3), (7, 9, 3),
                  (480, 640, 3), (640, 480, 3), (448, 448, 3)]
}
if args.observation:
    with np.load(args.observation, allow_pickle=False) as data:
        inputs.update({key: data[key] for key in
                       ("cam_high", "cam_left_wrist", "cam_right_wrist")})
cases = []
for name, value in inputs.items():
    expected = reference(image=value)["image"]
    actual = adapted(image=value)["image"]
    cases.append({"input": name, "shape": list(value.shape),
                  "exact_equal": bool(np.array_equal(actual, expected)),
                  "max_absolute_error": int(np.abs(actual.astype(int) - expected.astype(int)).max())})
result = {"status": "passed" if all(x["exact_equal"] for x in cases) else "failed",
          "versions": {"opencv": cv2.__version__, "numpy": np.__version__,
                       "albumentations": albumentations.__version__},
          "code_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [directory / "augmentations.py", directory / "inference_images.py"]},
          "cases": cases}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["status"] == "passed" else 1)
