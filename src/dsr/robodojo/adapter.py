"""Native RoboDojo episode execution, with lazy Isaac/array imports."""

from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
import json
import math
import shutil
import time
from typing import Any, Mapping

from dsr.system1.actor import System1Actor
from dsr.system1.dm05_policy import DM05Policy, DM05SkillRegistry
from dsr.system2.codex_backend import write_json
from dsr.harness.runtime import native_harness_prompt, normalize_native_harness


def compact(value):
    if isinstance(value, Mapping):
        return {str(k): compact(v) for k, v in value.items() if k != "vision"}
    if isinstance(value, (list, tuple)):
        return [compact(x) for x in value]
    if hasattr(value, "tolist"):
        return compact(value.tolist())
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def instruction_of(observation):
    value = observation.get("instruction", observation.get("instructions"))
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Native observation has no task instruction")
    return value


def save_images(observation, directory, prefix):
    from PIL import Image
    import numpy as np
    from dsr.paths import bootstrap_policy
    bootstrap_policy()
    from XPolicyLab.process_data import decode_obs_images

    observation = {**observation, "vision": {
        key: dict(value) for key, value in observation["vision"].items()
    }}
    decode_obs_images(observation)
    paths = []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        data = observation["vision"][camera]
        image = data.get("color", data.get("colors"))
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError(f"Invalid native RGB observation for {camera}")
        path = directory / f"{prefix}_{camera}.jpg"
        Image.fromarray(array).save(path, quality=95)
        paths.append(str(path.resolve()))
    return paths


class RoboDojoAdapter:
    def __init__(self, root: str | Path, *, env=None, client=None):
        self.root = Path(root)
        self.env, self.client = env, client
        if env is not None and int(env.num_envs) != 1:
            raise ValueError("DSR requires one native environment per worker")
        self.initial_observation = None

    def check(self) -> dict[str, Any]:
        return {
            "status": "READY" if self.env is not None and self.client is not None else "NOT_ATTACHED",
            "root": str(self.root),
            "assets_exist": self.root.exists(),
            "reason": None if self.env is not None else "Launch run_rsi.py --backend robodojo to attach Isaac Sim",
        }

    def preview(self):
        from XPolicyLab.policy.OpenDM.deploy import _get_valid_obs
        if self.env is None:
            raise RuntimeError("No native environment attached; use run_rsi.py --backend robodojo")
        if self.initial_observation is None:
            self.initial_observation = _get_valid_obs(self.env)
        return self.initial_observation

    @staticmethod
    def validate_harness(harness):
        """Validate declarative roles without contacting Isaac or DM0.5."""

        from dsr.system1.dm05_policy import DM05SkillRegistry

        normalized = normalize_native_harness(harness)
        DM05SkillRegistry(None).validate_harness(normalized)

    def identity(self):
        return {
            "task": str(self.env.eval_cfg["task_name"]),
            "seed": int(self.env.eval_cfg["seed"]),
            "layout_id": int(self.env.env_seeds[0]),
        }

    def run_episode(self, harness, *, output_dir, retry=False, record_training=False, failure_type=None):
        """Execute real motor actions. A retry resets the same saved layout."""
        from XPolicyLab.policy.OpenDM.deploy import _get_valid_obs
        from XPolicyLab.utils.process_data import pack_robot_state

        if self.env is None or self.client is None:
            raise RuntimeError("No native environment/client attached; use the RoboDojo launcher")
        harness = normalize_native_harness(harness)
        self.validate_harness(harness)
        action_steps = harness["action_steps"]
        if type(action_steps) is not int or not 1 <= action_steps <= 25:
            raise ValueError("Invalid executed action horizon")
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        identity = self.identity()
        if retry:
            self.env.reset(seed=[identity["layout_id"]])
            self.env.run_reward()
            if hasattr(self.env, "get_score"):
                self.env.get_score()
            self.initial_observation = None
        observation = self.preview()
        original_instruction = instruction_of(observation)
        prompt_patch = native_harness_prompt(harness)
        scope = str(self.env.deploy_cfg.get("evaluation_id", self.env.run_id)) + ":" + directory.name
        policy = DM05Policy(self.client, scope, action_steps=action_steps)
        registry = DM05SkillRegistry(policy)
        registry.validate_harness(harness)
        actor = System1Actor(policy, harness=harness, skills=registry)
        if actor.controller is not None:
            actor.controller.failure_type = failure_type
        # Keep the original goal as the immutable task identity.  The
        # controller supplies the current stage, memory and recovery rule on
        # every System-1 call; baking the patch into ``state.task`` would make
        # the harness impossible to audit separately.
        actor.reset(original_instruction)
        from dsr.evidence import EvidenceRecorder
        recorder = EvidenceRecorder(directory / 'evidence', identity=identity,
                                    attempt=2 if retry else 1)
        recorder.bind_plan(harness, {'id': 'dm05', 'evaluation_id': scope})
        recorder.observe(observation)
        evidence_stage = None
        samples, trace, steps = [], [], 0
        started = time.monotonic()
        images = save_images(observation, directory / "images", "start")
        dims = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
        write_json(directory / "request.json", {
            **identity, "harness": harness, "retry": retry,
            "harness_prompt": prompt_patch,
            "system1_protocol": "physicalrsi_system1_episode_binding/v1",
        })
        try:
            training_context = (directory / "training.jsonl").open("w") if record_training else nullcontext()
            with (directory / "trace.jsonl").open("w") as stream, training_context as training:
                while not self.env.is_episode_end():
                    if actor.should_terminate(observation):
                        break
                    action = actor.act(observation)
                    stage = actor.controller.stage if actor.controller is not None else None
                    if stage is not None and stage != evidence_stage:
                        # This is the actual harness stage, not a visually
                        # verified manipulation outcome or task success.
                        recorder.record_stage(stage, observed={
                            'controller_stage': stage,
                            'selected_skill': actor.state.selected_skill,
                        })
                        evidence_stage = stage
                    self.env.validate_action_dict(action)
                    row = {
                        "step": steps, "observation": compact(observation),
                        "action": compact(action),
                    }
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    # Full 25Hz state/action pairs, not inferred action chunks.
                    if record_training:
                        frame_paths = save_images(observation, directory / "frames", f"{steps:06d}")
                        data = {
                            f"images_{i+1}": {"type": "image", "url": str(Path(path).relative_to(directory))}
                            for i, path in enumerate(frame_paths)
                        }
                        data.update(
                            state=pack_robot_state(observation, "joint", dims).tolist(),
                            action=pack_robot_state({"state": action}, "joint", dims).tolist(),
                            # The controller may update stage, memory and
                            # recovery guidance between action chunks.  Store
                            # exactly the instruction sent to DM05 for this
                            # frame instead of a stale episode-level string.
                            prompt=actor.current_instruction, is_robot=True,
                            task_name="robodojo_" + identity["task"],
                        )
                        training.write(json.dumps(data, allow_nan=False) + "\n")
                    self.env.take_action(action)
                    recorder.executed(action)
                    steps += 1
                    if not self.env.is_episode_end():
                        observation = _get_valid_obs(self.env)
                        actor.after_action(observation)
                        recorder.observe(observation)
                    if steps % 25 == 0:
                        samples.append(row)
                        write_json(directory.parents[2] / "progress.json", {
                            **identity, "episode_id": directory.parent.name,
                            "phase": directory.name, "steps": steps,
                        })
                    trace = (trace + [row])[-4:]
                    if self.env.is_episode_end():
                        break
                if 0 in self.env.unstable_envs:
                    raise RuntimeError("Unstable native episode cannot be used as agent evidence")
                final_observation = self.env.get_obs_batch([0], last_frame=True)[0]
                recorder.observe(final_observation)
                images += save_images(final_observation, directory / "images", "end")
                success = bool(self.env.success[0])
                score = (1.0 if success else float(self.env.reward_manager.get_score()[0]) / 100
                         if hasattr(self.env, "get_score") else 0.0)
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid native episode score")
            result = {
                **identity, "status": "COMPLETED", "backend": "native_robodojo_dm05",
                "success": success, "score": score, "steps": steps, "policy_calls": policy.calls,
                "wall_seconds": time.monotonic() - started, "harness": harness,
                "instruction": original_instruction, "images": images,
                "trace_path": str(directory / "trace.jsonl"),
                "trace_summary": [samples[round(i * (len(samples) - 1) / 11)] for i in range(12)] + trace
                if len(samples) > 12 else samples + trace,
                "training_jsonl": str(directory / "training.jsonl") if record_training and success else None,
                "training_image_root": str(directory),
                "recorded_frames": steps if record_training and success else 0,
                "harness_runtime": actor.audit(),
                "evidence": recorder.finalize(outcome={'success':success, 'score':score}),
            }
            if record_training and not success:
                (directory / "training.jsonl").unlink(missing_ok=True)
                shutil.rmtree(directory / "frames", ignore_errors=True)
            self.env.save_video(0, str(directory / "rollout.mp4"), "success" if success else "fail")
            write_json(directory / "result.json", result)
            return result
        except BaseException as exc:
            write_json(directory / "result.json", {
                **identity, "status": "FAILED", "steps": steps,
                "error": f"{type(exc).__name__}: {exc}",
                "evidence": recorder.finalize(outcome={'status':'FAILED','error':str(exc)}),
            })
            raise
        finally:
            policy.reset()
            self.initial_observation = None
