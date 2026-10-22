"""Run frozen skill-library bundles through the actual DM0.5 RoboDojo launcher.

This port measures one preregistered case, checks its native outcome and loaded
library, and writes a MetaEvaluator receipt. It cannot invent missing stability
measurements, select a survivor, or promote an untested controller.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from dsr.paths import bootstrap, repository_root
bootstrap()
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json, identifier
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness


ROOT = repository_root()
CHECKS = ("native_task_success", "physical_actions")
SOURCE_SUFFIXES = frozenset({".py", ".pyi", ".sh", ".yaml", ".yml", ".toml", ".json"})
SOURCE_EXCLUDES = frozenset({".git", "__pycache__", ".pytest_cache", "Assets", "assets",
                             "checkpoints", "eval_result", "node_modules"})

# These fields are the public proprioception contract used by the native
# observer.  They are deliberately kept local to the verifier: a trajectory
# report may describe the robot's motion, but it cannot infer object motion or
# task completion from that motion.
_ROBOT_STATE_FIELDS = frozenset({
    "arm_joint_state", "ee_joint_state", "ee_pose", "tcp_pose", "delta_ee_pose",
    "left_arm_joint_state", "right_arm_joint_state", "left_ee_joint_state",
    "right_ee_joint_state", "left_ee_pose", "right_ee_pose", "left_tcp_pose",
    "right_tcp_pose", "left_delta_ee_pose", "right_delta_ee_pose",
})


def source_files(roots):
    """Hash local executable/configuration trees, including indirect imports.

    Installed third-party binaries and large model/scene assets are external
    deployment dependencies, not a claim of a hermetic environment. JSON
    configuration/memory files in code roots are pinned alongside their code;
    generated run/evidence directories must stay outside those import roots.
    """
    roots = sorted({Path(root).resolve() for root in roots}, key=lambda path: (len(path.parts), str(path)))
    selected = []
    for root in roots:
        if not root.is_dir():
            raise ValueError("Executable import root is missing: " + str(root))
        if not any(root.is_relative_to(parent) and not (set(root.relative_to(parent).parts) & SOURCE_EXCLUDES)
                   for parent in selected):
            selected.append(root)
    files = {}
    for root in selected:
        for directory, names, filenames in os.walk(root):
            names[:] = sorted(name for name in names if name not in SOURCE_EXCLUDES)
            for name in sorted(filenames):
                path = Path(directory) / name
                if path.suffix in SOURCE_SUFFIXES:
                    files[str(path)] = file_digest(path)
    return files


def library_foundation(config):
    return {"schema": "dsr.native-foundation/v1",
            "trained_skills": {key: value for key, value in config["skills"].items()
                               if value.get("implementation") in {"dm05", "pi05", "pi05-sparse-memory"}},
            "model_substitution": config.get("model_substitution"),
            "robot": "dual_arx_x5", "observation": "robodojo.observation-batch/v1",
            "action": "robodojo.action-chunks/v1"}


def native_case(task, seed, layout_id, *, split):
    identifier(task)
    if (type(seed) is not int or seed not in (0, 1, 2)
            or type(layout_id) is not int or layout_id < 0
            or split not in {"development", "validation", "meta_holdout"}):
        raise ValueError("Invalid native case identity")
    directory = ROOT / "vendor/RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5" / str(seed)
    paths = sorted((path for path in directory.glob(task + "_*.json")
                    if path.stem[len(task) + 1:].isdigit()),
                   key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if layout_id >= len(paths):
        raise ValueError("Native layout is unavailable")
    return {"id": task + "-s" + str(seed) + "-l" + str(layout_id), "task": task,
            "seed": seed, "layout_id": layout_id, "split": split,
            "layout_sha256": file_digest(paths[layout_id])}


@contextmanager
def gpu_pair_lock(state, indices):
    # Different experiment state roots still share the same physical GPUs.
    # This process-local filesystem location is independent of TMPDIR and each
    # campaign's private caches; all native verifiers for this user cooperate.
    directory = Path("/tmp") / ("dsr-native-gpu-locks-" + str(os.getuid()))
    directory.mkdir(parents=True, exist_ok=True)
    streams = []
    try:
        for index in sorted(indices):
            stream = (directory / ("gpu-" + str(index) + ".lock")).open("a")
            streams.append(stream)
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for stream in streams:
            stream.close()


def dependency_files(library):
    """Pin declared files plus installed source trees exposed to code workers."""
    files, configurations = {}, []
    def visit(value):
        if isinstance(value, dict):
            if value.get("runtime") == "frozen-program-service":
                configurations.append(value)
            for key, child in value.items():
                if key == "files" and isinstance(child, dict):
                    for name, expected in child.items():
                        path = Path(name)
                        if (not path.is_absolute() or file_digest(path) != expected
                                or name in files and files[name] != expected):
                            raise ValueError("Code dependency changed: " + name)
                        files[name] = expected
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(library)
    if not files:
        raise ValueError("Native library must pin executable dependencies")
    roots = set()
    for config in configurations:
        command = config.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(item, str) for item in command):
            raise ValueError("Explicit executable command required")
        for item in command:
            path = Path(item)
            if path.suffix in SOURCE_SUFFIXES and "{" not in item:
                if not path.is_absolute() or str(path) not in files:
                    raise ValueError("Command source or descriptor is outside declared dependencies: " + item)
                # A descriptor/configuration file is pinned individually, but
                # its containing evidence directory is not an import root.
                # Candidate descriptors live beside requests and frozen bundle
                # metadata; recursively hashing that directory would make the
                # dependency closure include its own generated manifests.
                if path.suffix in {".py", ".pyi", ".sh"}:
                    roots.add(path.parent)
        if "--descriptor" not in command:
            continue
        def argument(flag):
            if flag not in command or command.index(flag) + 1 >= len(command):
                raise ValueError("Code worker command is missing " + flag)
            return command[command.index(flag) + 1]
        descriptor = read_json(Path(argument("--descriptor")))
        assets = Path(argument("--assets")).resolve()
        framework_flag = "--framework-root" if "--framework-root" in command else "--xpolicylab"
        framework = Path(argument(framework_flag)).resolve()
        variables = {"assets": str(assets), "xpolicylab": str(framework),
                     "xpolicylab_parent": str(framework.parent), "python": command[0]}
        def expand(value):
            for key, replacement in variables.items():
                value = value.replace("{" + key + "}", replacement)
            if "{" in value or "}" in value or not Path(value).is_absolute():
                raise ValueError("Unresolved executable dependency: " + value)
            return Path(value)
        implementation = descriptor.get("implementation", {})
        source = expand(implementation["source"])
        if files.get(str(source)) != implementation.get("sha256"):
            raise ValueError("Descriptor implementation is outside declared dependencies")
        # The worker also imports sibling controller modules, shared skills,
        # transports and perception services through these actual search paths.
        for owner in [descriptor, *descriptor.get("services", [])]:
            roots.update(expand(path) for path in owner.get("pythonpath", []))
            if owner.get("cwd"):
                roots.add(expand(owner["cwd"]))
    extra = source_files(roots)
    if any(name in files and files[name] != sha for name, sha in extra.items()):
        raise ValueError("Executable dependency changed during closure collection")
    files.update(extra)
    return files


def freeze_library(library, output, *, name):
    """Package the actual launcher configuration in the existing F/H closure.

    Large installed assets remain external, with dependencies checked before
    and after execution. DM0.5 weights undergo the launcher's checkpoint audit.
    """
    config = read_json(Path(library))
    files = dependency_files(config)
    if not config.get("program_registry") or not config.get("skills"):
        raise ValueError("An executable skill library is required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "skill_library.json", config)
    atomic_json(output / "dependencies.json", files)
    # F must stay invariant when H changes: never hash candidate code/routing
    # into foundation, or SelfHarness would reject every real controller edit.
    foundation = library_foundation(config)
    atomic_json(output / "foundation.json", foundation)
    components = {kind: {"skill_library.json": file_digest(output / "skill_library.json")}
                  for kind in CLOSURE}
    for kind in ("foundation", "weights"):
        components[kind] = {"foundation.json": file_digest(output / "foundation.json")}
    components["dependencies"]["dependencies.json"] = file_digest(output / "dependencies.json")
    bundle = {"id": name, "root": str(output), "components": components,
              "qualification": None, "scope": "pending_native_library_evaluation"}
    verify_harness(bundle)
    atomic_json(output / "bundle.json", bundle)
    return bundle


def library_from_bundle(bundle):
    verify_harness(bundle)
    root = Path(bundle["root"])
    if ("skill_library.json" not in bundle["components"]["skills"]
            or "dependencies.json" not in bundle["components"]["dependencies"]
            or any("foundation.json" not in bundle["components"][kind] for kind in ("foundation", "weights"))):
        raise ValueError("Native library and dependencies must belong to frozen closure")
    library = read_json(root / "skill_library.json")
    if library_foundation(library) != read_json(root / "foundation.json"):
        raise ValueError("Native foundation differs from the executable trained-skill configuration")
    if dependency_files(library) != read_json(root / "dependencies.json"):
        raise ValueError("Native library dependency closure changed")
    return library


def deployed_library(library, output):
    """Relocate evidence directories; keep all execution and memory settings."""
    result = deepcopy(library)
    for collection in (result["skills"], result["program_registry"]):
        for name, entry in collection.items():
            settings = entry.get("configuration", {})
            if settings.get("runtime") == "frozen-program-service":
                settings["output"] = str(Path(output) / "code_skills" / digest(name)[:16])
    return result


def _evidence_files(directory):
    """Return hashes for an episode's passive evidence tree.

    The directory is trusted only after every file is hashed and all paths are
    contained by the episode.  Generated reports outside this tree are not
    silently included in the evidence claim.
    """
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Native trajectory evidence must not contain symbolic links")
    directory = directory.resolve()
    if not directory.is_dir():
        return {}
    files = {}
    for path in sorted(directory.rglob("*")):
        # Hash only regular files physically inside the evidence root.  A
        # symlink could otherwise make a receipt claim bytes from an unrelated
        # directory while its lexical path still looks contained.
        if path.is_symlink():
            raise ValueError("Native trajectory evidence must not contain symbolic links")
        resolved = path.resolve()
        if not resolved.is_relative_to(directory):
            raise ValueError("Native trajectory evidence escapes its root")
        if resolved.is_file():
            files[str(resolved)] = file_digest(resolved)
    return files


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def native_trajectory_metrics(episode, *, task, seed, layout_id, report, trace):
    """Measure only claims supported by the recorded native trajectory.

    ``status`` is ``unknown`` when the passive evidence recorder was not
    present.  A malformed recorder output is rejected because it cannot be
    used as evidence.  The motion summary is a robot end-effector diagnostic;
    it is intentionally not called object stability or task success.
    """
    episode = Path(episode).resolve()
    evidence_dir = episode / "evidence"
    manifest_path = evidence_dir / "manifest.json"
    events_path = evidence_dir / "events.jsonl"
    evidence = _evidence_files(evidence_dir)
    if not manifest_path.exists() or not events_path.exists():
        return {"schema": "dsr.native-trajectory/v1", "status": "unknown",
                "reason": "passive evidence manifest or event log unavailable",
                "episode_id": report.get("episode_id"), "evidence": {}}
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != "dsr.attempt-evidence/v1":
        raise ValueError("Native trajectory manifest schema is invalid")
    identity = manifest.get("identity")
    episode_id = report.get("episode_id")
    if (not isinstance(episode_id, str) or not episode_id.strip()
            or episode_id != episode.name):
        raise ValueError("Native trajectory report episode identity is invalid")
    expected_identity = {"task": task, "seed": seed, "layout_id": layout_id,
                         "episode_id": episode_id}
    first = report.get("first_attempt", {})
    if (not isinstance(identity, dict) or not isinstance(first, dict)
            or any(type(identity.get(key)) is not type(value) or identity.get(key) != value
                   or type(first.get(key)) is not type(value) or first.get(key) != value
                   for key, value in expected_identity.items())):
        raise ValueError("Native trajectory evidence is not bound to the episode identity")
    attempt = manifest.get("attempt")
    if type(attempt) is not int or attempt != 1:
        raise ValueError("Native trajectory manifest attempt is invalid")
    if type(first.get("steps")) is not int or first["steps"] < 1:
        raise ValueError("Native trajectory reported execution count is invalid")
    events = []
    previous_timestamp, steps, observations, last_observation_step = 0, 0, 0, None
    with events_path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Native trajectory event log is invalid") from exc
            if not isinstance(row, dict) or row.get("ref") != f"event_{line_number:06d}":
                raise ValueError("Native trajectory event references are not contiguous")
            if type(row.get("attempt")) is not int or row["attempt"] != attempt:
                raise ValueError("Native trajectory event attempt is not bound")
            step, observation_index, timestamp = (row.get("step"), row.get("observation_index"),
                                                  row.get("timestamp"))
            if (type(step) is not int or step < 0 or type(observation_index) is not int
                    or observation_index < 0 or not _finite_number(timestamp) or timestamp < 0):
                raise ValueError("Native trajectory event chronology is invalid")
            kind = row.get("kind")
            if not isinstance(kind, str) or not kind:
                raise ValueError("Native trajectory event kind is invalid")
            if kind == "executed_action":
                steps += 1
                age = None if last_observation_step is None else steps - last_observation_step
                if (row.get("observation_step") != last_observation_step
                        or row.get("observation_age_steps") != age):
                    raise ValueError("Native trajectory action observation age is invalid")
            elif kind == "observation":
                observations += 1
                last_observation_step = steps
                if not isinstance(row.get("state"), dict):
                    raise ValueError("Native trajectory observation state is invalid")
            if timestamp < previous_timestamp or step != steps or observation_index != observations:
                raise ValueError("Native trajectory event chronology is out of order")
            previous_timestamp = timestamp
            events.append(row)
    action_events = [row for row in events if row.get("kind") == "executed_action"]
    observation_events = [row for row in events if row.get("kind") == "observation"]
    if len(action_events) != first["steps"]:
        raise ValueError("Native trajectory evidence differs from reported execution count")
    action_steps = [row.get("step") for row in action_events]
    if action_steps != list(range(1, len(action_events) + 1)):
        raise ValueError("Native trajectory action steps are not contiguous")
    observation_steps = [row.get("step") for row in observation_events]
    if any(type(step) is not int or step < 0 for step in observation_steps):
        raise ValueError("Native trajectory observation steps are invalid")
    if any(after < before for before, after in zip(observation_steps, observation_steps[1:])):
        raise ValueError("Native trajectory observations are out of order")
    terminal = [row for row in events if row["kind"] == "attempt_end"]
    if len(terminal) != 1 or terminal[0] is not events[-1] or terminal[0].get("outcome") != first:
        raise ValueError("Native trajectory terminal outcome differs from the first attempt")
    summary = manifest.get("summary", {})
    if (not isinstance(summary, dict) or type(summary.get("events_seen")) is not int
            or summary["events_seen"] != len(events)):
        raise ValueError("Native trajectory manifest event count is incomplete")

    # Reconcile the passive executed-action rows with the physical trace.  The
    # trace is the exact action sent to the simulator; event metadata alone is
    # insufficient if a recorder was tampered with after execution.
    trace = Path(trace)
    if trace.is_symlink() or trace.resolve() != episode / "trace.jsonl":
        raise ValueError("Native physical action trace is outside the episode")
    trace_sha256 = file_digest(trace)
    trace_actions = []
    try:
        with trace.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                action = json.loads(line)
                if (not isinstance(action, dict) or type(action.get("step")) is not int
                        or action["step"] != line_number
                        or "action" not in action):
                    raise ValueError("Native physical action trace is incomplete or out of order")
                trace_actions.append(action["action"])
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Native physical action trace is invalid") from exc
    if len(trace_actions) != len(action_events):
        raise ValueError("Native trajectory action trace differs from evidence actions")
    for index, (event, traced) in enumerate(zip(action_events, trace_actions), start=1):
        try:
            same = ("action" in event and
                    json.dumps(event["action"], sort_keys=True, allow_nan=False) ==
                    json.dumps(traced, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("Native trajectory contains an invalid physical action") from exc
        if not same:
            raise ValueError("Native trajectory action differs from physical trace at step " + str(index))
    keyframes = manifest.get("keyframes", [])
    if not isinstance(keyframes, list):
        raise ValueError("Native trajectory keyframe index is invalid")
    events_by_ref = {row["ref"]: row for row in events}
    frame_refs, frame_paths = set(), set()
    for frame in keyframes:
        if not isinstance(frame, dict) or not isinstance(frame.get("path"), str):
            raise ValueError("Native trajectory keyframe reference is invalid")
        ref, event_ref = frame.get("ref"), frame.get("event_ref")
        if (not isinstance(ref, str) or ref in frame_refs or not isinstance(event_ref, str)
                or event_ref not in events_by_ref):
            raise ValueError("Native trajectory keyframe event reference is invalid")
        event = events_by_ref[event_ref]
        if any(frame.get(key) != event.get(key) for key in ("step", "observation_index", "timestamp")):
            raise ValueError("Native trajectory keyframe event identity is invalid")
        frame_path = (evidence_dir / frame["path"]).resolve()
        if (not frame_path.is_relative_to(evidence_dir) or str(frame_path) not in evidence
                or frame_path in frame_paths or evidence[str(frame_path)] != frame.get("sha256")):
            raise ValueError("Native trajectory keyframe differs from its manifest")
        frame_refs.add(ref)
        frame_paths.add(frame_path)
    if _evidence_files(evidence_dir) != evidence or file_digest(trace) != trace_sha256:
        raise ValueError("Native trajectory evidence changed during measurement")
    evidence[str(trace)] = trace_sha256
    if not observation_events:
        return {"schema": "dsr.native-trajectory/v1", "status": "unknown",
                "reason": "no policy-visible observation events", "episode_id": report.get("episode_id"),
                "evidence": evidence}
    state_events = [row for row in observation_events
                    if isinstance(row.get("state"), dict)
                    and set(row["state"]) & _ROBOT_STATE_FIELDS]
    pose_channels = {}
    invalid_pose_samples = 0
    pose_fields = ("ee_pose", "tcp_pose", "left_ee_pose", "right_ee_pose",
                   "left_tcp_pose", "right_tcp_pose")
    for row in observation_events:
        state = row.get("state")
        if not isinstance(state, dict):
            continue
        for key in pose_fields:
            if key not in state:
                continue
            value = state[key]
            valid = (isinstance(value, (list, tuple)) and len(value) == 7
                     and all(_finite_number(item) for item in value))
            if not valid:
                invalid_pose_samples += 1
                continue
            pose_channels.setdefault(key, []).append(tuple(float(item) for item in value))
    if invalid_pose_samples:
        raise ValueError("Native trajectory end-effector pose has invalid or non-finite samples")
    channel_metrics = {}
    max_step = None
    sample_count = 0
    for key, samples in sorted(pose_channels.items()):
        channel_step = None
        if len(samples) >= 2:
            channel_step = max(math.dist(before[:3], after[:3])
                               for before, after in zip(samples, samples[1:]))
            if not math.isfinite(channel_step):
                raise ValueError("Native trajectory end-effector displacement is non-finite")
            max_step = channel_step if max_step is None else max(max_step, channel_step)
        sample_count += len(samples)
        channel_metrics[key] = {"samples": len(samples), "finite": True,
                                "max_step_m": channel_step}
    return {"schema": "dsr.native-trajectory/v1", "status": "measured",
            "episode_id": report.get("episode_id"), "observation_events": len(observation_events),
            "action_events": len(action_events), "state_observation_events": len(state_events),
            "state_coverage": len(state_events) / len(observation_events),
            "end_effector": {"available": bool(pose_channels), "samples": sample_count,
                              "invalid_samples": invalid_pose_samples,
                              "finite": bool(pose_channels) and invalid_pose_samples == 0,
                              "max_step_m": max_step,
                              "channels": channel_metrics,
                              "interpretation": "robot end-effector motion diagnostic; object stability unknown"},
            "evidence": evidence}


def native_outcome(run, *, task, case, expected_library, operational_state=None):
    """Reconcile original official success with the actual bound episode."""
    run = Path(run)
    if operational_state is None:
        manifest = read_json(run / "manifest.json")
        state = read_json(run / "status.json")
        verification = read_json(run / "checkpoint_verification.json")
        library_path = run / "skill_library.json"
        protocol_files = [run / "status.json", run / "manifest.json",
                          run / "checkpoint_verification.json"]
    else:
        checkpoint_path = Path(operational_state) / "run.lock"
        checkpoint = read_json(checkpoint_path)
        manifest, state = checkpoint.get("manifest", {}), checkpoint.get("status", {})
        verification = manifest.get("checkpoint_verification", {})
        library_path = Path(operational_state) / "skill_library.json"
        accuracy = read_json(run / "task_accuracy.json")
        if (accuracy.get("status") != "COMPLETED" or accuracy.get("episodes_completed") != 1
                or accuracy.get("run") != run.name or manifest.get("run_id") != run.name
                or accuracy.get("coverage_issues") != []
                or [(r.get("task"), r.get("seed"), r.get("episodes"))
                    for r in accuracy.get("completed", [])] != [(task, case["seed"], 1)]):
            raise ValueError("Native accuracy output is incomplete")
        protocol_files = [checkpoint_path, run / "task_accuracy.json"]
    if (manifest.get("episode_retry") is not False or manifest.get("agent_mode") != "frozen"
            or manifest.get("task_names") != [task] or manifest.get("seeds") != [case["seed"]]
            or manifest.get("layout_ids") != [case["layout_id"]]
            or verification.get("status") != "VERIFIED"
            or read_json(library_path) != expected_library):
        raise ValueError("Native run protocol or deployed library differs from the pinned case")
    key = str(case["seed"]) + "/" + task
    if (state.get("status") != "COMPLETED" or set(state.get("results", {})) != {key}
            or state["results"][key].get("exit_code") != 0):
        raise ValueError("Native case did not complete; no capability score exists")
    directory = run / "tasks" / ("seed_" + str(case["seed"])) / task
    raw = read_json(directory / "raw_result.json")
    if raw.get("eval_time") != 1 or set(raw.get("details", {})) != {"0"}:
        raise ValueError("Exactly one complete native case is required")
    detail = raw["details"]["0"]
    if type(detail.get("success")) is not bool or detail.get("layout_id") != case["layout_id"]:
        raise ValueError("Native case identity or success is invalid")
    paths = list((directory / "agent/episodes").glob("*/report.json"))
    if len(paths) != 1:
        raise ValueError("Native episode evidence is missing or duplicated")
    report = read_json(paths[0])
    first = report.get("first_attempt", {})
    if (report.get("status") != "COMPLETED" or first.get("status") != "COMPLETED"
            or report.get("task") != task or report.get("seed") != case["seed"]
            or report.get("layout_id") != case["layout_id"]
            or first.get("success") is not detail["success"]
            or type(first.get("steps")) is not int or first["steps"] < 1
            or type(report.get("physical_rollouts")) is not int or report["physical_rollouts"] != 1
            or report.get("retry_attempted", False)
            or report.get("binding", {}).get("library_sha256") != digest(expected_library)):
        raise ValueError("Native outcome is not bound to this frozen library/first attempt")
    trace = paths[0].parent / "trace.jsonl"
    count = 0
    with trace.open() as stream:
        for count, line in enumerate(stream, start=1):
            action = json.loads(line)
            if type(action.get("step")) is not int or action["step"] != count or "action" not in action:
                raise ValueError("Native physical action trace is incomplete or out of order")
    if count != first["steps"]:
        raise ValueError("Native physical action trace differs from reported execution count")
    trajectory = native_trajectory_metrics(
        paths[0].parent, task=task, seed=case["seed"], layout_id=case["layout_id"],
        report=report, trace=trace)
    files = [*protocol_files, library_path, directory / "raw_result.json", paths[0], trace]
    evidence = {str(path): file_digest(path) for path in files}
    # Passive evidence is part of the receipt's hash closure when available.
    # A legacy run without the recorder remains a valid native outcome, but its
    # trajectory quality is explicitly unknown and cannot be used as a quality
    # gate by a future campaign.
    if any(name in evidence and evidence[name] != sha for name, sha in trajectory.get("evidence", {}).items()):
        raise ValueError("Native trajectory evidence changed before receipt collection")
    evidence.update(trajectory.get("evidence", {}))
    return {"native_success": detail["success"], "physical_steps": first["steps"],
            "checks": {"native_task_success": detail["success"], "physical_actions": True},
            "trajectory": trajectory, "evidence": evidence}


class NativeLibraryVerifier:
    """MetaEvaluator verifier port; uses unchanged native task success checkers.

    Only the registered checks in CHECKS are available. A proposed lift/release
    stability checker must be independently implemented and pinned first.
    """
    def __init__(self, *, state_root, policy_gpu, env_gpu, codex_home, timeout=2400,
                 python=None):
        if policy_gpu == env_gpu or min(policy_gpu, env_gpu) < 0 or not 120 <= timeout <= 7200:
            raise ValueError("Explicit disjoint GPU pair and bounded native timeout required")
        self.state_root = Path(state_root).resolve()
        self.policy_gpu, self.env_gpu = policy_gpu, env_gpu
        self.codex_home, self.timeout = str(Path(codex_home).resolve()), timeout
        self.python = str(Path(python or sys.executable).absolute())

    def identity(self):
        paths = [Path(__file__), ROOT / "tools/full_robodojo_agent_eval.py",
                 ROOT / "tools/verify_dm05_checkpoint.py", ROOT / "tools/check_dm05_training.py",
                 ROOT / "tools/configure_physicalrsi_dm05.py", ROOT / "src/dsr/runtime_config.py",
                 ROOT / "src/dsr/paths.py", ROOT / "src/dsr/__init__.py",
                 ROOT / "src/dsr/evidence.py", ROOT / "src/dsr/schema.py",
                 Path(self.codex_home) / "config.toml"]
        trees = [ROOT / name for name in (
            "src/dsr/system1", "src/dsr/system2", "src/dsr/robodojo",
            "src/dsr/agent", "src/dsr/harness", "src/dsr/diagnosis",
            "src/dsr/memory", "src/dsr/legacy_evolution",
            "vendor/RoboDojo/env", "vendor/RoboDojo/task", "vendor/RoboDojo/src",
            "vendor/RoboDojo/utils", "vendor/RoboDojo/env_cfg", "vendor/XPolicyLab",
            "vendor/PhysicalRSI/PhysicalRSI_core", "vendor/PhysicalRSI/PhysicalRSI_baselines/robodojo")]
        files = {**source_files(trees), **{str(path): file_digest(path) for path in paths}}
        # Values are hashes: do not record proxy/API credentials in receipts.
        environment = {key: digest(value) for key, value in os.environ.items()
                       if key.startswith(("DSR_", "ROBODOJO_")) and key not in {
                           "DSR_LOCAL_STATE", "DSR_KIT_ARGS", "DSR_CODEX_HOME"}}
        return {"kind": "dsr.native-library-verifier/v2", "checks": list(CHECKS),
                "files": files, "environment_sha256": environment,
                "policy_gpu": self.policy_gpu, "env_gpu": self.env_gpu,
                "python": self.python, "codex_home": self.codex_home, "timeout": self.timeout,
                "scope": "native_robodojo_first_attempt; source closure; external installed binaries/assets; no supplemental stability qualification"}

    def evaluate(self, *, descendant, task, case, required_checks, output):
        if (case.get("task") != task or case.get("split") not in {"development", "validation", "meta_holdout"}
                or type(case.get("seed")) is not int or case["seed"] not in (0, 1, 2)
                or type(case.get("layout_id")) is not int or case["layout_id"] < 0
                or not set(required_checks) <= set(CHECKS)):
            raise ValueError("Explicit native case and independently available checks required")
        expected = native_case(task, case["seed"], case["layout_id"], split=case["split"])
        if case.get("layout_sha256") != expected["layout_sha256"]:
            raise ValueError("Native layout changed or was not preregistered")
        frozen, identity = verify_harness(descendant), self.identity()
        library = library_from_bundle(descendant)
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        run_id = "native_" + digest(str(output))[:24]
        run = output / run_id
        run.mkdir(exist_ok=False)
        config = deployed_library(library, run)
        state_root = self.state_root / digest(str(output))[:20]
        state = state_root / run_id
        state.mkdir(parents=True, exist_ok=True)
        atomic_json(state / "skill_library.json", config)
        request = {"descendant_sha256": digest(descendant), "freeze_sha256": frozen,
                   "case": case, "library_sha256": digest(config), "verifier": identity,
                   "relocation": "only frozen-program-service output directories"}
        atomic_json(output / "request.json", request)
        command = [self.python, "-u", str(ROOT / "tools/full_robodojo_agent_eval.py"),
                   "--run-id", run_id, "--output-root", str(output), "--state-root",
                   str(state_root), "--skill-library", str(state / "skill_library.json"),
                   "--policy-name", "DSRRSI",
                   "--agent-strategy", "physicalrsi", "--agent-mode", "frozen",
                   "--codex-home", self.codex_home, "--policy-gpus", str(self.policy_gpu),
                   "--env-gpus", str(self.env_gpu), "--tasks", task, "--episodes", "1",
                   "--layout-ids", str(case["layout_id"]), "--seeds", str(case["seed"]),
                   "--attempts", "1", "--run-timeout", str(self.timeout),
                   "--task-timeout", str(self.timeout), "--stall-timeout", str(self.timeout)]
        env = os.environ.copy()
        runtime_state = self.state_root / digest(str(output))[:20] / "runtime"
        env["DSR_LOCAL_STATE"] = str(runtime_state)
        for key, relative in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                              ("TORCH_EXTENSIONS_DIR", "cache/torch_extensions"),
                              ("CUDA_CACHE_PATH", "cache/cuda")):
            target = runtime_state / relative
            target.mkdir(parents=True, exist_ok=True)
            env[key] = str(target)
        kit = runtime_state / "kit"
        env["DSR_KIT_ARGS"] = ("--portable-root " + str(kit)
                               + " --/UJITSO/datastore/localCachePath=" + str(kit / "DerivedDataCache"))
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "tools"),
                                           str(ROOT / "vendor/PhysicalRSI")])
        started = time.monotonic()
        with gpu_pair_lock(self.state_root, [self.policy_gpu, self.env_gpu]):
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=self.timeout + 120)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if code != 0:
            atomic_json(output / "infrastructure-failure.json", {"exit_code": code,
                        "case": case, "wall_seconds": time.monotonic() - started})
            raise RuntimeError("Native launcher failed: exit=" + str(code))
        if verify_harness(descendant) != frozen or self.identity() != identity:
            raise ValueError("Native candidate or verifier changed during execution")
        if native_case(task, case["seed"], case["layout_id"], split=case["split"]) != expected:
            raise ValueError("Native layout changed during execution")
        library_from_bundle(descendant)
        measured = native_outcome(run, task=task, case=case, expected_library=config,
                                  operational_state=state)
        receipt = {"schema": "dsr.meta-native-evaluation/v1", "state": "completed", "task": task,
                   "split": case["split"],
                   "case_sha256": digest(case), "descendant_sha256": digest(descendant),
                   "verifier_sha256": digest(identity), "native_success": measured["native_success"],
                   "checks": {key: measured["checks"][key] for key in required_checks},
                   "evidence": {**measured["evidence"], str(output / "request.json"): file_digest(output / "request.json")},
                   "physical_steps": measured["physical_steps"], "wall_seconds": time.monotonic() - started,
                   "qualification": None, "scope": identity["scope"]}
        # Keep compatibility with isolated verifier fixtures that return only
        # the legacy native fields; real RoboDojo receipts always include the
        # measured trajectory object from ``native_outcome``.
        if "trajectory" in measured:
            receipt["trajectory"] = measured["trajectory"]
        atomic_json(output / "receipt.json", receipt)
        return output / "receipt.json"
