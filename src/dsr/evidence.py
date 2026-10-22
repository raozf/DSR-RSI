"""Evidence indexing. A declared clip is not a validated physical execution."""
from __future__ import annotations
import hashlib
import json
import io
import os
import time
from itertools import islice
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path


ROBOT_STATE_FIELDS = frozenset({
    'arm_joint_state', 'ee_joint_state', 'ee_pose', 'tcp_pose', 'delta_ee_pose',
    'left_arm_joint_state', 'right_arm_joint_state', 'left_ee_joint_state',
    'right_ee_joint_state', 'left_ee_pose', 'right_ee_pose', 'left_tcp_pose',
    'right_tcp_pose', 'left_delta_ee_pose', 'right_delta_ee_pose',
})


def permitted_observation(observation, *, max_side=640):
    """Copy only the public RGB/proprioception contract, without mutating inputs.

    Encoded camera values follow XPolicyLab's RGB decoding convention. Optional
    downsampling affects evidence copies only, never the controller observation.
    """
    import numpy as np
    from PIL import Image

    result = {'state': {}, 'vision': {}}
    if isinstance(observation.get('instruction'), str):
        result['instruction'] = observation['instruction']
    for key, value in observation.get('state', {}).items():
        if key in ROBOT_STATE_FIELDS:
            result['state'][key] = deepcopy(value)
    for name, camera in observation.get('vision', {}).items():
        if name not in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
            continue
        if not isinstance(camera, dict) or 'color' not in camera:
            continue
        value = camera['color']
        if isinstance(value, (bytes, bytearray)):
            # This decoder recognizes both legacy channel-reversed bytes and
            # marked RGB JPEGs. Generic image decoders silently swap legacy RGB.
            from dsr.paths import bootstrap_policy
            bootstrap_policy()
            from XPolicyLab.process_data import decode_image_bit
            value = decode_image_bit(value)
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = array.transpose(1, 2, 0)
        if array.ndim != 3 or array.shape[-1] not in (3, 4) or array.dtype != np.uint8:
            raise ValueError(f'Invalid RGB evidence for camera {name}')
        image = Image.fromarray(array).convert('RGB')
        image.thumbnail((max_side, max_side))
        result['vision'][str(name)] = {'color': np.array(image, copy=True)}
    return result


def _json_value(value, *, depth=0):
    """Bound individual records; never stringify arbitrary simulator objects."""
    if depth > 8:
        return {'omitted': 'depth budget'}
    if isinstance(value, dict):
        return {str(k)[:160]: _json_value(v, depth=depth + 1)
                for k, v in islice(value.items(), 128) if k != 'vision'}
    if isinstance(value, (list, tuple, deque)):
        return [_json_value(v, depth=depth + 1) for v in islice(value, 256)]
    if hasattr(value, 'tolist'):
        if getattr(value, 'size', 0) > 256:
            return {'omitted': 'array budget', 'shape': list(value.shape)}
        return _json_value(value.tolist(), depth=depth + 1)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else {'omitted': 'non-finite value'}
    if isinstance(value, str):
        return value[:4096]
    return {'omitted': 'unsupported value type'}


@dataclass(frozen=True)
class EvidenceConfig:
    """Passive capture limits, per physical attempt; no extra sensor requests."""
    max_bytes: int = 32 * 1024 * 1024
    max_metadata_bytes: int = 4 * 1024 * 1024
    max_frames: int = 48
    max_events: int = 256
    max_actions: int = 512
    max_samples: int = 128
    max_side: int = 640
    observation_stride: int = 25
    pre_event_frames: int = 2
    post_event_frames: int = 2
    max_agent_images: int = 24

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or value < (0 if key.endswith('event_frames') else 1):
                raise ValueError(f'Invalid evidence budget {key}')
        if self.max_metadata_bytes > self.max_bytes:
            raise ValueError('Evidence metadata budget exceeds total byte budget')

    @classmethod
    def from_environment(cls):
        names = {'DSR_EVIDENCE_MAX_BYTES': 'max_bytes',
                 'DSR_EVIDENCE_METADATA_BYTES': 'max_metadata_bytes',
                 'DSR_EVIDENCE_MAX_FRAMES': 'max_frames',
                 'DSR_EVIDENCE_MAX_SIDE': 'max_side',
                 'DSR_EVIDENCE_OBSERVATION_STRIDE': 'observation_stride',
                 'DSR_EVIDENCE_MAX_AGENT_IMAGES': 'max_agent_images'}
        return cls(**{field: int(os.environ[name]) for name, field in names.items() if name in os.environ})


class EvidenceRecorder:
    """Bounded event and RGB recorder independent of policy and simulator APIs.

    State/action rows preserve actual call order and observation age. Stage
    outcomes are recorded only when explicitly reported by a controller using
    permitted observations; changing a stage never establishes task success.
    Collection faults are retained and do not interrupt robot execution.
    """

    def __init__(self, directory=None, *, identity=None, attempt=1, plan=None, config=None):
        self.config = config or EvidenceConfig.from_environment()
        self.directory = Path(directory) if directory is not None else None
        self.identity = _json_value(identity or {})
        self.attempt = attempt
        self.started = time.monotonic()
        self.wall_started = time.time()
        self.steps = 0
        self.observations = 0
        self.events = deque(maxlen=self.config.max_events)
        self.keyframes = []
        self.errors = []
        self.dropped = {'metadata': 0, 'images': 0, 'events': 0}
        self.storage_bytes = self.metadata_bytes = 0
        self.collection_seconds = 0.0
        self.event_count = 0
        self.current_stage = None
        self.earliest_deviation = None
        self.completed_stages = []
        self._initial_state = None
        self._last = None
        self._observation_step = None
        self._pre = deque(maxlen=self.config.pre_event_frames)
        self._pending_snapshots = deque(maxlen=self.config.max_frames)
        self._pending_bytes = 0
        self._post_remaining = 0
        self._captured = set()
        self._events_created = False
        self._output_blocked = False
        self._closed = False
        self._bound_plan = False
        self.plan = dict(schema='dsr.evidence-plan/v1', original_goal=None,
                         planned_stages=[], important_objects=[], high_risk_stages=[],
                         expected_transitions=[], available_cameras=[], proprioceptive_signals=[],
                         collection='passive; existing observations only',
                         unknowns=['Object poses, forces, contact and task progress are not inferred.'])
        self.plan.update(_json_value(plan or {}))
        if self.directory is not None:
            self._guard(lambda: self.directory.mkdir(parents=True, exist_ok=True))
            self._protect_output()

    def _guard(self, operation):
        started = time.perf_counter()
        try:
            return operation()
        except Exception as exc:
            if len(self.errors) < 16:
                self.errors.append(f'{type(exc).__name__}: {exc}'[:400])
            return None
        finally:
            self.collection_seconds += time.perf_counter() - started

    def _protect_output(self):
        def occupied():
            if not self.directory.is_dir():
                raise NotADirectoryError(self.directory)
            return any(self.directory.iterdir())

        result = self._guard(occupied)
        if result is None or result:
            self._output_blocked = True
            if result:
                self.errors.append('Evidence output directory is not empty; preserving existing files')

    def _write(self, name, data, *, metadata=True):
        if self.directory is None or self._output_blocked:
            return False
        reserve = 0 if name == 'manifest.json' else min(128 * 1024, self.config.max_metadata_bytes // 2)
        if (self.storage_bytes + len(data) > self.config.max_bytes - reserve or
                (metadata and self.metadata_bytes + len(data) > self.config.max_metadata_bytes - reserve)):
            self.dropped['metadata' if metadata else 'images'] += 1
            return False
        path = contained(self.directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'ab' if name == 'events.jsonl' and self._events_created else 'xb'
        with path.open(mode) as stream:
            stream.write(data)
        if name == 'events.jsonl':
            self._events_created = True
        self.storage_bytes += len(data)
        if metadata:
            self.metadata_bytes += len(data)
        return True

    def _event(self, kind, **payload):
        self.event_count += 1
        row = dict(ref=f'event_{self.event_count:06d}', kind=kind, step=self.steps,
                   observation_index=self.observations, timestamp=time.monotonic() - self.started,
                   attempt=self.attempt, stage=self.current_stage, **_json_value(payload))
        if len(self.events) == self.events.maxlen:
            self.dropped['events'] += 1
        self.events.append(row)
        self._write('events.jsonl', (json.dumps(row, allow_nan=False) + '\n').encode())
        return row

    def observe(self, observation):
        def capture():
            projected = permitted_observation(observation, max_side=self.config.max_side)
            self.observations += 1
            self._last = projected
            self._observation_step = self.steps
            if self._initial_state is None:
                self._initial_state = _json_value(self._last)
                self.plan['original_goal'] = self._last.get('instruction')
                self.plan['available_cameras'] = list(self._last['vision'])
                self.plan['proprioceptive_signals'] = list(self._last['state'])
                self._write('plan.json', (json.dumps(self.plan, indent=2) + '\n').encode())
            row = self._event('observation', state=self._last['state'])
            if self.observations == 1:
                self._snapshot(self._last, row, 'initial')
            elif self._post_remaining:
                self._snapshot(self._last, row, 'post_event')
                self._post_remaining -= 1
            elif self.observations % self.config.observation_stride == 0:
                self._snapshot(self._last, row, 'periodic')
            self._pre.append((self._last, row))
            return projected
        return self._guard(capture)

    def executed(self, action):
        self.steps += 1
        return self._guard(lambda: self._event('executed_action', action=action,
            observation_step=self._observation_step,
            observation_age_steps=None if self._observation_step is None else self.steps - self._observation_step,
            state=(self._last or {}).get('state', {})))

    def returned_chunk(self, actions):
        """Record a delivered action horizon without copying unbounded queues."""
        return self._guard(lambda: self._event(
            'action_chunk', delivered_count=len(actions), action_horizon=len(actions)))

    def _snapshot(self, observation, event, reason):
        from PIL import Image
        if self.directory is None:
            # The evaluator may allocate its output directory only after the
            # episode identity is bound. Keep a bounded copy for later
            # materialization; this does not request another observation.
            size = sum(camera['color'].nbytes for camera in observation.get('vision', {}).values())
            if (len(self._pending_snapshots) >= self._pending_snapshots.maxlen or
                    self._pending_bytes + size > self.config.max_bytes):
                self.dropped['images'] += 1
            else:
                self._pending_snapshots.append((deepcopy(observation), deepcopy(event), reason))
                self._pending_bytes += size
            return
        for camera, values in observation.get('vision', {}).items():
            key = (event['observation_index'], camera)
            if key in self._captured:
                continue
            if len(self.keyframes) >= self.config.max_frames:
                self.dropped['images'] += 1
                continue
            if reason == 'periodic' and len(self.keyframes) >= max(1, self.config.max_frames // 2):
                self.dropped['images'] += 1
                continue
            stream = io.BytesIO()
            Image.fromarray(values['color']).save(stream, format='PNG')
            data = stream.getvalue()
            ref = f"keyframe_{len(self.keyframes):06d}"
            path = f'images/{ref}.png'
            if self._write(path, data, metadata=False):
                self.keyframes.append(dict(ref=ref, path=path, sha256=hashlib.sha256(data).hexdigest(),
                    camera=camera, event_ref=event['ref'], observation_index=event['observation_index'],
                    step=event['step'], timestamp=event['timestamp'], reason=reason,
                    shape=list(values['color'].shape), source='permitted_rgb'))
                self._captured.add(key)

    def record_stage(self, stage, *, expected=None, observed=None, discrepancy=None,
                     evidence_refs=(), outcome=None, confidence=None):
        """Record explicit controller evidence; unknown is never called a failure.

        An observed discrepancy requires references to retained events or images.
        Reported completion remains a controller-stage claim, not benchmark success.
        """
        def capture():
            available = {row['ref'] for row in self.events} | {row['ref'] for row in self.keyframes}
            if set(evidence_refs) - available:
                raise ValueError('Stage event cites unavailable evidence')
            if discrepancy and (not observed or not evidence_refs):
                raise ValueError('An observed deviation requires an observation and evidence references')
            if outcome not in (None, 'success', 'failure', 'uncertain'):
                raise ValueError('Invalid controller stage outcome')
            if confidence is not None and not 0 <= confidence <= 1:
                raise ValueError('Invalid stage confidence')
            previous = self.current_stage
            self.current_stage = str(stage)[:160]
            row = self._event('stage_transition' if stage != previous else 'stage_check',
                previous_stage=previous, expected=expected, observed=observed,
                discrepancy=discrepancy, evidence_refs=list(evidence_refs),
                outcome=outcome, confidence=confidence)
            if discrepancy and self.earliest_deviation is None:
                self.earliest_deviation = deepcopy(row)
            if (outcome == 'success' and stage not in self.completed_stages
                    and len(self.completed_stages) < self.config.max_events):
                self.completed_stages.append(stage)
            for observation, event in self._pre:
                self._snapshot(observation, event, 'pre_event')
            if self._last is not None:
                self._snapshot(self._last, row, 'stage_boundary')
            self._post_remaining = self.config.post_event_frames
            return row
        return self._guard(capture)

    def bind_plan(self, plan, controller=None):
        """Audit the actual bound plan; this does not trigger or modify planning."""
        def capture():
            if self.steps or self._bound_plan:
                raise ValueError('Episode plan must be bound once before the first action')
            if not isinstance(plan, dict):
                raise ValueError('Bound episode plan must be a dictionary')
            bounded_plan = _json_value(plan)
            bounded_controller = _json_value(controller)
            units = plan.get('operations')
            if not isinstance(units, list) or not units:
                composition = plan.get('composition')
                units = [composition] if isinstance(composition, str) else []
            self.plan.update(initial_agent_plan=bounded_plan,
                             selected_controller=bounded_controller,
                             planned_execution_units=_json_value(units))
            row = self._event('bound_plan', plan=bounded_plan, controller=bounded_controller)
            self._write('bound-plan.json', (json.dumps(row, indent=2, allow_nan=False) + '\n').encode())
            self._bound_plan = True
            return row
        return self._guard(capture)

    def plan_capture_error(self, message):
        """Retain a metadata failure without interrupting System 1 execution."""
        def capture():
            reason = str(message)[:400]
            if len(self.errors) < 16:
                self.errors.append('Bound plan unavailable: ' + reason)
            return self._event('bound_plan_unavailable', reason=reason)
        return self._guard(capture)

    def failure_package(self, *, terminal=None, terminal_observation=None, original_plan=None, controller=None):
        stages = [row for row in self.events if row['kind'] in ('stage_transition', 'stage_check')]
        return dict(schema='dsr.failure-evidence/v1', original_goal=self.plan['original_goal'],
            evidence_plan=deepcopy(self.plan), original_plan=_json_value(original_plan),
            selected_controller=_json_value(controller), initial_observation=self._initial_state,
            terminal_observation=_json_value(permitted_observation(terminal_observation,
                max_side=self.config.max_side)) if terminal_observation is not None else _json_value(self._last),
            terminal_outcome=_json_value(terminal),
            completed_stages=list(self.completed_stages), stage_events=stages,
            earliest_observed_deviation=deepcopy(self.earliest_deviation),
            final_visible_failure='Not inferred from the native outcome; inspect terminal RGB.',
            cause_candidates=[], diagnostic_confidence=None,
            uncertainty='No root cause is established by action commands or a terminal failure alone.',
            executed_action_count=self.steps, observation_count=self.observations,
            keyframes=deepcopy(self.keyframes), remaining_physical_attempts=1,
            collection=self.summary())

    def summary(self):
        return dict(status='DEGRADED' if self.errors else 'BOUNDED' if any(self.dropped.values()) else 'RECORDED',
            passive=True, additional_actions=0, additional_observations=0,
            elapsed_collection_seconds=self.collection_seconds, storage_bytes=self.storage_bytes,
            metadata_bytes=self.metadata_bytes, recorded_images=len(self.keyframes),
            events_seen=self.event_count, events_in_memory=len(self.events),
            dropped=dict(self.dropped), errors=list(self.errors), budgets=asdict(self.config))

    def finalize(self, directory=None, *, identity=None, outcome=None,
                 plan=None, policy=None):
        """Persist a bounded attempt manifest without affecting execution.

        ``directory`` and metadata are optional so callers can allocate the
        recorder before knowing the episode output path. Repeated finalization
        is idempotent and returns the same summary.
        """
        if self._closed:
            return self.summary()
        attached_late = directory is not None and self.directory is None
        if directory is not None and self.directory is None:
            self.directory = Path(directory)
            self._guard(lambda: self.directory.mkdir(parents=True, exist_ok=True))
            self._protect_output()
        if identity is not None:
            self.identity = _json_value(identity)
        if plan is not None:
            self.plan.update(_json_value(plan))
        if policy is not None:
            self._guard(lambda: self._event('bound_policy', policy=policy))
        if not self._closed:
            def finish():
                if self.directory is None:
                    self.errors.append('Evidence has no output directory; only bounded in-memory context is available')
                    return
                if attached_late:
                    self._write('plan.json', (json.dumps(self.plan, indent=2) + '\n').encode())
                    for event in self.events:
                        self._write('events.jsonl', (json.dumps(event, allow_nan=False) + '\n').encode())
                while self.directory is not None and self._pending_snapshots and len(self.keyframes) < self.config.max_frames:
                    observation, event, reason = self._pending_snapshots.popleft()
                    self._pending_bytes -= sum(camera['color'].nbytes for camera in observation.get('vision', {}).values())
                    self._snapshot(observation, event, reason)
                row = self._event('attempt_end', outcome=outcome)
                if self._last is not None:
                    self._snapshot(self._last, row, 'final_available_observation')
                manifest = dict(schema='dsr.attempt-evidence/v1', identity=self.identity,
                    attempt=self.attempt, wall_started=self.wall_started,
                    keyframes=self.keyframes, evidence_plan=self.plan,
                    earliest_observed_deviation=self.earliest_deviation,
                    summary=self.summary())
                payload = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
                if not self._write('manifest.json', payload):
                    # Keep the completed attempt truthful when a deliberately
                    # tiny storage budget prevents the manifest itself.
                    if (not self._output_blocked and
                            'manifest not persisted: evidence budget exhausted' not in self.errors):
                        self.errors.append('manifest not persisted: evidence budget exhausted')
            self._guard(finish)
            self._closed = True
        return self.summary()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evidence path escapes its root")
    return path


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


CLIP_FIELDS = {
    "clip_id", "run_id", "episode_id", "task_family", "task_configuration",
    "seed", "layout", "attempt", "skill_id", "skill_version", "policy_id",
    "policy_version", "camera", "start_timestamp", "end_timestamp",
    "parent_rollout", "skill_outcome", "task_outcome", "validation_status",
    "sha256", "path", "events", "start_event_id", "end_event_id",
}


def validate_clip(root, clip):
    missing = CLIP_FIELDS - set(clip)
    if missing:
        raise ValueError("Clip lacks fields: " + ", ".join(sorted(missing)))
    if clip["validation_status"] != "validated" or clip["skill_outcome"] != "success":
        raise ValueError("Clip is not a validated successful skill execution")
    if not 0 <= clip["start_timestamp"] < clip["end_timestamp"]:
        raise ValueError("Invalid clip boundary")
    if sha256(contained(root, clip["path"])) != clip["sha256"]:
        raise ValueError("Clip content hash changed")
    if not contained(root, clip["parent_rollout"]).is_file():
        raise ValueError("Missing parent rollout")
    events = read_json(contained(root, clip["events"]), [])
    index = {row["id"]: row for row in events}
    start, end = index[clip["start_event_id"]], index[clip["end_event_id"]]
    for row, kind, timestamp in ((start, "skill_start", clip["start_timestamp"]),
                                 (end, "skill_end", clip["end_timestamp"])):
        if (row["kind"] != kind or row["timestamp"] != timestamp or
                row["skill_id"] != clip["skill_id"] or
                row["episode_id"] != clip["episode_id"] or row["attempt"] != clip["attempt"]):
            raise ValueError("Clip boundary does not match execution events")
    if end.get("outcome") != "success":
        raise ValueError("Skill end event does not establish success")
    return (clip["run_id"], clip["episode_id"], clip["attempt"],
            clip["start_event_id"], clip["end_event_id"])


def inventory(root):
    root = Path(root)
    benchmark = read_json(root / "metadata/robodojo_inventory.json", {})
    capabilities = read_json(root / "metadata/physicalrsi_capabilities.json", {})
    code = read_json(root / "assets/physicalrsi/implementations/code-skills/programs.json", {})
    declared_clips = read_json(root / "evidence/clips.json", [])
    clips, rejected, executions, hashes = [], [], set(), set()
    for clip in declared_clips:
        try:
            key = validate_clip(root, clip)
            if key in executions or clip["sha256"] in hashes:
                raise ValueError("Duplicate execution or video; camera variants do not increase count")
            executions.add(key)
            hashes.add(clip["sha256"])
            clips.append(clip)
        except (ValueError, OSError, KeyError, TypeError) as error:
            rejected.append({"clip_id": clip.get("clip_id"), "error": str(error)})
    reports = []
    for base in (root / "evidence", root / "outputs", root / "results"):
        if not base.exists():
            continue
        for path in base.rglob("report.json"):
            data = read_json(path)
            if isinstance(data, dict) and "first_attempt" in data:
                reports.append(dict(data, evidence_path=str(path.relative_to(root))))
    # Count original episode identities, never retry attempts as new episodes.
    unique = {}
    for report in reports:
        key = (report.get("run_id", str(Path(report["evidence_path"]).parents[2])),
               report.get("task"), report.get("seed"), report.get("layout_id"))
        unique[key] = report
    completed = [row for row in unique.values() if row.get("status") == "COMPLETED"]
    first_successes = sum(row["first_attempt"].get("success") is True for row in completed)
    combined = sum(row["first_attempt"].get("success") is True or (
        row.get("retry_success") is True and row.get("final_success") is True)
        for row in completed)
    return {
        "schema": "dsr.evidence-index/v1",
        "counts": {
            "official_task_definitions": benchmark.get("canonical_tasks", 0),
            "native_configuration_definitions": benchmark.get("native_configurations", 0),
            "nominal_official_episodes": benchmark.get("nominal_episodes", 0),
            "declared_code_programs": len(code.get("programs", [])),
            "completed_native_episodes": len(completed),
            "first_successes": first_successes,
            "combined_successes": combined,
            "validated_distinct_skill_executions": len(clips),
        },
        "scope": "Counts distinguish source inventory from fresh execution evidence; no parity or superiority claim.",
        "tasks": benchmark.get("tasks", []),
        "policies": [{"name": p["name"], "description": p["description"],
                      "action_type": p["action_type"], "validation": "source registry only"}
                     for p in code.get("programs", [])],
        "capabilities": capabilities,
        "episodes": completed,
        "clips": clips,
        "invalid_clips": rejected,
        "research": read_json(root / "evidence/research-summary.json", {}),
    }
