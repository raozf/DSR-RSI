"""A plan is not a success claim: execution and measured verification are separate."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from .config import read_json
from .geometry import vector
from .hardware import emergency_stop
from .model import PreparationError
from .perception import age_seconds, check_observation
from .safety import check_robot_state, check_segment, validate_plan


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


class Journal:
    def __init__(self, root, task):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = stamp+"_"+task+"_"+uuid.uuid4().hex[:8]
        self.path = Path(root).resolve()/self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path/"checkpoints").mkdir()
        self._lock = threading.Lock()

    def event(self, kind, **values):
        with self._lock:
            with (self.path/"events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"timestamp": utc_now(), "kind": kind, **values}, ensure_ascii=False, allow_nan=False)+"\n")


class MockRig:
    """No sockets, serial ports, SDK imports or controller register writes."""
    def __init__(self, config):
        self.config = config
        self.poses = {side: config.park(side).tolist() for side in config.arms}
        self.commands = []
        self.stopped = False
        self.sequence = 0

    def connect(self):
        pass

    def prepare(self):
        check_robot_state(self.config, self.states())

    def states(self):
        self.sequence += 1
        return {side: {"mode": 5, "error": 0, "drag": 0, "paused": 0,
                       "joints_deg": list(self.config.frames[side]["park_joints_deg"]),
                       "pose_world_mm_deg": pose.copy(), "sequence": self.sequence}
                for side, pose in self.poses.items()}

    def move(self, side, target, heartbeat):
        if self.stopped:
            raise PreparationError("Mock already stopped")
        other = "right" if side == "left" else "left"
        states = heartbeat(parked_side=other)
        check_segment(self.config, side, states[side]["pose_world_mm_deg"], target, states[other]["pose_world_mm_deg"])
        self.commands.append(("move", side, target))
        self.poses[side] = list(target)
        return self.states()[side]

    def grip(self, side, opening, heartbeat):
        if self.stopped:
            raise PreparationError("Mock already stopped")
        heartbeat(parked_side="right" if side == "left" else "left")
        self.commands.append(("gripper", side, opening))
        return {"simulated": True, "opening": opening}

    def stop(self):
        self.stopped = True
        return {side: {"acknowledged": True, "simulated": True} for side in self.poses}

    def close(self):
        pass


class Executor:
    def __init__(self, config, rig, journal, scene, mock=False, checkpoint_dir=None):
        self.config, self.rig, self.journal, self.scene, self.mock = config, rig, journal, scene, mock
        self.checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else journal.path/"checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._stop_monitor = threading.Event()
        self._fault_lock = threading.Lock()
        self._fault = None
        self._stopped = False
        self._stop_lock = threading.Lock()
        self._parked_side = None

    def stop(self, reason):
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            result = self.rig.stop() if self.mock else emergency_stop(self.config.data)
            self.journal.event("emergency_stop", reason=reason, arms=result)

    def heartbeat(self, parked_side=None):
        with self._fault_lock:
            fault = self._fault
        if fault:
            raise PreparationError("Background safety monitor: "+fault)
        states = self.rig.states()
        check_robot_state(self.config, states, parked_side or self._parked_side)
        return states

    def _monitor(self):
        while not self._stop_monitor.wait(0.05):
            try:
                self.heartbeat()
            except BaseException as exc:
                with self._fault_lock:
                    self._fault = str(exc)
                self.stop(str(exc))
                return

    def _mock_checkpoint(self, expected):
        if expected["kind"] == "grasp":
            return {"grasped_object": expected["object"]}
        if expected["kind"] == "cover_state":
            return {"covered_labels": expected["covered_labels"], "all_cups_opening_down": True,
                    "block_positions_world_mm": {label: self.scene["objects"][label]["position_world_mm"] for label in ["red", "green", "blue"]}}
        objects = {label: dict(self.scene["objects"][label], yaw_deg=0) for label in ["cube0", "cube1", "cube2"]}
        for item in objects.values():
            item["position_world_mm"] = list(item["position_world_mm"])
            item["position_world_mm"][1] = item["size_mm"]/2+100
        return {"objects": objects, "max_block_lift_mm": {label: 0 for label in objects}}

    def before_step(self, index, step):
        """Endpoint adapters may require a fresh remote action before execution."""
        return step

    def after_step(self, index, step, actual):
        pass

    def finish_plan(self):
        pass

    def checkpoint(self, step):
        requested = time.time()
        request = {"schema": "dsr.xtrainer-checkpoint-request/v1", "run_id": self.journal.run_id,
                   "stage": step.stage, "calibration_id": self.config.calibration["calibration_id"],
                   "requested_at": utc_now(), "expected": step.expected, "instruction": step.instruction}
        self.journal.event("checkpoint_requested", **request)
        target = self.checkpoint_dir/(step.stage+".json")
        write_json(self.checkpoint_dir/(step.stage+".request.json"), request)
        if self.mock:
            record = {"schema": "dsr.xtrainer-checkpoint/v1", "run_id": self.journal.run_id,
                      "stage": step.stage, "calibration_id": self.config.calibration["calibration_id"],
                      "captured_at": utc_now(), "source": "mock", "simulated": True, "approved": True,
                      **self._mock_checkpoint(step.expected)}
            write_json(target, record)
        else:
            print(f"Checkpoint {step.stage}: {step.instruction}\nWrite measured observation to: {target}", flush=True)
        deadline = time.monotonic()+self.config.safety["checkpoint_timeout_s"]
        while time.monotonic() < deadline:
            self.heartbeat()
            if target.exists():
                record = read_json(target)
                if record.get("schema") != "dsr.xtrainer-checkpoint/v1" or record.get("run_id") != self.journal.run_id or record.get("stage") != step.stage:
                    raise PreparationError("Checkpoint identity/run/stage does not match request")
                if record.get("calibration_id") != self.config.calibration["calibration_id"] or record.get("approved") is not True:
                    raise PreparationError("Checkpoint calibration or approval failed")
                if not self.mock and (record.get("simulated") is not False or record.get("source") not in {"manual", "vision"}):
                    raise PreparationError("Hardware checkpoint requires real manual/vision observation, simulated=false")
                age = age_seconds(record.get("captured_at"))
                if age > self.config.safety["scene_max_age_s"] or time.time()-age < requested-2:
                    raise PreparationError("Checkpoint observation predates the request or is stale")
                check_observation(step.expected, record, self.scene)
                # Keep the exact accepted record even when an external directory is used.
                write_json(self.journal.path/"checkpoints"/(step.stage+".accepted.json"), record)
                self.journal.event("checkpoint_accepted", stage=step.stage, source=record.get("source"))
                return
            time.sleep(0.1)
        raise PreparationError("Timed out waiting for measured checkpoint: "+step.stage)

    def run(self, plan):
        validate_plan(self.config, plan)
        write_json(self.journal.path/"plan.json", plan.to_dict())
        plan_hash = hashlib.sha256((self.journal.path/"plan.json").read_bytes()).hexdigest()
        summary = {"schema": "dsr.xtrainer-run/v1", "run_id": self.journal.run_id, "task": plan.task,
                   "mode": "mock" if self.mock else "hardware", "plan_sha256": plan_hash,
                   "started_at": utc_now(), "steps_completed": 0, "execution_completed": False,
                   "real_robot_success_rate": None, "hardware_motion_tested": False,
                   "hardware_actuation_requested": not self.mock,
                   "task_verified_by": None}
        thread = None
        prepared = False
        try:
            self.rig.connect()
            # Preparation may enable hardware, so any failure from here must stop both.
            prepared = True
            self.rig.prepare()
            self.heartbeat()
            thread = threading.Thread(target=self._monitor, daemon=True, name="xtrainer-safety-monitor")
            thread.start()
            for i, step in enumerate(plan.steps):
                if step.arm:
                    self._parked_side = "right" if step.arm == "left" else "left"
                self.heartbeat()
                step = self.before_step(i, step)
                self.journal.event("step_started", index=i, stage=step.stage, step_kind=step.kind, arm=step.arm)
                if step.kind == "move":
                    actual = self.rig.move(step.arm, step.pose_world_mm_deg, self.heartbeat)
                    if not self.mock:
                        summary["hardware_motion_tested"] = True
                elif step.kind == "gripper":
                    actual = self.rig.grip(step.arm, step.opening, self.heartbeat)
                else:
                    self.checkpoint(step)
                    actual = {"checkpoint": "accepted"}
                self.after_step(i, step, actual)
                self.journal.event("step_completed", index=i, actual=actual)
                summary["steps_completed"] = i+1
            self.heartbeat()
            self.finish_plan()
            summary["execution_completed"] = True
            summary["task_verified_by"] = "mock_only" if self.mock else "external_checkpoint_observations"
            self.journal.event("execution_completed", simulated=self.mock)
            return summary
        except BaseException as exc:
            summary["error"] = str(exc) or type(exc).__name__
            self.journal.event("aborted", error=summary["error"])
            if prepared:
                self.stop(summary["error"])
            raise
        finally:
            self._stop_monitor.set()
            if thread:
                thread.join(timeout=5)
            self.rig.close()
            summary["finished_at"] = utc_now()
            write_json(self.journal.path/"summary.json", summary)
