"""Observation/acknowledgement driven DSR Code Policy over the SDK's JSON /ws.

Actions are explicitly typed Cartesian skills, not Motus joint predictions.
Controller-specific conversion remains on the robot PC. One connection owns one
episode; neither endpoint reconnects, retries or advances without a receipt.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import tempfile
import time
import uuid

import numpy as np

from dsr_xtrainer.geometry import rotation_distance, vector
from dsr_xtrainer.model import PreparationError, Step
from dsr_xtrainer.perception import age_seconds, check_observation
from dsr_xtrainer.safety import check_robot_state, check_segment
from .protocol import ACTION_SPACE, SCHEMA, digest, projected_config

STREAM_MODE = "feedback_skill_stream/v1"


def now():
    return datetime.now(timezone.utc).isoformat()


class ServerEpisode:
    def __init__(self, request, response):
        options = request["execution"]
        if (not isinstance(options, dict) or options.get("mode") != STREAM_MODE or type(options.get("mock")) is not bool or
                not isinstance(options.get("run_id"), str) or not 1 <= len(options["run_id"]) <= 160):
            raise PreparationError("Invalid feedback stream configuration")
        self.mock, self.run_id = options["mock"], options["run_id"]
        with tempfile.TemporaryDirectory(prefix="dsr-ws-stream-") as tmp:
            self.config = projected_config(request["dsr"]["planning_config"], tmp)
        self.scene = request["dsr"]["scene"]
        if not self.mock:
            if (self.config.data["example_only"] is not False or
                    self.config.calibration["example_only"] is not False or self.scene["example_only"] is not False):
                raise PreparationError("Hardware stream requires measured configuration and scene")
            if age_seconds(self.scene.get("captured_at")) > self.config.safety["scene_max_age_s"]:
                raise PreparationError("Hardware stream scene is stale")
        self.plan = response["plan"]
        self.plan_sha256 = response["plan_sha256"]
        self.stream_id = uuid.uuid4().hex
        self.parent_id = request["request_id"]
        self.sequence, self.pending, self.closed = 0, None, False
        self.issued_at = None
        self.feedback_sequences = {side: -1 for side in ("left", "right")}
        self.parked_side = None
        self.expected_poses = {side: self.config.park(side) for side in ("left", "right")}
        self.metadata = {"mode": STREAM_MODE, "stream_id": self.stream_id, "run_id": self.run_id}

    def _acknowledgement(self, ack):
        if self.pending is None:
            if ack is not None:
                raise PreparationError("First inference cannot acknowledge an unissued action")
            return
        index, step = self.pending
        if (not isinstance(ack, dict) or type(ack.get("index")) is not int or ack["index"] != index or
                ack.get("step_sha256") != digest(step) or ack.get("completed") is not True):
            raise PreparationError("Previous skill needs its exact completed acknowledgement")
        if not isinstance(ack.get("actual"), dict):
            raise PreparationError("Execution acknowledgement requires actual feedback")
        if step["kind"] == "move":
            self.expected_poses[step["arm"]] = vector(step["pose_world_mm_deg"], 6, "previous target")
        elif step["kind"] == "gripper":
            key = "opening" if self.mock else "opening_commanded"
            if ack["actual"].get(key) != step["opening"]:
                raise PreparationError("Gripper acknowledgement differs from issued opening")
        else:
            record = ack.get("checkpoint", {})
            if (not isinstance(record, dict) or record.get("schema") != "dsr.xtrainer-checkpoint/v1" or record.get("run_id") != self.run_id or
                    record.get("stage") != step["stage"] or record.get("approved") is not True or
                    record.get("calibration_id") != self.config.calibration["calibration_id"]):
                raise PreparationError("Checkpoint acknowledgement identity/calibration failed")
            sources = {"mock"} if self.mock else {"manual", "vision"}
            if record.get("source") not in sources or record.get("simulated") is not self.mock:
                raise PreparationError("Checkpoint source does not match execution mode")
            age = age_seconds(record.get("captured_at"))
            if age > self.config.safety["scene_max_age_s"] or time.time()-age < self.issued_at-2:
                raise PreparationError("Checkpoint acknowledgement is stale or predates the issued skill")
            check_observation(step["expected"], record, self.scene)

    def infer(self, request):
        if self.closed:
            raise PreparationError("Feedback stream is closed; no replay is allowed")
        try:
            return self._infer(request)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.closed = True
            raise PreparationError("Malformed feedback stream message: "+str(exc)) from exc
        except Exception:
            self.closed = True
            raise

    def _infer(self, request):
        if (request.get("type") != "inference" or request.get("schema") != SCHEMA or request.get("phase") != "step" or
                request.get("stream_id") != self.stream_id or request.get("parent_request_id") != self.parent_id or
                type(request.get("sequence")) is not int or request["sequence"] != self.sequence or
                not isinstance(request.get("request_id"), str) or len(request["request_id"]) != 32):
            raise PreparationError("Feedback stream identity/sequence mismatch; no replay")
        if age_seconds(request.get("captured_at")) > 4:
            raise PreparationError("Remote inference needs fresh robot feedback")
        observation = request.get("observation", {})
        source = "mock" if self.mock else "controller_feedback_with_commanded_grippers"
        if (not isinstance(observation, dict) or observation.get("source") != source or
                observation.get("gripper_position_measured") is not False):
            raise PreparationError("Declare the actual robot/gripper feedback source")
        states = observation.get("robot_state", {})
        if not isinstance(states, dict) or not all(isinstance(s, dict) for s in states.values()):
            raise PreparationError("Robot feedback must contain arm state objects")
        check_robot_state(self.config, states)
        proprio = request.get("proprio_data")
        if not isinstance(proprio, list) or len(proprio) != 1:
            raise PreparationError("Expected one SDK-order 14D proprioception sample")
        state14 = vector(proprio[0], 14, "SDK-order proprioception")
        if not 0 <= state14[6] <= 1 or not 0 <= state14[13] <= 1:
            raise PreparationError("SDK gripper command must be normalized")
        for i, side in enumerate(("left", "right")):
            seq = states[side].get("sequence")
            if type(seq) is not int or seq < 0 or seq <= self.feedback_sequences[side]:
                raise PreparationError("Robot feedback sequence must advance for both arms")
            if not np.allclose(np.rad2deg(state14[7*i:7*i+6]), states[side]["joints_deg"], atol=1e-6, rtol=0):
                raise PreparationError("14D proprioception does not match measured arm order/units")
            self.feedback_sequences[side] = seq
        self._acknowledgement(request.get("acknowledgement"))
        # A completed receipt alone cannot establish arrival or stationary arms.
        for side in ("left", "right"):
            actual, target = np.asarray(states[side]["pose_world_mm_deg"]), self.expected_poses[side]
            if (np.linalg.norm(actual[:3]-target[:3]) > self.config.safety["position_tolerance_mm"] or
                    rotation_distance(actual[3:], target[3:]) > self.config.safety["rotation_tolerance_deg"]):
                raise PreparationError(side+": measured TCP differs from the acknowledged policy state")
        done = self.sequence == len(self.plan["steps"])
        if self.sequence == 0 or done:
            for side in ("left", "right"):
                check_robot_state(self.config, states, parked_side=side)
        actions = []
        if not done:
            step = self.plan["steps"][self.sequence]
            if step["arm"]:
                other = "right" if step["arm"] == "left" else "left"
                self.parked_side = other
                if step["kind"] == "move":
                    check_segment(self.config, step["arm"], states[step["arm"]]["pose_world_mm_deg"],
                                  step["pose_world_mm_deg"], states[other]["pose_world_mm_deg"])
            check_robot_state(self.config, states, parked_side=self.parked_side)
            self.pending = (self.sequence, step)
            self.issued_at = time.time()
            actions = [step]
        else:
            self.closed = True
        response = {"type": "inference", "schema": SCHEMA, "execution_mode": STREAM_MODE,
                    "stream_id": self.stream_id, "request_id": request["request_id"],
                    "sequence": self.sequence, "action_space": ACTION_SPACE,
                    "request_sha256": digest(request), "plan_sha256": self.plan_sha256,
                    "skill_actions": actions, "done": done}
        self.sequence += 1
        return response


class StepClient:
    def __init__(self, session, request, response, journal, mock):
        self.session, self.journal, self.mock = session, journal, mock
        self.metadata = response.get("execution", {})
        if (not isinstance(self.metadata, dict) or self.metadata.get("mode") != STREAM_MODE or
                self.metadata.get("run_id") != journal.run_id or
                not isinstance(self.metadata.get("stream_id"), str) or len(self.metadata["stream_id"]) != 32):
            raise PreparationError("Server did not negotiate the required feedback stream")
        self.parent_id, self.plan_sha256 = request["request_id"], response["plan_sha256"]
        self.sequence, self.ack, self.completed = 0, None, False
        self.pending, self.closed = None, False

    def next(self, expected, states, state14):
        if self.closed:
            raise PreparationError("Client feedback stream is closed; no replay is allowed")
        try:
            if self.pending is not None and self.ack is None:
                raise PreparationError("Complete and acknowledge the issued remote skill first")
            return self._next(expected, states, state14)
        except Exception:
            self.closed = True
            raise

    def _next(self, expected, states, state14):
        request = {"type": "inference", "schema": SCHEMA, "phase": "step",
                   "stream_id": self.metadata["stream_id"], "parent_request_id": self.parent_id,
                   "request_id": uuid.uuid4().hex, "sequence": self.sequence, "captured_at": now(),
                   "proprio_data": [list(state14)], "acknowledgement": self.ack,
                   "observation": {"source": "mock" if self.mock else "controller_feedback_with_commanded_grippers",
                                   "gripper_position_measured": False, "robot_state": states}}
        response = self.session.exchange(request, timeout=3)
        self.journal.event("remote_inference", index=self.sequence, request=request, response=response)
        if age_seconds(request["captured_at"]) > 4:
            raise PreparationError("Remote skill arrived after the feedback deadline")
        if response.get("type") == "error":
            raise PreparationError("Remote Code Policy rejected feedback: "+str(response.get("detail")))
        for key, value in [("type", "inference"), ("schema", SCHEMA), ("execution_mode", STREAM_MODE),
                           ("stream_id", self.metadata["stream_id"]), ("request_id", request["request_id"]),
                           ("sequence", self.sequence), ("action_space", ACTION_SPACE),
                           ("request_sha256", digest(request)), ("plan_sha256", self.plan_sha256)]:
            if response.get(key) != value or (key == "sequence" and type(response[key]) is not int):
                raise PreparationError("Invalid remote skill response: "+key)
        actions = [] if expected is None else [asdict(expected)]
        if response.get("done") is not (expected is None) or digest(response.get("skill_actions")) != digest(actions):
            raise PreparationError("Remote skill differs from the validated scene/plan contract")
        self.pending = None if expected is None else (self.sequence, asdict(expected))
        self.sequence += 1
        self.ack = None
        self.completed = expected is None
        self.closed = self.completed
        return None if expected is None else Step(**response["skill_actions"][0])

    def acknowledge(self, index, step, actual, checkpoint=None):
        if (self.closed or self.pending is None or self.ack is not None or type(index) is not int or
                index != self.pending[0] or digest(asdict(step)) != digest(self.pending[1])):
            self.closed = True
            raise PreparationError("Cannot acknowledge an unissued remote skill")
        self.ack = {"index": index, "step_sha256": digest(asdict(step)), "completed": True, "actual": actual}
        if checkpoint is not None:
            self.ack["checkpoint"] = checkpoint
