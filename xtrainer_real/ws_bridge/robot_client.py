"""Convert Cartesian skills to the existing ZMQClientRobot's 14D joint API.

No SDK edits, direct ServoJ/MovL, serial access, auto-homing or auto-enabling.
Read-only controller IK/FK and feedback qualify commands before the node receives them.
"""
import math
import threading
import time

import numpy as np
import zmq

from dsr_xtrainer.config import bounded, read_json
from dsr_xtrainer.geometry import rotation_distance, segment_distance, transform_pose, vector
from dsr_xtrainer.hardware import Feedback, RealRig, emergency_stop, numeric_reply
from dsr_xtrainer.model import PreparationError
from dsr_xtrainer.safety import check_robot_state, check_segment
from .sdk import load_sdk


class Transport:
    def __init__(self, path):
        self.path = str(path)
        self.data = read_json(path)
        d = self.data
        if d.get("schema") != "dsr.xtrainer-zmq/v1":
            raise PreparationError("Unsupported XTrainer transport schema")
        if d.get("robot_host") not in {"127.0.0.1", "localhost", "::1"}:
            raise PreparationError("SDK pickle RPC must stay on trusted loopback")
        for key, low, high in [("robot_port", 1, 65535), ("rpc_timeout_s", .1, 3),
                               ("control_hz", 5, 30), ("cartesian_step_mm", .1, 2),
                               ("max_joint_step_deg", .1, 3), ("max_joint_speed_deg_s", .1, 5),
                               ("joint_tolerance_deg", .005, .1),
                               ("path_tolerance_mm", .1, 2), ("gripper_settle_s", .1, 5)]:
            bounded(d.get(key), low, high, "transport."+key, key == "robot_port")
        if d["joint_tolerance_deg"] >= d["max_joint_speed_deg_s"]*.03/2:
            raise PreparationError("Joint arrival tolerance must be below half the per-ServoJ speed budget")
        for side in ("left", "right"):
            g = d.get("grippers", {}).get(side, {})
            for key in ("open_command", "closed_command", "initial_opening"):
                bounded(g.get(key), 0, 1, side+"."+key)
            if g["open_command"] == g["closed_command"]:
                raise PreparationError("Gripper open/closed commands must differ")

    def require_hardware(self, config, execute=False):
        config.require_hardware(execute=False)  # No serial device ownership in this adapter.
        if self.data.get("example_only") is not False:
            raise PreparationError("Template transport cannot connect to hardware")
        if execute:
            for key in ("tcp_and_frames_measured", "workspace_and_paths_reviewed", "grippers_calibrated"):
                if config.calibration.get(key) is not True:
                    raise PreparationError("calibration."+key+" requires completed on-site measurement")
            if self.data.get("gripper_mapping_verified") is not True:
                raise PreparationError("Verify SDK gripper mapping and initial_opening on site first")
            for side in config.arms:
                vector(config.frames[side].get("tool_tcp_controller_mm_deg"), 6, "measured TCP")


def action14(states, openings, transport):
    """SDK order: left q(rad), closure, right q(rad), closure. No fake feedback."""
    result = []
    for side in ("left", "right"):
        result.extend(np.deg2rad(vector(states[side]["joints_deg"], 6, "joints")))
        opening = bounded(openings[side], 0, 1, "commanded opening")
        g = transport.data["grippers"][side]
        result.append(g["closed_command"]+opening*(g["open_command"]-g["closed_command"]))
    return vector(result, 14, "SDK joint command")


class ExistingClientRig(RealRig):
    def __init__(self, config, transport, execute=False):
        transport.require_hardware(config, execute)
        self.config, self.transport, self.execute = config, transport, execute
        self.dashboards, self.moves, self.feedback, self.grippers = {}, {}, {}, {}
        self.node = None
        self.sdk = load_sdk(transport.data["sdk_root"])
        self.openings = {s: transport.data["grippers"][s]["initial_opening"] for s in config.arms}
        self.aborted = threading.Event()
        self._command_lock = threading.Lock()
        self._last_send = 0.0

    def connect(self):
        Dashboard, Client, dtype = self.sdk
        s, d = self.config.safety, self.transport.data
        try:
            for side, arm in self.config.arms.items():
                self.dashboards[side] = Dashboard(arm["ip"], arm["dashboard_port"],
                                                   s["connect_timeout_s"], s["reply_timeout_s"])
                self.feedback[side] = Feedback(arm["ip"], arm["feedback_port"], dtype,
                                               s["feedback_timeout_s"], s["connect_timeout_s"])
                self.feedback[side].wait_first()
            self.node = Client(port=d["robot_port"], host=d["robot_host"])
            self.node._socket.setsockopt(zmq.LINGER, 0)
            self.node._socket.setsockopt(zmq.RCVTIMEO, round(1000*d["rpc_timeout_s"]))
            self.node._socket.setsockopt(zmq.SNDTIMEO, round(1000*d["rpc_timeout_s"]))
            if self.node.num_dofs() != 14:
                raise PreparationError("Existing robot_node must expose 14 DOF, left then right")
            self.verify_node_mapping()
        except BaseException:
            self.close()
            raise

    def verify_node_mapping(self):
        q = vector(self.node.get_joint_state(), 14, "robot_node joint state")
        tcp = vector(self.node.get_XYZrxryrz_state(), 12, "robot_node TCP state")
        states = self.states()
        for i, side in enumerate(("left", "right")):
            live = states[side]
            if np.max(np.abs(np.rad2deg(q[i*7:i*7+6])-live["joints_deg"])) > self.config.safety["park_joint_tolerance_deg"]:
                raise PreparationError(side+": node/direct joint feedback mapping differs")
            pose = transform_pose(np.asarray(self.config.frames[side]["world_T_base"]), tcp[i*6:i*6+6])
            self._pose_close(pose, live["pose_world_mm_deg"], "node/direct TCP feedback mapping differs")
        # Upstream q[6]/q[13] are constants, not gripper position measurements.
        return {"joint_mapping_verified": True, "tcp_mapping_verified": True,
                "gripper_feedback_available": False}

    def _pose_close(self, actual, target, reason):
        s = self.config.safety
        if (np.linalg.norm(np.asarray(actual)[:3]-np.asarray(target)[:3]) > s["position_tolerance_mm"] or
                rotation_distance(actual[3:], target[3:]) > s["rotation_tolerance_deg"]):
            raise PreparationError(reason)

    def prepare(self):
        if not self.execute or self.aborted.is_set():
            raise PreparationError("Motion requires --execute and a non-aborted client")
        states = self.states()
        check_robot_state(self.config, states)
        for side, state in states.items():
            if state["mode"] != 5:
                raise PreparationError("Existing node must have both arms enabled and idle at park")
            self._pose_close(state["pose_world_mm_deg"], self.config.park(side), side+": manually place at measured park first")
            if np.max(np.abs(np.asarray(state["joints_deg"])-self.config.frames[side]["park_joints_deg"])) > self.config.safety["park_joint_tolerance_deg"]:
                raise PreparationError(side+": park joint branch differs from calibration")
        self.verify_node_mapping()
        # Deliberately no EnableRobot, SetTool, speed writes or gripper commands.

    def _joints_valid(self, side, q):
        q = vector(q, 6, "IK joints (degrees)")
        f = self.config.frames[side]
        if np.any(q < f["joint_min_deg"]) or np.any(q > f["joint_max_deg"]):
            raise PreparationError(side+": IK joint limit exceeded")
        return q

    def forward(self, side, q):
        q = self._joints_valid(side, q)
        dash, arm = self.dashboards[side], self.config.arms[side]
        reply = dash.PositiveSolution(*q, arm["user_index"], arm["tool_index"])
        base = numeric_reply(reply, dash.last_command, 6)
        return transform_pose(np.asarray(self.config.frames[side]["world_T_base"]), base)

    def solve(self, side, target, current):
        base = transform_pose(np.linalg.inv(np.asarray(self.config.frames[side]["world_T_base"])), target)
        dash, arm = self.dashboards[side], self.config.arms[side]
        # Upstream optional IK seed serialization is invalid; use its exact base API.
        reply = dash.InverseSolution(*base, arm["user_index"], arm["tool_index"])
        q = self._joints_valid(side, numeric_reply(reply, dash.last_command, 6))
        if np.max(np.abs(q-vector(current, 6, "current joints"))) > self.transport.data["max_joint_step_deg"]:
            raise PreparationError(side+": IK branch jump; no command sent for this waypoint")
        self._pose_close(self.forward(side, q), target, side+": IK/FK TCP mismatch")
        return q

    def send_joints(self, side, q, states, heartbeat):
        # Rate limiting is based on wall-clock intervals; there is no catch-up burst.
        period = 1/self.transport.data["control_hz"]
        while time.monotonic()-self._last_send < period:
            heartbeat(parked_side="right" if side == "left" else "left")
            if self.aborted.is_set():
                raise PreparationError("Client aborted before node command")
            if self.aborted.wait(min(.02, period)):
                raise PreparationError("Client aborted")
        with self._command_lock:
            if self.aborted.is_set():
                raise PreparationError("Client aborted; no subsequent node commands allowed")
            if not self.execute:
                raise PreparationError("Robot commands require --execute")
            states = heartbeat(parked_side="right" if side == "left" else "left")
            if self.aborted.is_set():
                raise PreparationError("Client aborted before node command")
            q = self._joints_valid(side, q)
            limit = min(self.transport.data["max_joint_step_deg"], self.transport.data["max_joint_speed_deg_s"]*.03)
            if np.max(np.abs(q-states[side]["joints_deg"])) > limit+1e-8:
                raise PreparationError("Command exceeds the measured-state ServoJ step/speed budget")
            command_states = {s: dict(state) for s, state in states.items()}
            command_states[side]["joints_deg"] = q.tolist()
            command = action14(command_states, self.openings, self.transport)
            self._last_send = time.monotonic()
            # No retries: a lost acknowledgement may still mean the node executed it.
            result = self.node.command_joint_state(command, np.array([side == "left", side == "right"], dtype=int))
            if result != 1:
                raise PreparationError("robot_node rejected/unacknowledged the joint command")

    def move(self, side, target, heartbeat):
        other = "right" if side == "left" else "left"
        states = heartbeat(parked_side=other)
        start = vector(states[side]["pose_world_mm_deg"], 6, "start TCP")
        target = vector(target, 6, "target TCP")
        check_segment(self.config, side, start, target, states[other]["pose_world_mm_deg"])
        # These two ports keep a fixed grasp attitude. Do not interpolate Euler rotations.
        if rotation_distance(start[3:], target[3:]) > self.config.safety["rotation_tolerance_deg"]:
            raise PreparationError("This adapter requires the fixed measured task orientation")
        count = max(1, math.ceil(np.linalg.norm(target[:3]-start[:3])/self.transport.data["cartesian_step_mm"]))
        for k in range(1, count+1):
            states = heartbeat(parked_side=other)
            begin = np.asarray(states[side]["pose_world_mm_deg"])
            waypoint = np.r_[start[:3]+k/count*(target[:3]-start[:3]), target[3:]]
            check_segment(self.config, side, begin, waypoint, states[other]["pose_world_mm_deg"])
            current = np.asarray(states[side]["joints_deg"])
            q = self.solve(side, waypoint, current)
            # The pinned node uses ServoJ(t=0.03), even at a slower client rate.
            limit = min(self.transport.data["max_joint_step_deg"],
                        self.transport.data["max_joint_speed_deg_s"]*0.03/2,
                        self.transport.data["max_joint_speed_deg_s"]/self.transport.data["control_hz"])
            n = max(1, math.ceil(float(np.max(np.abs(q-current)))/limit))
            previous = begin
            for j in range(1, n+1):
                command_q = current+j/n*(q-current)
                fk = self.forward(side, command_q)
                if (segment_distance(fk[:3], begin[:3], waypoint[:3]) > self.transport.data["path_tolerance_mm"] or
                        rotation_distance(fk[3:], target[3:]) > self.config.safety["rotation_tolerance_deg"]):
                    raise PreparationError("Joint interpolation leaves the qualified Cartesian path")
                check_segment(self.config, side, previous, fk, states[other]["pose_world_mm_deg"])
                states = heartbeat(parked_side=other)
                # Limit jumps from measured state as well as between commanded targets.
                if np.max(np.abs(command_q-states[side]["joints_deg"])) > self.transport.data["max_joint_step_deg"]:
                    raise PreparationError("Measured joints lag the qualified command trajectory")
                self.send_joints(side, command_q, states, heartbeat)
                self._wait_arrived(side, fk, command_q, heartbeat)
                previous = fk
        return heartbeat(parked_side=other)[side]

    def _wait_arrived(self, side, target, q, heartbeat):
        deadline = time.monotonic()+self.config.safety["motion_timeout_s"]
        sequence, stable = -1, 0
        while time.monotonic() < deadline:
            state = heartbeat(parked_side="right" if side == "left" else "left")[side]
            if state["sequence"] > sequence:
                sequence = state["sequence"]
                p = np.asarray(state["pose_world_mm_deg"])
                ok = np.linalg.norm(p[:3]-target[:3]) <= self.config.safety["position_tolerance_mm"]
                ok &= rotation_distance(p[3:], target[3:]) <= self.config.safety["rotation_tolerance_deg"]
                ok &= np.max(np.abs(np.asarray(state["joints_deg"])-q)) <= self.transport.data["joint_tolerance_deg"]
                stable = stable+1 if ok and state["mode"] == 5 else 0
                if stable >= 3:
                    return
            if self.aborted.wait(.03):
                raise PreparationError("Client aborted")
        raise PreparationError("Node acknowledgement did not establish motion completion")

    def grip(self, side, opening, heartbeat):
        bounded(opening, 0, 1, "gripper opening")
        states = heartbeat(parked_side="right" if side == "left" else "left")
        self.openings[side] = opening
        # Hold the arm at its measured joints while changing only this gripper.
        self.send_joints(side, states[side]["joints_deg"], states, heartbeat)
        deadline = time.monotonic()+self.transport.data["gripper_settle_s"]
        while time.monotonic() < deadline:
            heartbeat(parked_side="right" if side == "left" else "left")
            if self.aborted.wait(.03):
                raise PreparationError("Client aborted")
        return {"opening_commanded": opening, "position_measured": False, "grasp_verified": False}

    def stop(self):
        self.aborted.set()  # Prevent any subsequent writes, including after a blocked RPC.
        # Stop immediately, then serialize a final stop after any in-flight RPC.
        # The node RPC has a finite timeout and is never replayed after a timeout.
        first = emergency_stop(self.config.data)
        with self._command_lock:
            final = emergency_stop(self.config.data)
        return {side: dict(final[side], initial_attempt=first[side]) for side in final}

    def close(self):
        super().close()
        if self.node is not None:
            self.node.close()
            self.node._context.term()
            self.node = None
