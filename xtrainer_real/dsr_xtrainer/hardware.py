"""Official low-level SDK with strict framing, telemetry and explicit actuation.

Do NOT instantiate official DobotRobot: its constructor enables the robot, sets
TCP values and its gripper constructor writes hardware settings. Only the
side-effect-free message methods and Feetech packet primitives are reused here.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import numpy as np
from .config import KIT_ROOT, SDK_REVISION, kit_path, read_json
from .geometry import rotation_distance, transform_pose, vector
from .model import PreparationError
from .safety import check_robot_state, check_segment


def parse_reply(reply, command):
    match = re.fullmatch(r"\s*(-?\d+)\s*,\s*(\{.*\})\s*,\s*(.*?)\s*;\s*", reply, re.S)
    if not match or match[3].strip() != command.strip():
        raise PreparationError(f"Malformed or mismatched Dobot reply to {command}: {reply!r}")
    if int(match[1]) != 0:
        raise PreparationError(f"Dobot rejected {command}: {reply.strip()}")
    return match[2]


def numeric_reply(reply, command, n):
    payload = parse_reply(reply, command)
    try:
        values = [float(x.strip()) for x in payload[1:-1].split(",")]
    except ValueError as exc:
        raise PreparationError("Non-numeric controller reply: "+reply) from exc
    return vector(values, n, command+" reply")


class StrictTransport:
    """Override the SDK's swallowed exceptions and one-recv reply handling."""
    def __init__(self, ip, port, connect_timeout, reply_timeout):
        self.ip, self.port = ip, port
        self.socket_dobot = None
        self._io_lock = threading.Lock()
        self._received = b""
        self.reply_timeout = reply_timeout
        self.socket_dobot = socket.create_connection((ip, port), timeout=connect_timeout)
        self.socket_dobot.settimeout(reply_timeout)

    def sendRecvMsg(self, command):
        with self._io_lock:
            try:
                self.socket_dobot.sendall(command.encode("ascii"))
                deadline = time.monotonic()+self.socket_dobot.gettimeout()
                while b";" not in self._received:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("incomplete reply deadline")
                    self.socket_dobot.settimeout(remaining)
                    part = self.socket_dobot.recv(4096)
                    if not part:
                        raise ConnectionError("controller closed socket")
                    self._received += part
                    if len(self._received) > 65536:
                        raise PreparationError("Controller reply exceeds 64 KiB")
                raw, self._received = self._received.split(b";", 1)
                reply = (raw+b";").decode("ascii")
                parse_reply(reply, command)
                return reply
            except (OSError, UnicodeError) as exc:
                raise PreparationError(f"{self.ip}:{self.port}: failed {command}: {exc}") from exc
            finally:
                if self.socket_dobot:
                    self.socket_dobot.settimeout(self.reply_timeout)

    def close(self):
        if self.socket_dobot is not None:
            self.socket_dobot.close()
            self.socket_dobot = None


def load_sdk(config):
    sdk_path = kit_path(config.data.get("sdk", {}).get("root"))
    lock = read_json(KIT_ROOT/"sdk.lock.json")
    try:
        revision = subprocess.check_output(["git", "-C", str(sdk_path), "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise PreparationError("Official SDK is absent; run scripts/install.sh") from exc
    if revision != SDK_REVISION or lock.get("commit") != revision:
        raise PreparationError("SDK revision differs from sdk.lock.json; use the pinned install script")
    for relative, digest in lock["files_sha256"].items():
        p = sdk_path/relative
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise PreparationError("SDK file missing or modified: "+str(p))
    sys.path.insert(0, str(sdk_path))
    try:
        api = importlib.import_module("dobot_control.robots.dobot_api")
        feetech = importlib.import_module("third_party.feetech.scservo_sdk")
    except ImportError as exc:
        raise PreparationError(f"Missing hardware dependency: {exc}. Run scripts/install.sh; tkinter requires python3-tk on some hosts.") from exc
    if (Path(api.__file__).resolve() != (sdk_path/"dobot_control/robots/dobot_api.py").resolve() or
            Path(feetech.__file__).resolve() != (sdk_path/"third_party/feetech/scservo_sdk/__init__.py").resolve()):
        raise PreparationError("A different Dobot/Feetech module was already imported; use a clean process with the pinned SDK")

    class Dashboard(StrictTransport, api.DobotApiDashboard):
        pass

    class Move(StrictTransport, api.DobotApiMove):
        pass

    if api.MyType.itemsize != 1440:
        raise PreparationError("Unsupported feedback packet size")
    return Dashboard, Move, api.MyType, feetech


def recv_exact(sock, count):
    data = bytearray()
    deadline = time.monotonic()+sock.gettimeout()
    original_timeout = sock.gettimeout()
    try:
        while len(data) < count:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError("feedback frame deadline")
            sock.settimeout(remaining)
            part = sock.recv(count-len(data))
            if not part:
                raise ConnectionError("feedback socket closed")
            data.extend(part)
        return bytes(data)
    finally:
        sock.settimeout(original_timeout)


class Feedback:
    def __init__(self, ip, port, dtype, timeout, connect_timeout):
        self.dtype, self.timeout = dtype, timeout
        self.sock = socket.create_connection((ip, port), timeout=connect_timeout)
        self.sock.settimeout(timeout)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._data = None
        self._error = None
        self._sequence = 0
        self.thread = threading.Thread(target=self._read, daemon=True, name="xtrainer-feedback-"+ip)
        self.thread.start()

    def _read(self):
        try:
            while not self._stop.is_set():
                raw = recv_exact(self.sock, self.dtype.itemsize)
                entry = np.frombuffer(raw, dtype=self.dtype)[0]
                if int(entry["len"]) != 1440 or int(entry["test_value"]) != 0x123456789ABCDEF:
                    raise PreparationError("Unsupported/corrupted feedback frame; confirm firmware protocol")
                with self._lock:
                    self._sequence += 1
                    self._data = (entry.copy(), time.monotonic(), self._sequence)
        except Exception as exc:
            if not self._stop.is_set():
                with self._lock:
                    self._error = str(exc)

    def latest(self):
        with self._lock:
            if self._error:
                raise PreparationError("Feedback failure: "+self._error)
            if self._data is None:
                raise PreparationError("Feedback not received")
            data, received, sequence = self._data
            if time.monotonic()-received > self.timeout:
                raise PreparationError("Stale feedback")
            return data.copy(), sequence

    def wait_first(self):
        end = time.monotonic()+self.timeout
        while time.monotonic() < end:
            with self._lock:
                ready = self._data is not None or self._error is not None
            if ready:
                return self.latest()
            time.sleep(0.01)
        raise PreparationError("No initial robot feedback")

    def close(self):
        self._stop.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self.thread.join(timeout=2)


class Gripper:
    """Opening/reading is read-only. All register writes require execute=True."""
    def __init__(self, spec, sdk, execute=False):
        self.spec, self.execute = spec, execute
        self.port = sdk.PortHandler(spec["port"])
        if not self.port.setBaudRate(spec["baudrate"]):
            raise PreparationError("Cannot open gripper serial port: "+spec["port"])
        self.servo = sdk.sms_sts(self.port)
        try:
            _, comm, error = self.servo.ping(spec["id"])
            self.check(comm, error)
            self.position()
        except BaseException:
            self.close()
            raise

    def check(self, comm, error):
        if comm != 0 or error != 0:
            raise PreparationError(f"Gripper {self.spec['port']}: communication={comm}, servo error={error}")

    def position(self):
        value, comm, error = self.servo.ReadPos(self.spec["id"])
        self.check(comm, error)
        if not 0 <= value <= 4095:
            raise PreparationError("Invalid gripper position feedback")
        return value

    def prepare(self):
        if not self.execute:
            raise PreparationError("Gripper register writes require --execute")
        self.check(*self.servo.write2ByteTxRx(self.spec["id"], 48, self.spec["torque_limit"]))
        self.check(*self.servo.write1ByteTxRx(self.spec["id"], 40, 1))

    def move(self, opening, heartbeat):
        if not self.execute:
            raise PreparationError("Gripper movement requires --execute")
        target = round(self.spec["closed_ticks"]+opening*(self.spec["open_ticks"]-self.spec["closed_ticks"]))
        self.check(*self.servo.WritePosEx(self.spec["id"], target, self.spec["speed"], self.spec["acceleration"]))
        start, settled = time.monotonic(), 0
        last = self.position()
        while time.monotonic()-start < 8:
            heartbeat()
            position = self.position()
            moving, comm, error = self.servo.ReadMoving(self.spec["id"])
            self.check(comm, error)
            settled = settled+1 if not moving and abs(position-last) <= 3 else 0
            last = position
            if time.monotonic()-start >= self.spec["settle_s"] and settled >= 3:
                # A closed gripper may stop on an object. Grasp is checked separately.
                if opening >= 0.99 and abs(position-target) > self.spec["position_tolerance_ticks"]:
                    raise PreparationError("Gripper did not reach its calibrated open position")
                return {"target_ticks": target, "measured_ticks": position, "grasp_verified": False}
            time.sleep(0.04)
        raise PreparationError("Gripper motion timed out")

    def close(self):
        if self.port and self.port.is_open:
            self.port.closePort()


class RealRig:
    def __init__(self, config, execute=False):
        config.require_hardware(execute)
        self.config, self.execute = config, execute
        self.dashboards, self.moves, self.feedback, self.grippers = {}, {}, {}, {}
        self.sdk = load_sdk(config)

    def connect(self):
        dashboard, move, dtype, feetech = self.sdk
        s = self.config.safety
        try:
            for side, arm in self.config.arms.items():
                args = (s["connect_timeout_s"], s["reply_timeout_s"])
                self.dashboards[side] = dashboard(arm["ip"], arm["dashboard_port"], *args)
                self.dashboards[side].reply_timeout = s["reply_timeout_s"]
                self.feedback[side] = Feedback(arm["ip"], arm["feedback_port"], dtype, s["feedback_timeout_s"], s["connect_timeout_s"])
                self.feedback[side].wait_first()
                if self.execute:
                    self.moves[side] = move(arm["ip"], arm["move_port"], *args)
                    self.moves[side].reply_timeout = s["reply_timeout_s"]
                    self.grippers[side] = Gripper(arm["gripper"], feetech, execute=True)
        except BaseException:
            self.close()
            raise

    def states(self):
        result = {}
        for side, stream in self.feedback.items():
            sample, sequence = stream.latest()
            arm, f = self.config.arms[side], self.config.frames[side]
            if int(sample["user"]) != arm["user_index"] or int(sample["tool"]) != arm["tool_index"]:
                raise PreparationError(side+": controller User/Tool differs from calibration; select the measured frame on site")
            tcp = f.get("tool_tcp_controller_mm_deg")
            if tcp is not None:
                actual_tcp = vector(sample["tool[6]"], 6, "feedback TCP calibration")
                desired_tcp = vector(tcp, 6, "measured TCP calibration")
                if np.linalg.norm(actual_tcp[:3]-desired_tcp[:3]) > self.config.safety["position_tolerance_mm"] or rotation_distance(actual_tcp[3:], desired_tcp[3:]) > self.config.safety["rotation_tolerance_deg"]:
                    raise PreparationError(side+": controller TCP value differs from measured calibration")
            result[side] = {"mode": int(sample["robot_mode"]), "error": int(sample["error_status"]),
                            "drag": int(sample["drag_status"]), "paused": int(sample["pause_cmd_flag"]),
                            "joints_deg": vector(sample["q_actual"], 6, "feedback joints").tolist(),
                            "pose_world_mm_deg": transform_pose(np.asarray(f["world_T_base"]), sample["tool_vector_actual"]).tolist(),
                            "sequence": sequence, "tool": int(sample["tool"]), "user": int(sample["user"])}
        return result

    def queries(self):
        """Additional read-only dashboard validation for the preflight command."""
        result = {}
        for side, dash in self.dashboards.items():
            result[side] = {"joints_deg": numeric_reply(dash.GetAngle(), "GetAngle()", 6).tolist(),
                            "pose_base_mm_deg": numeric_reply(dash.GetPose(), "GetPose()", 6).tolist(),
                            "mode": numeric_reply(dash.RobotMode(), "RobotMode()", 1).tolist(),
                            "alarms": parse_reply(dash.GetErrorID(), "GetErrorID()")}
        return result

    def prepare(self):
        if not self.execute:
            raise PreparationError("Enabling/configuring robots requires --execute")
        states = self.states()
        for side, state in states.items():
            if state["mode"] not in {4, 5} or state["error"] or state["drag"] or state["paused"]:
                raise PreparationError(side+": initial mode must be idle enabled (5) or disabled (4), without alarms")
            park = self.config.park(side)
            p = np.asarray(state["pose_world_mm_deg"])
            if np.linalg.norm(p[:3]-park[:3]) > self.config.safety["position_tolerance_mm"] or rotation_distance(p[3:], park[3:]) > self.config.safety["rotation_tolerance_deg"]:
                raise PreparationError(side+": manually place the robot at its calibrated park pose first; no automatic homing is performed")
            if np.max(np.abs(np.asarray(state["joints_deg"])-self.config.frames[side]["park_joints_deg"])) > self.config.safety["park_joint_tolerance_deg"]:
                raise PreparationError(side+": measured park joints do not match calibration")
        s = self.config.safety
        for side, dash in self.dashboards.items():
            # Never ClearError, SetTool, ResetRobot or disable collision protection.
            dash.SpeedFactor(s["speed_percent"])
            dash.SpeedJ(s["speed_percent"])
            dash.SpeedL(s["speed_percent"])
            dash.AccJ(s["acceleration_percent"])
            dash.AccL(s["acceleration_percent"])
            if states[side]["mode"] == 4:
                dash.EnableRobot()
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            states = self.states()
            if all(s["mode"] == 5 for s in states.values()):
                check_robot_state(self.config, states)
                for gripper in self.grippers.values():
                    gripper.prepare()
                return
            time.sleep(0.05)
        raise PreparationError("Robots did not become enabled/idle")

    def move(self, side, target, heartbeat):
        if not self.execute:
            raise PreparationError("Robot motion requires --execute")
        other = "right" if side == "left" else "left"
        states = heartbeat(parked_side=other)
        check_segment(self.config, side, states[side]["pose_world_mm_deg"], target, states[other]["pose_world_mm_deg"])
        initial_sequence = states[side]["sequence"]
        base = transform_pose(np.linalg.inv(np.asarray(self.config.frames[side]["world_T_base"])), target)
        self.moves[side].MovL(*base)
        end = time.monotonic()+self.config.safety["motion_timeout_s"]
        stable, sequence = 0, initial_sequence
        while time.monotonic() < end:
            states = heartbeat(parked_side=other)
            live = states[side]
            if live["sequence"] > sequence:
                sequence = live["sequence"]
                p = np.asarray(live["pose_world_mm_deg"])
                arrived = np.linalg.norm(p[:3]-np.asarray(target)[:3]) <= self.config.safety["position_tolerance_mm"]
                arrived &= rotation_distance(p[3:], target[3:]) <= self.config.safety["rotation_tolerance_deg"]
                stable = stable+1 if arrived and live["mode"] == 5 else 0
                if stable >= 3:
                    return live
            time.sleep(0.03)
        raise PreparationError(side+": motion timed out; acknowledgement did not establish completion")

    def grip(self, side, opening, heartbeat):
        if not self.execute:
            raise PreparationError("Gripper motion requires --execute")
        other = "right" if side == "left" else "left"
        return self.grippers[side].move(opening, lambda: heartbeat(parked_side=other))

    def close(self):
        # Disconnecting never disables robots or opens grippers: an object may be held.
        for group in [self.grippers, self.feedback, self.moves, self.dashboards]:
            for obj in group.values():
                try:
                    obj.close()
                except Exception:
                    pass
            group.clear()


def emergency_stop(config_data):
    """Independent fresh dashboard sockets: a blocked motion socket cannot hold these."""
    def stop(side, arm):
        channel = None
        try:
            channel = StrictTransport(arm["ip"], int(arm.get("dashboard_port", 29999)), 1, 1)
            channel.reply_timeout = 1
            channel.sendRecvMsg("EmergencyStop()")
            return side, {"acknowledged": True}
        except Exception as exc:
            return side, {"acknowledged": False, "error": str(exc)}
        finally:
            if channel:
                channel.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(stop, side, arm) for side, arm in config_data["arms"].items()]
        return dict(f.result() for f in futures)
