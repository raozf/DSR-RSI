"""Offline contract tests for the isolated WebSocket/ZMQ adapter."""
import hashlib
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import zmq

from dsr_xtrainer.config import Config, KIT_ROOT
from dsr_xtrainer.executor import Journal, MockRig
from dsr_xtrainer.model import PreparationError
from dsr_xtrainer.perception import load_scene
from ws_bridge.protocol import accept_plan, compile_request, make_request
from ws_bridge.robot_client import ExistingClientRig, Transport, action14
from ws_bridge.sdk import load_sdk
from ws_bridge.client_adapter import ClientExecutor, Session, main
from ws_bridge.protocol import ACTION_SPACE, SCHEMA, dumps, loads
from ws_bridge.server_adapter import make_server
from dsr_xtrainer.tasks import build_plan


@contextmanager
def ws_server(handler=None):
    if handler is None:
        server = make_server(port=0)
    else:
        from websockets.sync.server import serve
        server = serve(handler, "127.0.0.1", 0, compression=None)
    with server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}/ws", server
        finally:
            server.shutdown()
            worker.join(timeout=3)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(KIT_ROOT/"examples/site.demo.json")
        self.task = "align_blocks"
        self.scene = load_scene(KIT_ROOT/"examples/align_blocks.scene.json", self.config, self.task)
        self.request = make_request(self.config, self.scene, self.task)
        self.response = compile_request(self.request)

    def test_remote_plan_is_bound_to_local_scene_and_identity(self):
        accepted = accept_plan(self.response, self.request, self.config, self.scene)
        self.assertEqual(accepted.task, self.task)
        bad = dict(self.response, request_id="f"*32)
        with self.assertRaisesRegex(PreparationError, "identity"):
            accept_plan(bad, self.request, self.config, self.scene)
        bad = dict(self.response, plan_sha256="0"*64)
        with self.assertRaisesRegex(PreparationError, "checksum"):
            accept_plan(bad, self.request, self.config, self.scene)
        bad_plan = json.loads(json.dumps(self.response))
        bad_plan["plan"]["memory"]["sweep_mm"] += 1
        from ws_bridge.protocol import digest
        bad_plan["plan_sha256"] = digest(bad_plan["plan"])
        with self.assertRaisesRegex(PreparationError, "differs"):
            accept_plan(bad_plan, self.request, self.config, self.scene)

    def test_stale_or_malformed_request_rejected(self):
        stale = dict(self.request, captured_at="2000-01-01T00:00:00+00:00")
        with self.assertRaisesRegex(PreparationError, "Stale"):
            compile_request(stale)
        bad = dict(self.request, sequence=True)
        with self.assertRaisesRegex(PreparationError, "schema"):
            compile_request(bad)

    def test_units_nan_and_recomputed_malicious_plan_rejected(self):
        bad = dict(self.response, action_space="joint_radians_14d")
        with self.assertRaisesRegex(PreparationError, "action space"):
            accept_plan(bad, self.request, self.config, self.scene)
        with self.assertRaises(ValueError):
            loads('{"q":NaN}')
        projection = self.request["dsr"]["planning_config"]
        self.assertNotIn("sdk", projection)
        self.assertNotIn("ip", dumps(projection["arms"]))

    def test_real_websocket_health_and_plan_only_default(self):
        with ws_server() as (url, server), tempfile.TemporaryDirectory() as tmp:
            with Session(url) as session:
                self.assertEqual(session.health()["action_space"], ACTION_SPACE)
                for task in ("align_blocks", "cover_blocks"):
                    scene = load_scene(KIT_ROOT/("examples/"+task+".scene.json"), self.config, task)
                    req = make_request(self.config, scene, task)
                    plan = accept_plan(session.exchange(req), req, self.config, scene)
                    self.assertGreater(len(plan.steps), 100)
                session.ws.send('{"q":NaN}')
                self.assertEqual(loads(session.ws.recv(timeout=2))["type"], "error")
            with patch("ws_bridge.client_adapter.ExistingClientRig", side_effect=AssertionError("hardware")), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--url",url,"--task","align_blocks","--output-root",tmp]), 0)
            self.assertEqual(len(list(Path(tmp).glob("*/plan.json"))), 1)
            self.assertFalse(list(Path(tmp).glob("*/events.jsonl")))

    def test_original_upstream_client_health_works_unmodified(self):
        with ws_server() as (url, server):
            port = server.socket.getsockname()[1]
            result = subprocess.run([sys.executable, str(KIT_ROOT/".deps/amsongbc_dobot_xtrainer/dobot_client/client_ws.py"),
                                     "127.0.0.1", "--port", str(port), "--check-only"],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn("PASS", result.stdout)

    def test_execute_rejects_demo_before_any_network_connection(self):
        with patch("ws_bridge.client_adapter.Session", side_effect=AssertionError("network")), redirect_stderr(io.StringIO()):
            result = main(["--task","align_blocks","--execute","--config",str(KIT_ROOT/"examples/site.demo.json")])
            self.assertEqual(result, 2)

    def test_disconnect_during_execution_stops_and_records_failure(self):
        def handler(connection):
            health_count = 0
            for raw in connection:
                request = loads(raw)
                if request["type"] == "health":
                    health_count += 1
                    if health_count >= 3:
                        connection.close(1011, "injected disconnect")
                        return
                    response = {"type":"health","status":"healthy","schema":SCHEMA,"action_space":ACTION_SPACE}
                else:
                    response = compile_request(request)
                connection.send(dumps(response))
        class SlowRig(MockRig):
            def move(self, *args):
                time.sleep(.1)
                return super().move(*args)
        with ws_server(handler) as (url, server), tempfile.TemporaryDirectory() as tmp, Session(url) as session:
            session.health()
            session.watch()
            rig, journal = SlowRig(self.config), Journal(tmp, "disconnect")
            plan = build_plan(self.config, self.scene, self.task)
            with self.assertRaises(PreparationError):
                ClientExecutor(self.config, rig, journal, self.scene, mock=True, session=session).run(plan)
            self.assertTrue(rig.stopped)
            summary = json.loads((journal.path/"summary.json").read_text())
            self.assertFalse(summary["execution_completed"])
            self.assertIn("WebSocket", summary["error"])
            events = [json.loads(line) for line in (journal.path/"events.jsonl").read_text().splitlines()]
            stops = [event for event in events if event["kind"] == "emergency_stop"]
            self.assertEqual(len(stops), 1)
            self.assertIn("WebSocket", stops[0]["reason"])

    def test_timed_out_planning_request_is_sent_once(self):
        received = []
        def handler(connection):
            received.append(loads(connection.recv()))
            time.sleep(.15)
        with ws_server(handler) as (url, server), Session(url, timeout=.03) as session:
            with self.assertRaisesRegex(PreparationError, "not be replayed"):
                session.exchange(self.request)
        self.assertEqual(len(received), 1)
        bad = dict(self.request, images=["one", "two"])
        with self.assertRaisesRegex(PreparationError, "Images"):
            compile_request(bad)

    def test_sdk_lock_matches_unmodified_upstream_source(self):
        lock = json.loads((KIT_ROOT/"ws-sdk.lock.json").read_text())
        root = KIT_ROOT/".deps/amsongbc_dobot_xtrainer"
        for relative, expected in lock["files_sha256"].items():
            self.assertEqual(hashlib.sha256((root/relative).read_bytes()).hexdigest(), expected)
        self.assertEqual(load_sdk(".deps/amsongbc_dobot_xtrainer")[2].itemsize, 1440)


class MappingTests(unittest.TestCase):
    def test_action14_left_right_radians_and_gripper_inversion(self):
        transport = Transport(KIT_ROOT/"configs/transport.template.json")
        states = {side: {"joints_deg": [180, 90, 0, -90, -180, 45]} for side in ("left", "right")}
        command = action14(states, {"left": 1, "right": 0}, transport)
        expected = np.deg2rad([180,90,0,-90,-180,45])
        np.testing.assert_allclose(command[:6], expected)
        self.assertEqual(command[6], 0)  # open in DSR -> 0 in upstream SDK
        np.testing.assert_allclose(command[7:13], expected)
        self.assertEqual(command[13], 1) # closed in DSR -> 1 in upstream SDK

    def test_zmqclientrobot_serialization_to_original_robot_node_protocol(self):
        context = zmq.Context()
        server = context.socket(zmq.REP)
        port = server.bind_to_random_port("tcp://127.0.0.1")
        captured = {}
        ready = threading.Event()
        def serve():
            for _ in range(2):
                request = pickle.loads(server.recv())
                if request["method"] == "num_dofs":
                    server.send(pickle.dumps(14))
                else:
                    captured.update(request["args"])
                    server.send(pickle.dumps(1))
                    ready.set()
        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        try:
            _, Client, _ = load_sdk(".deps/amsongbc_dobot_xtrainer")
            client = Client(host="127.0.0.1", port=port)
            client._socket.setsockopt(zmq.LINGER, 0)
            client._socket.setsockopt(zmq.RCVTIMEO, 2000)
            client._socket.setsockopt(zmq.SNDTIMEO, 2000)
            self.assertEqual(client.num_dofs(), 14)
            q = np.arange(14, dtype=float)/10
            client.command_joint_state(q, np.array([1,0]))
            self.assertTrue(ready.wait(2))
            np.testing.assert_equal(captured["joint_state"], q)
            np.testing.assert_equal(captured["flag_in"], np.array([1,0]))
            client.close()
            client._context.term()
        finally:
            server.close(0)
            context.term()
            worker.join(timeout=2)

    def test_aborted_rig_cannot_send_new_node_command(self):
        config = Config(KIT_ROOT/"examples/site.demo.json")
        rig = ExistingClientRig.__new__(ExistingClientRig)
        rig.execute = True
        rig.aborted = threading.Event()
        rig.aborted.set()
        rig.transport = type("T", (), {"data": {"control_hz": 30, "max_joint_step_deg": 1,
                                                   "max_joint_speed_deg_s": 1, "grippers": {
                                                       "left": {"open_command": 0,"closed_command": 1},
                                                       "right": {"open_command": 0,"closed_command": 1}}}})()
        rig._command_lock = threading.Lock()
        rig._last_send = 0
        rig.config = config
        rig.openings = {"left": 1, "right": 1}
        rig.node = type("N", (), {"command_joint_state": lambda *a: self.fail("unexpected command")})()
        states = {side: {"joints_deg": [0]*6} for side in ("left", "right")}
        with self.assertRaisesRegex(PreparationError, "aborted"):
            rig.send_joints("left", np.zeros(6), states, lambda **kw: states)

    def test_prepare_uses_no_enable_or_sdk_motion_methods(self):
        config = Config(KIT_ROOT/"examples/site.demo.json")
        rig = ExistingClientRig.__new__(ExistingClientRig)
        rig.execute, rig.aborted, rig.config = True, threading.Event(), config
        rig.states = MockRig(config).states
        rig.verify_node_mapping = lambda: {"verified":True}
        rig.dashboards = {}  # Any controller writes in prepare would fail the test.
        rig.prepare()
        rig.execute = False
        with self.assertRaisesRegex(PreparationError,"execute"):
            rig.prepare()

    def test_read_only_dashboard_uses_original_ik_fk_serialization(self):
        Dashboard, _, _ = load_sdk(".deps/amsongbc_dobot_xtrainer")
        class Socket:
            def __init__(self):
                self.timeout, self.sent = 1, []
            def settimeout(self, v): self.timeout = v
            def gettimeout(self): return self.timeout
            def sendall(self, v): self.sent.append(v)
            def recv(self, n): return b"0,{1,2,3,4,5,6},"+self.sent[-1]+b";"
            def close(self): pass
        fake = Socket()
        with patch("socket.create_connection", return_value=fake):
            dash = Dashboard("unused",29999,1,1)
            dash.InverseSolution(1,2,3,180,0,0,0,1)
            self.assertEqual(fake.sent[-1],b"InverseSolution(1.000000,2.000000,3.000000,180.000000,0.000000,0.000000,0,1)")
            dash.PositiveSolution(1,2,3,4,5,6,0,1)
            self.assertTrue(fake.sent[-1].startswith(b"PositiveSolution("))
            count = len(fake.sent)
            with self.assertRaisesRegex(PreparationError,"read-only"):
                dash.EnableRobot()
            self.assertEqual(len(fake.sent),count)
            dash.close()
        self.assertNotIn("dobot_control.robots.dobot",sys.modules)


class KinematicTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(KIT_ROOT/"examples/site.demo.json")
        self.transport = Transport(KIT_ROOT/"configs/transport.template.json")
        self.rig = ExistingClientRig.__new__(ExistingClientRig)
        self.rig.config, self.rig.transport = self.config, self.transport
        self.states = MockRig(self.config).states()

    def test_ik_branch_jump_and_fk_mismatch_refuse_target(self):
        class Dash:
            def __init__(self, q): self.q = q
            def InverseSolution(self, *args):
                self.last_command = "InverseSolution()"
                return "0,{"+",".join(str(x) for x in self.q)+"},InverseSolution();"
        self.rig.dashboards = {"left":Dash([30,0,0,0,0,0])}
        with self.assertRaisesRegex(PreparationError,"branch jump"):
            self.rig.solve("left",self.config.park("left"),np.zeros(6))
        self.rig.dashboards["left"] = Dash([0]*6)
        self.rig.forward = lambda side,q: self.config.park(side)+np.array([5,0,0,0,0,0])
        with self.assertRaisesRegex(PreparationError,"IK/FK"):
            self.rig.solve("left",self.config.park("left"),np.zeros(6))

    def test_interpolation_outside_cartesian_path_sends_no_command(self):
        start = self.config.park("left")
        target = start.copy()
        target[0] += 1
        self.rig.solve = lambda side, pose,q: np.array([.05,0,0,0,0,0])
        self.rig.forward = lambda side,q: start+np.array([0,5,0,0,0,0])
        self.rig.send_joints = lambda *a: self.fail("unqualified path command")
        with self.assertRaisesRegex(PreparationError,"Cartesian path"):
            self.rig.move("left",target,lambda **kw: self.states)

    def test_qualified_cartesian_waypoint_converts_and_waits_for_feedback(self):
        start = self.config.park("left")
        target = start.copy()
        target[0] += 1
        self.rig.solve = lambda side,pose,q: np.array([.15,0,0,0,0,0])
        self.rig.forward = lambda side,q: start+np.array([q[0]/.15,0,0,0,0,0])
        commands = []
        def send(side,q,states,heartbeat):
            commands.append(q.copy())
            self.states[side]["joints_deg"] = q.tolist()
            self.states[side]["pose_world_mm_deg"] = self.rig.forward(side,q).tolist()
        self.rig.send_joints = send
        self.rig.aborted = threading.Event()
        def heartbeat(**kw):
            for state in self.states.values(): state["sequence"] += 1
            return self.states
        actual = self.rig.move("left",target,heartbeat)
        self.assertEqual(len(commands),2)
        np.testing.assert_allclose(actual["pose_world_mm_deg"],target)

    def test_measured_joint_speed_budget_and_disabled_gate(self):
        rig = self.rig
        rig.execute, rig.aborted = True, threading.Event()
        rig._command_lock, rig._last_send = threading.Lock(), 0
        rig.node = SimpleNamespace(command_joint_state=lambda *a: self.fail("speed exceeded"))
        with self.assertRaisesRegex(PreparationError,"speed budget"):
            rig.send_joints("left",[.2,0,0,0,0,0],self.states,lambda **kw:self.states)
        rig.execute = False
        with self.assertRaisesRegex(PreparationError,"execute"):
            rig.send_joints("left",np.zeros(6),self.states,lambda **kw:self.states)

    def test_transport_requires_measured_calibration_but_no_serial_ports(self):
        c, t = self.config, self.transport
        c.data["example_only"] = c.calibration["example_only"] = False
        c.arms["left"]["ip"], c.arms["right"]["ip"] = "192.0.2.1", "192.0.2.2"
        t.data["example_only"] = False
        with self.assertRaisesRegex(PreparationError,"measurement"):
            t.require_hardware(c,execute=True)
        for key in ("tcp_and_frames_measured","workspace_and_paths_reviewed","grippers_calibrated"):
            c.calibration[key] = True
        for side in c.arms:
            c.frames[side]["tool_tcp_controller_mm_deg"] = [0]*6
            c.arms[side]["gripper"]["port"] = None
        t.data["gripper_mapping_verified"] = True
        t.require_hardware(c,execute=True)


if __name__ == "__main__":
    unittest.main()
