"""Offline tests of task semantics, actuation gates and concrete fault paths."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from dsr_xtrainer.calibration import fit_rigid
from dsr_xtrainer.cli import main
from dsr_xtrainer.config import Config, KIT_ROOT
from dsr_xtrainer.executor import Executor, Journal, MockRig, utc_now, write_json
from dsr_xtrainer.geometry import matrix, rotation_distance, rpy_matrix, transform_pose
from dsr_xtrainer.hardware import Gripper, RealRig, StrictTransport, load_sdk, numeric_reply, parse_reply, recv_exact
from dsr_xtrainer.model import PreparationError
from dsr_xtrainer.perception import check_observation, load_scene
from dsr_xtrainer.safety import check_robot_state, check_segment, validate_plan
from dsr_xtrainer.tasks import build_plan


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(KIT_ROOT/"examples/site.demo.json")

    def scene(self, task):
        return load_scene(KIT_ROOT/("examples/"+task+".scene.json"), self.config, task)

    def test_cover_remembers_color_before_occlusion(self):
        scene = self.scene("cover_blocks")
        plan = build_plan(self.config, scene, "cover_blocks")
        self.assertEqual(plan.memory["cover_order"], ["green", "blue", "red"])
        self.assertEqual(plan.memory["uncover_order"], ["red", "green", "blue"])
        checkpoints = [s for s in plan.steps if s.kind == "checkpoint" and s.expected["kind"] == "cover_state"]
        self.assertEqual([s.expected["covered_labels"] for s in checkpoints],
                         [["green"], ["green", "blue"], ["green", "blue", "red"], ["green", "blue"], ["blue"], []])
        self.assertEqual(plan.memory["assignment"]["red"]["cup"], "cup2")
        self.assertEqual(len([s for s in plan.steps if s.kind == "checkpoint"]), 12)

    def test_align_uses_ruler_and_keeps_blocks_on_table(self):
        plan = build_plan(self.config, self.scene("align_blocks"), "align_blocks")
        self.assertAlmostEqual(plan.memory["sweep_mm"], 36)
        sweep = [s.pose_world_mm_deg for s in plan.steps if s.stage == "ruler.sweep"]
        self.assertTrue(sweep)
        self.assertTrue(all(p[0] == 0 and p[2] == 100 for p in sweep))
        self.assertTrue(all(s.arm in {"left", None} for s in plan.steps))
        self.assertEqual(plan.steps[-1].expected["kind"], "alignment")

    def test_bad_ruler_geometry_and_cup_clearance_rejected(self):
        scene = self.scene("align_blocks")
        scene["objects"]["straightedge"]["length_mm"] = 40
        with self.assertRaisesRegex(PreparationError, "span"):
            build_plan(self.config, scene, "align_blocks")
        scene = self.scene("cover_blocks")
        scene["objects"]["cup0"]["inner_diameter_mm"] = 40
        with self.assertRaisesRegex(PreparationError, "clear"):
            build_plan(self.config, scene, "cover_blocks")
        scene = self.scene("cover_blocks")
        scene["objects"]["cup0"]["opening_down"] = False
        with self.assertRaisesRegex(PreparationError, "inverted"):
            build_plan(self.config, scene, "cover_blocks")

    def test_both_tasks_execute_mock_without_network_or_serial(self):
        with tempfile.TemporaryDirectory() as tmp, patch("socket.create_connection", side_effect=AssertionError("unexpected network")):
            for name in ["align_blocks", "cover_blocks"]:
                scene = self.scene(name)
                plan = build_plan(self.config, scene, name)
                rig = MockRig(self.config)
                journal = Journal(tmp, name)
                summary = Executor(self.config, rig, journal, scene, mock=True).run(plan)
                self.assertTrue(summary["execution_completed"])
                self.assertEqual(summary["steps_completed"], len(plan.steps))
                self.assertEqual(summary["task_verified_by"], "mock_only")
                self.assertIsNone(summary["real_robot_success_rate"])
                for side in rig.poses:
                    np.testing.assert_allclose(rig.poses[side], self.config.park(side))
                self.assertEqual(len(list((journal.path/"checkpoints").glob("*.accepted.json"))),
                                 sum(s.kind == "checkpoint" for s in plan.steps))

    def test_execution_fault_stops_both_and_keeps_failure_receipt(self):
        class FailingRig(MockRig):
            def move(self, side, target, heartbeat):
                if len(self.commands) > 5:
                    raise PreparationError("injected lost feedback")
                return super().move(side, target, heartbeat)
        scene = self.scene("align_blocks")
        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(tmp, "align_blocks")
            rig = FailingRig(self.config)
            with self.assertRaisesRegex(PreparationError, "lost feedback"):
                Executor(self.config, rig, journal, scene, mock=True).run(build_plan(self.config, scene, "align_blocks"))
            self.assertTrue(rig.stopped)
            summary = json.loads((journal.path/"summary.json").read_text())
            self.assertFalse(summary["execution_completed"])
            events = [json.loads(line) for line in (journal.path/"events.jsonl").read_text().splitlines()]
            stop = next(e for e in events if e["kind"] == "emergency_stop")
            self.assertEqual(set(stop["arms"]), {"left", "right"})

    def test_default_launcher_is_plan_only(self):
        with tempfile.TemporaryDirectory() as tmp, patch("socket.create_connection", side_effect=AssertionError("network")), redirect_stdout(io.StringIO()):
            for task in ["align_blocks", "cover_blocks"]:
                code = main(["run", "--task", task, "--output-root", tmp])
                self.assertEqual(code, 0)
            self.assertEqual(len(list(Path(tmp).glob("*/plan.json"))), 2)
            self.assertFalse(list(Path(tmp).glob("*/events.jsonl")))

    def test_execute_uses_local_configuration_and_rejects_demo_before_connect(self):
        with tempfile.TemporaryDirectory() as tmp, patch("socket.create_connection", side_effect=AssertionError("network")), redirect_stderr(io.StringIO()):
            with patch("dsr_xtrainer.cli.Config", side_effect=PreparationError("missing on-site configuration")) as constructor:
                self.assertEqual(main(["run", "--task", "align_blocks", "--execute", "--output-root", tmp]), 2)
                self.assertEqual(constructor.call_args.args[0], str(KIT_ROOT/"local/site.json"))
            self.assertEqual(main(["run", "--task", "align_blocks", "--execute", "--config", str(KIT_ROOT/"examples/site.demo.json"),
                                   "--output-root", tmp]), 2)

    def test_both_arm_feedback_and_park_joint_checks(self):
        states = MockRig(self.config).states()
        with self.assertRaisesRegex(PreparationError, "both left and right"):
            check_robot_state(self.config, {"left": states["left"]})
        states["right"]["joints_deg"][0] += 2
        with self.assertRaisesRegex(PreparationError, "inactive arm"):
            check_robot_state(self.config, states, parked_side="right")
        rig = RealRig.__new__(RealRig)
        rig.config, rig.execute = self.config, True
        rig.states = lambda: states
        rig.dashboards = {}  # any controller command here would be a test failure
        with self.assertRaisesRegex(PreparationError, "park joints"):
            rig.prepare()

    def test_demo_scene_config_and_stale_observations_block_hardware(self):
        with self.assertRaisesRegex(PreparationError, "cannot connect"):
            self.config.require_hardware(execute=True)
        with self.assertRaisesRegex(PreparationError, "Example scenes"):
            load_scene(KIT_ROOT/"examples/align_blocks.scene.json", self.config, "align_blocks", hardware=True)
        scene = self.scene("align_blocks")
        scene["example_only"] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"stale.json"
            write_json(path, scene)
            with self.assertRaisesRegex(PreparationError, "stale"):
                load_scene(path, self.config, "align_blocks", hardware=True)

    def test_observations_require_correct_order_and_no_large_block_motion(self):
        scene = self.scene("cover_blocks")
        expected = {"kind": "cover_state", "covered_labels": ["green"]}
        record = {"covered_labels": ["red"], "all_cups_opening_down": True,
                  "block_positions_world_mm": {label: obj["position_world_mm"].copy() for label, obj in scene["objects"].items() if label in {"red", "green", "blue"}}}
        with self.assertRaisesRegex(PreparationError, "covered"):
            check_observation(expected, record, scene)
        record["covered_labels"] = ["green"]
        record["block_positions_world_mm"]["red"][0] += 50
        with self.assertRaisesRegex(PreparationError, "displacement"):
            check_observation(expected, record, scene)

    def test_geometry_gate_checks_segments_not_only_endpoints(self):
        self.config.safety["forbidden_boxes_world_mm"] = [{"name": "thin obstacle", "min": [-1, 99, 90], "max": [1, 101, 110]}]
        self.config.safety["tcp_padding_mm"] = 0
        with self.assertRaisesRegex(PreparationError, "intersects"):
            check_segment(self.config, "left", [-5,100,100,180,0,0], [5,100,100,180,0,0], self.config.park("right"))
        self.config.safety["forbidden_boxes_world_mm"] = []
        with self.assertRaisesRegex(PreparationError, "other arm"):
            check_segment(self.config, "left", [-5,100,100,180,0,0], [5,100,100,180,0,0], [0,110,100,180,0,0])
        states = MockRig(self.config).states()
        states["right"]["error"] = 1
        with self.assertRaisesRegex(PreparationError, "alarm"):
            check_robot_state(self.config, states)

    def test_stale_checkpoint_cannot_be_reused_for_new_run(self):
        scene = self.scene("align_blocks")
        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(tmp, "align_blocks")
            rig = MockRig(self.config)
            executor = Executor(self.config, rig, journal, scene, mock=False)
            step = build_plan(self.config, scene, "align_blocks").steps[-1]
            write_json(journal.path/"checkpoints"/(step.stage+".json"),
                       {"schema":"dsr.xtrainer-checkpoint/v1", "run_id":"previous-run", "stage":step.stage})
            with self.assertRaisesRegex(PreparationError, "identity"):
                executor.checkpoint(step)


class GeometryTests(unittest.TestCase):
    def test_world_base_roundtrip_and_rotation_wrap(self):
        t = np.eye(4)
        t[:3,:3] = rpy_matrix([0,0,90])
        t[:3,3] = [100,200,300]
        pose = [10,20,30,180,0,170]
        out = transform_pose(t, pose)
        back = transform_pose(np.linalg.inv(t), out)
        np.testing.assert_allclose(back[:3], pose[:3], atol=1e-9)
        self.assertAlmostEqual(rotation_distance(back[3:], pose[3:]), 0, places=5)
        self.assertAlmostEqual(rotation_distance([0,0,179], [0,0,-179]), 2, places=6)
        with self.assertRaisesRegex(PreparationError, "right handed"):
            matrix(np.diag([1,1,-1,1]), "reflected")

    def test_rigid_frame_fit_and_degenerate_points(self):
        a = [[0,0,0],[100,0,0],[0,100,0],[0,0,100]]
        t = np.eye(4)
        t[:3,:3] = rpy_matrix([0,0,30])
        t[:3,3] = [30,10,20]
        b = (t[:3,:3]@np.array(a).T).T+t[:3,3]
        fit = fit_rigid(a,b)
        np.testing.assert_allclose(fit["world_T_base"], t, atol=1e-8)
        with self.assertRaisesRegex(PreparationError, "collinear"):
            fit_rigid([[i,0,0] for i in range(4)], [[i,1,1] for i in range(4)])


class FakeSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.timeout = 2
        self.sent = []
        self.closed = False

    def settimeout(self, value):
        self.timeout = value

    def gettimeout(self):
        return self.timeout

    def recv(self, count):
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if isinstance(chunk, Exception):
            raise chunk
        result = chunk[:count]
        if chunk[count:]:
            self.chunks.insert(0, chunk[count:])
        return result

    def sendall(self, command):
        self.sent.append(command)

    def close(self):
        self.closed = True


class ProtocolTests(unittest.TestCase):
    def test_fragmented_reply_and_exact_command_matching(self):
        fake = FakeSocket([b"0,{1,2,", b"3,4,5,6},GetAngle()", b";"])
        with patch("socket.create_connection", return_value=fake):
            channel = StrictTransport("unused", 29999, 1, 1)
            reply = channel.sendRecvMsg("GetAngle()")
            np.testing.assert_equal(numeric_reply(reply, "GetAngle()", 6), [1,2,3,4,5,6])
            self.assertEqual(fake.sent, [b"GetAngle()"])
            channel.close()
        for reply in ["", "0,{},EnableRobot();", "-1,{},GetAngle();", "0,{nan},GetAngle();"]:
            with self.assertRaises(PreparationError):
                numeric_reply(reply, "GetAngle()", 6)

    def test_timeout_and_mid_packet_disconnect_raise(self):
        fake = FakeSocket([b"0,{", socket.timeout("test timeout")])
        with patch("socket.create_connection", return_value=fake):
            channel = StrictTransport("unused", 29999, 1, 1)
            with self.assertRaisesRegex(PreparationError, "failed"):
                channel.sendRecvMsg("GetAngle()")
            channel.close()
        with self.assertRaises(ConnectionError):
            recv_exact(FakeSocket([b"a", b"b"]), 4)

    def test_pinned_sdk_imports_and_official_command_serialization(self):
        config = Config(KIT_ROOT/"examples/site.demo.json")
        dashboard, move, dtype, _ = load_sdk(config)
        self.assertEqual(dtype.itemsize, 1440)
        fake = FakeSocket([b"0,{},SpeedFactor(5);"])
        with patch("socket.create_connection", return_value=fake):
            d = dashboard("unused", 29999, 1, 1)
            d.SpeedFactor(5)
            self.assertEqual(fake.sent, [b"SpeedFactor(5)"])
            d.close()
        command = "MovL(1.000000,2.000000,3.000000,180.000000,0.000000,0.000000)"
        fake = FakeSocket([("0,{},"+command+";").encode()])
        with patch("socket.create_connection", return_value=fake), redirect_stdout(io.StringIO()):
            m = move("unused", 30003, 1, 1)
            m.MovL(1,2,3,180,0,0)
            self.assertEqual(fake.sent, [command.encode()])
            m.close()

    def test_readonly_gripper_never_writes_registers(self):
        class Port:
            is_open = False
            def __init__(self, name):
                pass
            def setBaudRate(self, value):
                self.is_open = True
                return True
            def closePort(self):
                self.is_open = False
        class Servo:
            def __init__(self, port):
                pass
            def ping(self, address):
                return 1,0,0
            def ReadPos(self, address):
                return 2100,0,0
        class SDK:
            PortHandler, sms_sts = Port, Servo
        config = Config(KIT_ROOT/"examples/site.demo.json")
        g = Gripper(config.arms["left"]["gripper"], SDK, execute=False)
        self.assertEqual(g.position(), 2100)
        with self.assertRaisesRegex(PreparationError, "--execute"):
            g.prepare()
        g.close()

    def test_controller_tcp_must_match_measured_calibration(self):
        _, _, dtype, _ = load_sdk(Config(KIT_ROOT/"examples/site.demo.json"))
        config = Config(KIT_ROOT/"examples/site.demo.json")
        sample = np.zeros(1, dtype=dtype)[0]
        sample["tool"] = config.arms["left"]["tool_index"]
        class Stream:
            def latest(self):
                return sample.copy(), 1
        rig = RealRig.__new__(RealRig)
        rig.config, rig.feedback = config, {"left": Stream()}
        config.frames["left"]["tool_tcp_controller_mm_deg"] = [0,0,0,0,0,0]
        self.assertEqual(rig.states()["left"]["tool"], 1)
        sample["tool[6]"][2] = 10
        with self.assertRaisesRegex(PreparationError, "TCP value"):
            rig.states()


if __name__ == "__main__":
    unittest.main()
