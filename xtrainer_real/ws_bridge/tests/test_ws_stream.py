"""Fault injection for feedback-gated client/server execution; no hardware."""
from contextlib import contextmanager
import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import uuid

from dsr_xtrainer.config import Config, KIT_ROOT
from dsr_xtrainer.executor import Journal, MockRig
from dsr_xtrainer.model import PreparationError
from dsr_xtrainer.perception import load_scene
from dsr_xtrainer.tasks import build_plan
from ws_bridge.client_adapter import ClientExecutor, Session
from ws_bridge.protocol import ACTION_SPACE, SCHEMA, accept_plan, compile_request, digest, dumps, loads, make_request
from ws_bridge.robot_client import Transport, action14
from ws_bridge.server_adapter import make_server
from ws_bridge.stream import STREAM_MODE, ServerEpisode, StepClient


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
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}/ws"
        finally:
            server.shutdown()
            worker.join(timeout=3)


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(KIT_ROOT/"examples/site.demo.json")
        self.task = "align_blocks"
        self.scene = load_scene(KIT_ROOT/"examples/align_blocks.scene.json", self.config, self.task)
        self.run_id = "test-run-"+uuid.uuid4().hex
        self.request = self.initial_request(self.run_id)
        self.response = compile_request(self.request)
        self.episode = ServerEpisode(self.request, self.response)
        self.rig = MockRig(self.config)
        self.plan = build_plan(self.config, self.scene, self.task)
        self.transport = Transport(KIT_ROOT/"configs/transport.template.json")

    def initial_request(self, run_id):
        return make_request(self.config, self.scene, self.task,
                            execution={"mode": STREAM_MODE, "run_id": run_id, "mock": True})

    def state14(self, states):
        return action14(states, {"left": 1, "right": 1}, self.transport).tolist()

    def step_request(self, sequence, states=None, ack=None):
        states = self.rig.states() if states is None else states
        return {"type": "inference", "schema": SCHEMA, "phase": "step",
                "stream_id": self.episode.stream_id, "parent_request_id": self.request["request_id"],
                "request_id": uuid.uuid4().hex, "sequence": sequence,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "proprio_data": [self.state14(states)], "acknowledgement": ack,
                "observation": {"source": "mock", "gripper_position_measured": False,
                                "robot_state": states}}

    def ack(self, index, step, actual=None, checkpoint=None):
        result = {"index": index, "step_sha256": digest(step), "completed": True,
                  "actual": actual if actual is not None else {"simulated": True, "opening": step.get("opening")}}
        if checkpoint is not None:
            result["checkpoint"] = checkpoint
        return result

    def issue_checkpoint(self):
        self.episode.infer(self.step_request(0))
        step0, step1 = self.response["plan"]["steps"][:2]
        self.episode.infer(self.step_request(1, ack=self.ack(0, step0)))
        return step1, {"schema": "dsr.xtrainer-checkpoint/v1", "run_id": self.run_id,
                       "stage": step1["stage"], "calibration_id": self.config.calibration["calibration_id"],
                       "captured_at": datetime.now(timezone.utc).isoformat(), "source": "mock",
                       "simulated": True, "approved": True, "grasped_object": "straightedge"}

    def direct_client(self, mutate=None):
        # Calls the actual server episode but can corrupt its wire reply.
        requests = []
        def exchange(request, timeout=None):
            requests.append(copy.deepcopy(request))
            response = copy.deepcopy(self.episode.infer(request))
            if mutate:
                mutate(response)
            return response
        response = dict(self.response, execution=self.episode.metadata)
        journal = SimpleNamespace(run_id=self.run_id, event=lambda *a, **kw: None)
        return StepClient(SimpleNamespace(exchange=exchange), self.request, response, journal, mock=True), requests

    def test_actual_loopback_full_feedback_execution_and_final_ack(self):
        with ws_server() as url, tempfile.TemporaryDirectory() as tmp, Session(url) as session:
            journal = Journal(tmp, self.task)
            request = self.initial_request(journal.run_id)
            response = session.exchange(request)
            plan = accept_plan(response, request, self.config, self.scene)
            remote = StepClient(session, request, response, journal, mock=True)
            session.watch()
            rig = MockRig(self.config)
            summary = ClientExecutor(self.config, rig, journal, self.scene, mock=True,
                                     session=session, remote=remote).run(plan)
            self.assertTrue(summary["execution_completed"])
            self.assertTrue(summary["remote_completion_acknowledged"])
            self.assertEqual(summary["remote_policy_calls"], len(plan.steps)+1)
            self.assertEqual(len(rig.commands), sum(step.kind != "checkpoint" for step in plan.steps))
            events = [json.loads(line) for line in (journal.path/"events.jsonl").read_text().splitlines()]
            remote_events = [event for event in events if event["kind"] == "remote_inference"]
            self.assertEqual(len(remote_events), len(plan.steps)+1)
            self.assertEqual(remote_events[-1]["response"]["skill_actions"], [])
            self.assertTrue(remote_events[-1]["response"]["done"])
            self.assertEqual(session.exchange(remote_events[-1]["request"])["type"], "error")

    def test_rejects_out_of_order_stale_and_joint_unit_mismatch(self):
        for field, value, message in [
            ("sequence", 1, "sequence"),
            ("sequence", False, "sequence"),
            ("captured_at", (datetime.now(timezone.utc)-timedelta(seconds=8)).isoformat(), "fresh"),
            ("proprio_data", [[.5]*14], "does not match"),
        ]:
            with self.subTest(field=field, value=value):
                self.episode = ServerEpisode(self.request, self.response)
                request = self.step_request(0)
                request[field] = value
                with self.assertRaisesRegex(PreparationError, message):
                    self.episode.infer(request)
                self.assertTrue(self.episode.closed)
                with self.assertRaisesRegex(PreparationError, "closed"):
                    self.episode.infer(self.step_request(0))

    def test_receipt_is_required_exact_and_single_use(self):
        step0 = self.response["plan"]["steps"][0]
        for change in [None, {"step_sha256": "0"*64}, {"index": 1}, {"completed": False},
                       {"actual": {"opening": 1}}]:
            with self.subTest(change=change):
                self.episode = ServerEpisode(self.request, self.response)
                first = self.step_request(0)
                self.episode.infer(first)
                ack = None if change is None else dict(self.ack(0, step0), **change)
                with self.assertRaises(PreparationError):
                    self.episode.infer(self.step_request(1, ack=ack))
                self.assertTrue(self.episode.closed)
        self.episode = ServerEpisode(self.request, self.response)
        first = self.step_request(0)
        self.episode.infer(first)
        with self.assertRaisesRegex(PreparationError, "sequence"):
            self.episode.infer(first)

    def test_checkpoint_evidence_is_verified_before_server_advances(self):
        for change, message in [
            ({"grasped_object": "wrong-object"}, "grasp"),
            ({"run_id": "another-run"}, "identity"),
            ({"source": "vision", "simulated": False}, "source"),
            ({"captured_at": (datetime.now(timezone.utc)-timedelta(seconds=10)).isoformat()}, "predates"),
        ]:
            with self.subTest(change=change):
                self.episode = ServerEpisode(self.request, self.response)
                step, record = self.issue_checkpoint()
                record.update(change)
                with self.assertRaisesRegex(PreparationError, message):
                    self.episode.infer(self.step_request(2, ack=self.ack(1, step, {"checkpoint": "accepted"}, record)))
                self.assertTrue(self.episode.closed)

    def test_completion_receipt_cannot_replace_measured_tcp_arrival(self):
        step1, record = self.issue_checkpoint()
        response = self.episode.infer(self.step_request(2, ack=self.ack(1, step1, {"checkpoint": "accepted"}, record)))
        move = response["skill_actions"][0]
        self.assertEqual(move["kind"], "move")
        with self.assertRaisesRegex(PreparationError, "measured TCP"):
            self.episode.infer(self.step_request(3, ack=self.ack(2, move, self.rig.states()["left"])))

    def test_stationary_arm_and_new_feedback_are_required(self):
        first = self.step_request(0)
        self.episode.infer(first)
        ack = self.ack(0, self.response["plan"]["steps"][0])
        with self.assertRaisesRegex(PreparationError, "sequence"):
            self.episode.infer(self.step_request(1, states=first["observation"]["robot_state"], ack=ack))
        self.episode = ServerEpisode(self.request, self.response)
        self.episode.infer(self.step_request(0))
        states = self.rig.states()
        states["right"]["joints_deg"][0] += 5
        with self.assertRaisesRegex(PreparationError, "inactive arm"):
            self.episode.infer(self.step_request(1, states=states, ack=ack))

    def test_malformed_feedback_gets_error_and_closes_stream(self):
        with ws_server() as url, Session(url) as session:
            response = session.exchange(self.request)
            request = self.step_request(0)
            request["stream_id"] = response["execution"]["stream_id"]
            request["observation"] = None
            self.assertEqual(session.exchange(request)["type"], "error")
            request["observation"] = self.step_request(0)["observation"]
            response = session.exchange(request)
            self.assertEqual(response["type"], "error")
            self.assertIn("closed", response["detail"])

    def test_hardware_stream_rejects_examples_without_opening_sdk(self):
        request = copy.deepcopy(self.request)
        request["execution"]["mock"] = False
        with self.assertRaisesRegex(PreparationError, "measured configuration"):
            ServerEpisode(request, compile_request(request))

    def test_client_waits_for_new_samples_and_rejects_frozen_feedback(self):
        self.config.safety["feedback_timeout_s"] = .05
        for refresh in (True, False):
            with self.subTest(refresh=refresh), tempfile.TemporaryDirectory() as tmp:
                snapshot = self.rig.states()
                reads, requests = [], []
                def states():
                    reads.append(1)
                    result = copy.deepcopy(snapshot)
                    if refresh and len(reads) >= 4:
                        for state in result.values():
                            state["sequence"] += 1
                    return result
                rig = MockRig(self.config)
                rig.states = states
                def next_step(expected, states, state14):
                    requests.append(states)
                    return expected
                remote = SimpleNamespace(next=next_step)
                executor = ClientExecutor(self.config, rig, Journal(tmp, self.task), self.scene,
                                          mock=True, remote=remote)
                executor._remote_next(self.plan.steps[0])
                if refresh:
                    executor._remote_next(self.plan.steps[1])
                    self.assertEqual(len(requests), 2)
                    self.assertGreater(requests[1]["left"]["sequence"], requests[0]["left"]["sequence"])
                else:
                    with self.assertRaisesRegex(PreparationError, "New controller feedback"):
                        executor._remote_next(self.plan.steps[1])
                    self.assertEqual(len(requests), 1)

    def test_client_cannot_skip_ack_or_change_issued_action(self):
        for mode in ("skip", "change"):
            with self.subTest(mode=mode):
                self.episode = ServerEpisode(self.request, self.response)
                client, requests = self.direct_client()
                states = self.rig.states()
                issued = client.next(self.plan.steps[0], states, self.state14(states))
                with self.assertRaisesRegex(PreparationError, "acknowledge"):
                    if mode == "skip":
                        client.next(self.plan.steps[1], self.rig.states(), self.state14(states))
                    else:
                        client.acknowledge(0, replace(issued, opening=1), {"opening": 1})
                self.assertTrue(client.closed)
                self.assertEqual(len(requests), 1)

    def test_client_rejects_altered_returned_skill_and_never_retries(self):
        def corrupt(response):
            response["skill_actions"][0]["opening"] = 1
        client, requests = self.direct_client(corrupt)
        states = self.rig.states()
        with self.assertRaisesRegex(PreparationError, "differs"):
            client.next(self.plan.steps[0], states, self.state14(states))
        with self.assertRaisesRegex(PreparationError, "closed"):
            client.next(self.plan.steps[0], states, self.state14(states))
        self.assertEqual(len(requests), 1)

    def test_second_connection_cannot_reuse_first_connection_stream(self):
        with ws_server() as url, Session(url) as first, Session(url) as second:
            response = first.exchange(self.request)
            request = self.step_request(0)
            request["stream_id"] = response["execution"]["stream_id"]
            response = second.exchange(request)
            self.assertEqual(response["type"], "error")
            self.assertIn("No active", response["detail"])
            self.assertEqual(first.exchange(request)["type"], "inference")

    def test_server_loss_or_rejection_stops_without_local_fallback(self):
        for failure in ("disconnect", "rejection", "terminal_disconnect"):
            with self.subTest(failure=failure):
                def injected_server(connection):
                    request = loads(connection.recv())
                    response = compile_request(request)
                    episode = ServerEpisode(request, response)
                    response["execution"] = episode.metadata
                    connection.send(dumps(response))
                    for raw in connection:
                        request = loads(raw)
                        if failure == "rejection":
                            connection.send(dumps({"type": "error", "detail": "injected rejection"}))
                            return
                        if failure == "disconnect":
                            connection.close(1011, "injected loss")
                            return
                        response = episode.infer(request)
                        if response["done"]:
                            connection.close(1011, "injected loss of terminal receipt")
                            return
                        connection.send(dumps(response))
                with ws_server(injected_server) as url, tempfile.TemporaryDirectory() as tmp, Session(url) as session:
                    journal, rig = Journal(tmp, self.task), MockRig(self.config)
                    request = self.initial_request(journal.run_id)
                    response = session.exchange(request)
                    plan = accept_plan(response, request, self.config, self.scene)
                    remote = StepClient(session, request, response, journal, mock=True)
                    with self.assertRaises(PreparationError):
                        ClientExecutor(self.config, rig, journal, self.scene, mock=True,
                                       session=session, remote=remote).run(plan)
                    self.assertTrue(rig.stopped)
                    if failure != "terminal_disconnect":
                        self.assertEqual(rig.commands, [])
                    summary = json.loads((journal.path/"summary.json").read_text())
                    self.assertFalse(summary["execution_completed"])
                    self.assertFalse(summary["remote_completion_acknowledged"])
                    self.assertEqual(summary["steps_completed"], len(plan.steps) if failure == "terminal_disconnect" else 0)

    def test_health_and_inference_share_serialized_request_reply_stream(self):
        def echo(connection):
            for raw in connection:
                request = loads(raw)
                if request["type"] == "health":
                    response = {"type": "health", "status": "healthy", "schema": SCHEMA,
                                "action_space": ACTION_SPACE}
                else:
                    time.sleep(.02)
                    response = {"type": "echo", "request_id": request["request_id"]}
                connection.send(dumps(response))
        with ws_server(echo) as url, Session(url) as session:
            barrier, got, errors = threading.Barrier(3), [], []
            def health():
                try:
                    barrier.wait(timeout=2)
                    got.append(("health", session.health(timeout=2)["type"]))
                except Exception as exc:
                    errors.append(exc)
            def infer():
                try:
                    barrier.wait(timeout=2)
                    response = session.exchange({"type": "inference", "request_id": "abc"}, timeout=2)
                    got.append(("infer", response["type"], response["request_id"]))
                except Exception as exc:
                    errors.append(exc)
            threads = [threading.Thread(target=health), threading.Thread(target=infer)]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=2)
            for thread in threads:
                thread.join(timeout=3)
            self.assertEqual(errors, [])
            self.assertCountEqual(got, [("health", "health"), ("infer", "echo", "abc")])


if __name__ == "__main__":
    unittest.main()
