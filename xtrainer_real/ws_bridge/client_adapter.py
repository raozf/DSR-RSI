"""DSR client: WebSocket feedback inference and robot-node execution.

The server never receives robot addresses or SDK handles.  The client owns the
calibrated scene contract and, after validating the exact returned plan, can
run it through the existing upstream ZMQ ``robot_node``.
"""
import argparse
import base64
import json
from pathlib import Path
import signal
import sys
import threading
import time

from websockets.sync.client import connect

from dsr_xtrainer.cli import hardware_lock
from dsr_xtrainer.config import Config, KIT_ROOT, read_json
from dsr_xtrainer.executor import Executor, Journal, MockRig, write_json
from dsr_xtrainer.model import PreparationError
from dsr_xtrainer.perception import load_scene
from dsr_xtrainer.tasks import build_plan
from .protocol import accept_plan, dumps, loads, make_request
from .robot_client import ExistingClientRig, Transport, action14
from .stream import STREAM_MODE, StepClient


def default_scene(config_path, task):
    config_path = Path(config_path).resolve()
    return KIT_ROOT/("examples" if config_path == (KIT_ROOT/"examples/site.demo.json").resolve() else "local")/(task+".scene.json")


class Session:
    """Serialized preview, feedback inference and health; no reconnect/replay."""
    def __init__(self, url, timeout=30):
        self.url, self.timeout = url, timeout
        self.ws = None
        self._done = threading.Event()
        self._fault = None
        self._last_alive = time.monotonic()
        self._thread = None
        self._io_lock = threading.Lock()

    def __enter__(self):
        self.ws = connect(self.url, open_timeout=5, close_timeout=2, ping_interval=10,
                          ping_timeout=5, max_size=16*1024*1024, compression=None, proxy=None)
        return self

    def exchange(self, payload, timeout=None):
        try:
            # Health and inference share one request/reply stream. Never let
            # the watchdog consume a skill response or overlap recv calls.
            with self._io_lock:
                self.ws.send(dumps(payload))
                response = loads(self.ws.recv(timeout=self.timeout if timeout is None else timeout))
                self._last_alive = time.monotonic()
                return response
        except Exception as exc:
            raise PreparationError("WebSocket failed; request will not be replayed: "+str(exc)) from exc

    def health(self, timeout=None):
        from .protocol import ACTION_SPACE, SCHEMA
        response = self.exchange({"type": "health"}, timeout)
        if (response.get("type") != "health" or response.get("status") != "healthy" or
                response.get("schema") != SCHEMA or response.get("action_space") != ACTION_SPACE):
            raise PreparationError("Remote endpoint is not the matching DSR server")
        self._last_alive = time.monotonic()
        return response

    def watch(self):
        self.health(timeout=2)
        def run():
            while not self._done.wait(1):
                try:
                    self.health(timeout=2)
                except Exception as exc:
                    self._fault = str(exc)
                    return
        self._thread = threading.Thread(target=run, name="dsr-ws-watchdog", daemon=True)
        self._thread.start()

    def ensure_alive(self):
        if self._fault or time.monotonic()-self._last_alive > 4:
            raise PreparationError("Remote WebSocket watchdog: "+(self._fault or "health expired"))

    def __exit__(self, *exc):
        self._done.set()
        if self.ws:
            self.ws.close()
        if self._thread:
            self._thread.join(timeout=4)


def check_server(url, timeout=5):
    with Session(url, timeout) as session:
        return session.health()


class ClientExecutor(Executor):
    """Use the adapter's stop gate; never race a later node command with an old stop."""
    def __init__(self, *args, session=None, remote=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = session
        self.remote = remote
        if remote is not None:
            self.remote_transport = self.rig.transport if not self.mock else Transport(KIT_ROOT/"configs/transport.template.json")
            self.remote_openings = {side: self.remote_transport.data["grippers"][side]["initial_opening"]
                                    for side in ("left", "right")}
            self.remote_feedback_sequences = {side: -1 for side in ("left", "right")}

    def heartbeat(self, parked_side=None):
        if self.session:
            self.session.ensure_alive()
        if self._stopped:
            raise PreparationError("Executor already stopped")
        return super().heartbeat(parked_side)

    def run(self, plan):
        try:
            summary = super().run(plan)
            if self.remote:
                summary.update(execution_mode=STREAM_MODE, remote_policy_calls=self.remote.sequence,
                               remote_completion_acknowledged=self.remote.completed)
            return summary
        except Exception as exc:
            # A monitor can stop the rig while a move is in progress. Preserve
            # that original fault when the move then reports "already stopped".
            # super().run has joined the monitor and written its final receipt.
            with self._fault_lock:
                fault = self._fault
            if fault and fault not in str(exc):
                path = self.journal.path/"summary.json"
                summary = read_json(path)
                summary["execution_error"] = summary["error"]
                summary["error"] = fault
                write_json(path, summary)
                self.journal.event("abort_cause", error=fault, execution_error=str(exc))
                raise PreparationError(fault) from exc
            raise
        finally:
            path = self.journal.path/"summary.json"
            if self.remote and path.is_file():
                summary = read_json(path)
                summary.update(execution_mode=STREAM_MODE, remote_policy_calls=self.remote.sequence,
                               remote_completion_acknowledged=self.remote.completed)
                write_json(path, summary)

    def _remote_next(self, expected):
        states = self.heartbeat()
        deadline = time.monotonic()+self.config.safety["feedback_timeout_s"]
        while any(states[side]["sequence"] <= last for side, last in self.remote_feedback_sequences.items()):
            if time.monotonic() >= deadline:
                raise PreparationError("New controller feedback is required before remote inference")
            self._stop_monitor.wait(.01)
            states = self.heartbeat()
        self.remote_feedback_sequences = {side: state["sequence"] for side, state in states.items()}
        state14 = action14(states, self.remote_openings, self.remote_transport).tolist()
        step = self.remote.next(expected, states, state14)
        self.heartbeat()  # A background fault during inference still gates execution.
        return step

    def before_step(self, index, step):
        return self._remote_next(step) if self.remote else step

    def after_step(self, index, step, actual):
        if not self.remote:
            return
        record = None
        if step.kind == "gripper":
            self.remote_openings[step.arm] = step.opening
        elif step.kind == "checkpoint":
            record = read_json(self.journal.path/"checkpoints"/(step.stage+".accepted.json"))
        self.remote.acknowledge(index, step, actual, record)

    def finish_plan(self):
        if self.remote:
            self._remote_next(None)

    def stop(self, reason):
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            result = self.rig.stop()
            self.journal.event("emergency_stop", reason=reason, arms=result)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="ws://127.0.0.1:18000/ws")
    p.add_argument("--config", help="site config; offline default is examples/site.demo.json")
    p.add_argument("--transport", default=str(KIT_ROOT/"local/transport.json"))
    p.add_argument("--task", choices=["align_blocks", "cover_blocks"])
    p.add_argument("--scene")
    p.add_argument("--output-root", default=str(KIT_ROOT/"runs"))
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--check-only", action="store_true", help="Only check remote WebSocket health")
    p.add_argument("--init", action="store_true", help="Create local templates without overwriting existing files")
    p.add_argument("--preflight", action="store_true", help="Read-only existing-node and direct-telemetry check")
    p.add_argument("--checkpoint-dir")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--mock", action="store_true", help="Validate and execute with MockRig")
    modes.add_argument("--execute", action="store_true", help="Execute through existing local robot_node")
    p.add_argument("--images", nargs=3, metavar=("TOP", "LEFT", "RIGHT"), help="Optional image files; archived inputs, no detection")
    return p


def init_local():
    target = KIT_ROOT/"local"
    target.mkdir(exist_ok=True)
    pairs = [("configs/site.ws.template.json", "site.json"),
             ("configs/calibration.template.json", "calibration.json"),
             ("configs/transport.template.json", "transport.json"),
             ("examples/align_blocks.scene.json", "align_blocks.scene.json"),
             ("examples/cover_blocks.scene.json", "cover_blocks.scene.json")]
    for relative, name in pairs:
        path = target/name
        if path.exists():
            print("Keeping existing "+str(path))
            continue
        with path.open("x") as out:
            out.write(json.dumps(read_json(KIT_ROOT/relative), indent=2)+"\n")
        print("Created "+str(path))


def run_task(args, config, scene, session):
    images = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in (args.images or [])]
    journal = Journal(args.output_root, args.task)
    execution = ({"mode": STREAM_MODE, "run_id": journal.run_id, "mock": args.mock}
                 if args.mock or args.execute else None)
    request = make_request(config, scene, args.task, images=images, execution=execution)
    response = session.exchange(request)
    if response.get("type") == "error":
        raise PreparationError("Remote planner rejected request: "+str(response.get("detail", "unknown error")))
    plan = accept_plan(response, request, config, scene)
    for name, data in [("config.snapshot", config.data), ("calibration.snapshot", config.calibration),
                       ("scene.snapshot", scene), ("request", request), ("response", response), ("plan", plan.to_dict())]:
        write_json(journal.path/(name+".json"), data)
    print("Validated remote DSR plan: "+str(journal.path), flush=True)
    if not (args.mock or args.execute):
        print("Plan only. No robot-node connection or motion.")
        return
    session.watch()
    remote = StepClient(session, request, response, journal, mock=args.mock)
    if args.mock:
        summary = ClientExecutor(config, MockRig(config), journal, scene, mock=True, session=session, remote=remote).run(plan)
    else:
        transport = Transport(args.transport)
        transport.require_hardware(config, execute=True)
        # Recheck scene freshness after the network exchange and before opening hardware.
        scene_now = load_scene(args.scene, config, args.task, hardware=True)
        if dumps(scene_now) != dumps(scene):
            raise PreparationError("Scene changed during planning; acquire and plan again")
        write_json(journal.path/"transport.snapshot.json", transport.data)
        with hardware_lock(config):
            rig = ExistingClientRig(config, transport, execute=True)
            summary = ClientExecutor(config, rig, journal, scene, session=session,
                                     checkpoint_dir=args.checkpoint_dir, remote=remote).run(plan)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main(argv=None):
    args = parser().parse_args(argv)
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    try:
        if args.timeout <= 0 or args.timeout > 600:
            raise PreparationError("--timeout must be in (0, 600] seconds")
        if args.init:
            init_local()
            return 0
        if args.check_only:
            print(json.dumps(check_server(args.url, args.timeout), indent=2, ensure_ascii=False))
            return 0
        if (args.preflight and (args.execute or args.mock)):
            raise PreparationError("--preflight cannot be combined with --execute/--mock")
        if not args.task and not args.preflight:
            raise PreparationError("--task is required unless --check-only is used")
        if args.config is None:
            args.config = str(KIT_ROOT/("local/site.json" if args.execute or args.preflight else "examples/site.demo.json"))
        config = Config(args.config)
        if args.preflight:
            transport = Transport(args.transport)
            transport.require_hardware(config, execute=False)
            with hardware_lock(config):
                rig = ExistingClientRig(config, transport, execute=False)
                try:
                    rig.connect()
                    report = {"hardware_connected": True, "motion_sent": False,
                              "states": rig.states(), "queries": rig.queries(),
                              "mapping": rig.verify_node_mapping()}
                    journal = Journal(args.output_root, "ws-preflight")
                    write_json(journal.path/"preflight.json", report)
                    print(json.dumps(report, indent=2))
                finally:
                    rig.close()
            return 0
        if args.scene is None:
            args.scene = str(default_scene(args.config, args.task))
        scene = load_scene(args.scene, config, args.task, hardware=args.execute)
        build_plan(config, scene, args.task)  # Reject malformed geometry before the network.
        if args.execute:
            Transport(args.transport).require_hardware(config, execute=True)
        with Session(args.url, args.timeout) as session:
            session.health()
            run_task(args, config, scene, session)
        return 0
    except Exception as exc:
        print("XTrainer WebSocket client: "+str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; verify the physical stop button and emergency-stop receipt.", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
