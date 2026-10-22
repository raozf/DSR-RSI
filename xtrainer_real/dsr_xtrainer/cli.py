import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import ipaddress
import json
from pathlib import Path
import signal
import sys
import tempfile
from .calibration import fit_rigid
from .config import Config, KIT_ROOT, kit_path, read_json
from .executor import Executor, Journal, MockRig, utc_now, write_json
from .hardware import Gripper, RealRig, emergency_stop, load_sdk
from .model import PreparationError
from .perception import load_scene
from .tasks import build_plan


@contextmanager
def hardware_lock(config):
    # Per-IP locks in /tmp also serialize independent copies of this kit.
    # Emergency stop intentionally does not acquire these locks.
    streams = []
    try:
        for ip in sorted({a["ip"] for a in config.arms.values()}):
            digest = hashlib.sha256(ip.encode()).hexdigest()[:20]
            stream = open(Path(tempfile.gettempdir())/("dsr-xtrainer-"+digest+".lock"), "a")
            streams.append(stream)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PreparationError("Another XTrainer process holds the hardware lock for "+ip) from exc
        yield
    finally:
        for stream in streams:
            stream.close()


def parser():
    p = argparse.ArgumentParser(description="DSR XTrainer standalone preparation and execution")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Copy templates to ignored local/ (never overwrite)")
    init.add_argument("--directory", default=str(KIT_ROOT/"local"))
    for name in ["plan", "run", "preflight"]:
        q = sub.add_parser(name)
        q.add_argument("--config", help="Site configuration; run/plan default to offline examples, --execute to local/site.json")
        q.add_argument("--task", choices=["align_blocks", "cover_blocks"])
        q.add_argument("--scene")
        q.add_argument("--output-root", default=str(KIT_ROOT/"runs"))
        if name == "run":
            modes = q.add_mutually_exclusive_group()
            modes.add_argument("--execute", action="store_true", help="Enable robots and execute the locally calibrated task")
            modes.add_argument("--mock", action="store_true", help="Exercise executor without hardware")
            q.add_argument("--checkpoint-dir")
        if name == "preflight":
            q.add_argument("--connect", action="store_true", help="Read-only dashboard and feedback queries")
            q.add_argument("--check-grippers", action="store_true", help="Read-only serial ping/position check (requires --connect)")
            q.add_argument("--check-camera", action="store_true", help="Capture configured camera; no robot connection required")
    stop = sub.add_parser("stop", help="Send EmergencyStop() to both IPs using independent sockets")
    stop.add_argument("--config", default=str(KIT_ROOT/"local/site.json"))
    stop.add_argument("--output-root", default=str(KIT_ROOT/"runs"))
    cap = sub.add_parser("capture")
    cap.add_argument("--config", default=str(KIT_ROOT/"local/site.json"))
    cap.add_argument("--output-root", default=str(KIT_ROOT/"runs"))
    fit = sub.add_parser("fit-frame")
    fit.add_argument("--points", required=True)
    fit.add_argument("--output", required=True)
    check = sub.add_parser("checkpoint", help="Bind a measured observation to the current request")
    check.add_argument("--request", required=True)
    check.add_argument("--observation", required=True)
    return p


def init_local(directory):
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    for source, name in [("configs/site.template.json", "site.json"), ("configs/calibration.template.json", "calibration.json"),
                         ("examples/align_blocks.scene.json", "align_blocks.scene.json"), ("examples/cover_blocks.scene.json", "cover_blocks.scene.json")]:
        destination = target/name
        data = read_json(KIT_ROOT/source)
        if name == "site.json":
            data["calibration_file"] = str(target/"calibration.json")
        if destination.exists():
            print("Keeping existing "+str(destination))
            continue
        write_json(destination, data)
        print("Created "+str(destination))
    print("Fill measured values, replace scene geometry and timestamps, then clear example_only in site/calibration/scene.")


def stop_command(args):
    # No calibration/SDK/camera dependency: stopping must work with only actual IPs.
    data = read_json(args.config)
    if data.get("example_only") is not False or set(data.get("arms", {})) != {"left", "right"}:
        raise PreparationError("Emergency stop needs local/site.json with actual left/right IPs, example_only=false")
    for arm in data["arms"].values():
        try:
            ip = ipaddress.ip_address(arm.get("ip", ""))
        except ValueError as exc:
            raise PreparationError("Fill both actual robot IPs before using emergency_stop.sh") from exc
        if ip.is_unspecified or ip.is_multicast:
            raise PreparationError("Invalid emergency stop IP")
    journal = Journal(args.output_root, "emergency_stop")
    result = emergency_stop(data)
    write_json(journal.path/"summary.json", result)
    print(json.dumps(result, indent=2))
    return 0 if all(r["acknowledged"] for r in result.values()) else 2


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            init_local(args.directory)
            return 0
        if args.command == "fit-frame":
            p = read_json(args.points)
            out = Path(args.output)
            if out.exists():
                raise PreparationError("Output already exists: "+str(out))
            write_json(out, fit_rigid(p["base_points_mm"], p["world_points_mm"]))
            print(str(out.resolve()))
            return 0
        if args.command == "checkpoint":
            request = read_json(args.request)
            obs = read_json(args.observation)
            if request.get("schema") != "dsr.xtrainer-checkpoint-request/v1":
                raise PreparationError("Invalid checkpoint request")
            if obs.get("source") not in {"manual", "vision"} or obs.get("simulated") is not False or obs.get("approved") is not True:
                raise PreparationError("Observation requires source=manual/vision, simulated=false, approved=true and measured predicate data")
            from .perception import age_seconds
            age_seconds(obs.get("captured_at"))  # timestamp is acquisition time; never silently refresh it
            record = dict(obs, schema="dsr.xtrainer-checkpoint/v1", **{k: request[k] for k in ["run_id", "stage", "calibration_id"]})
            path = Path(args.request)
            if not path.name.endswith(".request.json"):
                raise PreparationError("Request filename must end with .request.json")
            out = path.with_name(path.name.removesuffix(".request.json")+".json")
            if out.exists():
                raise PreparationError("Checkpoint response already exists")
            temp = out.with_suffix(".tmp")
            write_json(temp, record)
            temp.replace(out)
            print(str(out))
            return 0
        if args.command == "stop":
            return stop_command(args)
        execute = getattr(args, "execute", False)
        if args.config is None:
            args.config = str(KIT_ROOT/("examples/site.demo.json" if args.command in {"plan", "run"} and not execute else "local/site.json"))
        if args.command in {"plan", "run"} and args.task and not args.scene:
            scene_folder = "examples" if Path(args.config).resolve() == (KIT_ROOT/"examples/site.demo.json").resolve() else "local"
            args.scene = str(KIT_ROOT/scene_folder/(args.task+".scene.json"))
        config = Config(args.config)
        if args.command == "capture":
            from .camera import capture
            journal = Journal(args.output_root, "camera")
            capture(config, journal.path)
            print(str(journal.path))
            return 0
        if bool(args.task) != bool(args.scene):
            raise PreparationError("Pass --task and --scene together")
        if args.command != "preflight" and not args.task:
            raise PreparationError("--task and --scene are required")
        scene, plan = None, None
        if args.scene:
            scene = load_scene(args.scene, config, args.task, hardware=execute)
            plan = build_plan(config, scene, args.task)
        journal = Journal(args.output_root, args.task or "preflight")
        write_json(journal.path/"config.snapshot.json", config.data)
        write_json(journal.path/"calibration.snapshot.json", config.calibration)
        if scene:
            write_json(journal.path/"scene.snapshot.json", scene)
        if plan:
            write_json(journal.path/"plan.json", plan.to_dict())
            print(f"{plan.task}: {len(plan.steps)} steps; {sum(s.kind == 'checkpoint' for s in plan.steps)} observation checkpoints")
        print("Run directory: "+str(journal.path), flush=True)
        if args.command == "preflight":
            report = {"configuration_valid": True, "task_plan_valid": plan is not None, "hardware_connected": False,
                      "site_configuration_complete": False, "motion_sent": False}
            try:
                config.require_hardware(execute=True)
                report["site_configuration_complete"] = True
            except PreparationError as exc:
                report["site_configuration_issue"] = str(exc)
            if args.check_grippers and not args.connect:
                raise PreparationError("--check-grippers requires --connect")
            if args.connect:
                config.require_hardware()
                with hardware_lock(config):
                    rig = RealRig(config)
                    try:
                        rig.connect()
                        report.update(hardware_connected=True, states=rig.states(), queries=rig.queries())
                        if args.check_grippers:
                            report["grippers"] = {}
                            for side, arm in config.arms.items():
                                g = Gripper(arm["gripper"], rig.sdk[3], execute=False)
                                try:
                                    report["grippers"][side] = {"position_ticks": g.position(), "registers_written": False}
                                finally:
                                    g.close()
                    finally:
                        rig.close()
            if args.check_camera:
                from .camera import capture
                report["camera"] = capture(config, journal.path)
            write_json(journal.path/"preflight.json", report)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "plan" or not (execute or args.mock):
            print("Plan only. No hardware connection or motion. Use --mock for offline execution; --execute for configured hardware.")
            return 0
        if args.mock:
            summary = Executor(config, MockRig(config), journal, scene, mock=True).run(plan)
        else:
            config.require_hardware(execute=True)
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f"Received signal {signum}")
            previous = signal.signal(signal.SIGTERM, interrupted)
            try:
                with hardware_lock(config):
                    summary = Executor(config, RealRig(config, execute=True), journal, scene,
                                       checkpoint_dir=args.checkpoint_dir).run(plan)
            finally:
                signal.signal(signal.SIGTERM, previous)
        print(json.dumps(summary, indent=2))
        return 0
    except (PreparationError, KeyError, ValueError, OSError) as exc:
        print("XTrainer: "+str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; check emergency_stop records and the physical stop button.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
