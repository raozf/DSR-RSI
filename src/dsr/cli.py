"""DSR command-line entry point. Heavy simulators load only on explicit commands."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from dsr.paths import repository_root, bootstrap


def _load_tool(name):
    path = repository_root() / "tools" / (name + ".py")
    if not path.is_file():
        raise FileNotFoundError(path)
    tools_dir = str(path.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    spec = importlib.util.spec_from_file_location("dsr_tool_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doctor(args):
    from dsr.runtime_config import checkpoint_root, code_asset_root, interpreter
    root = repository_root()
    benchmark = root / "metadata" / "robodojo_inventory.json"
    checkpoint = checkpoint_root() / "DM05-MEM-Robodojo-Sim"
    code_root = code_asset_root(required=False) / "implementations" / "code-skills"
    report = {
        "schema": "dsr.doctor/v1",
        "root": str(root),
        "python": sys.version.split()[0],
        "benchmark_inventory": {k: v for k, v in json.loads(benchmark.read_text()).items() if k != "tasks"} if benchmark.is_file() else None,
        "policy_python": str(interpreter("policy", required=False)),
        "sim_python": str(interpreter("sim", required=False)),
        "code_python": str(interpreter("code", required=False)),
        "checkpoint_configured": (checkpoint / "config.json").is_file(),
        "norm_stats_configured": (checkpoint / "norm_stats.json").is_file(),
        "code_program_manifest": (code_root / "programs.json").is_file(),
        "robodojo_assets_configured": Path(os.environ.get("DSR_ROBODOJO_ASSETS", root / "assets" / "RoboDojo" / "Assets")).is_dir(),
    }
    try:
        manifest = json.loads((code_root / "programs.json").read_text())
        report["registered_code_programs"] = len(manifest["programs"])
    except (OSError, ValueError, KeyError):
        report["registered_code_programs"] = 0
    report["embodied_execution_ready"] = all([
        report["checkpoint_configured"], report["norm_stats_configured"],
        report["robodojo_assets_configured"],
        Path(report["policy_python"]).is_file(), Path(report["sim_python"]).is_file(),
    ])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["embodied_execution_ready"] or not args.strict else 2


def _evaluate(args, forwarded):
    forwarded = list(forwarded)
    reserved = {"--policy-name", "--agent-strategy", "--agent-mode", "--episode-retry"}
    for token in forwarded:
        option = token.split("=", 1)[0]
        if option.startswith("--") and any(name.startswith(option) for name in reserved):
            raise ValueError(f"{option} is controlled by --policy and --recovery; cannot override frozen evaluation")
    if not any(x.startswith("--policy-gpus") for x in forwarded) or not any(x.startswith("--env-gpus") for x in forwarded):
        raise ValueError("Declare --policy-gpus and --env-gpus explicitly after inspecting shared workloads")
    if "--full-protocol" in forwarded:
        forwarded.remove("--full-protocol")
    elif not any(x.startswith("--episodes") for x in forwarded):
        raise ValueError("Declare a bounded --episodes count, or --full-protocol for the official 6300-episode protocol")
    if args.recovery != "off" and args.policy != "integrated":
        raise ValueError("Failure restart requires the integrated DM0.5 + PhysicalRSI policy")
    if args.policy == "integrated" and not args.codex_home:
        raise ValueError("Set --codex-home or DSR_CODEX_HOME for a real multimodal Agent")
    # ``preflight`` validates the same process boundary used by the launcher.
    # Keep an explicit CLI value authoritative even when the parent shell did
    # not export DSR_CODEX_HOME; the downstream worker receives this value too.
    if args.codex_home:
        os.environ["DSR_CODEX_HOME"] = str(Path(args.codex_home).expanduser().resolve())
    from dsr.runtime_config import preflight
    preflight(agent=args.policy == "integrated")
    call = ["--policy-name", "DSRRSI" if args.policy == "integrated" else "OpenDM"]
    if args.policy == "integrated":
        call += ["--agent-strategy", "physicalrsi", "--agent-mode", "frozen", "--codex-home", args.codex_home]
    if args.recovery == "single-restart":
        call.append("--episode-retry")
    if os.environ.get("DSR_RUN_TIMEOUT") and not any(x.split("=", 1)[0] == "--run-timeout" for x in forwarded):
        call += ["--run-timeout", os.environ["DSR_RUN_TIMEOUT"]]
    call += forwarded
    return _load_tool("full_robodojo_agent_eval").main(call)


def _physicalrsi(forwarded):
    bootstrap()
    from PhysicalRSI.cli import main as upstream_main
    return upstream_main(forwarded)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dsr", description="DSR embodied execution and research")
    parser.add_argument("--config", type=Path, help="Versioned DSR runtime configuration JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Report independently configured runtime assets")
    doctor.add_argument("--strict", action="store_true")
    evaluation = sub.add_parser("evaluate", help="Run the official RoboDojo launcher")
    evaluation.add_argument("--policy", choices=["dm05", "integrated"], default="integrated")
    evaluation.add_argument("--recovery", choices=["off", "single-restart"], default="off")
    evaluation.add_argument("--codex-home", default=None)
    sub.add_parser("physicalrsi", help="Run the vendored PhysicalRSI CLI")
    sub.add_parser("evolve", help="Run a bounded DSR development experiment")
    sub.add_parser("dashboard", help="Serve local evidence browser")
    args, forwarded = parser.parse_known_args(argv)
    if args.config:
        from dsr.configuration import apply_config
        apply_config(args.config)
    if args.command == "evaluate" and args.codex_home is None:
        args.codex_home = os.environ.get("DSR_CODEX_HOME")
    if args.command == "doctor":
        if forwarded:
            parser.error("Unrecognized doctor arguments: " + " ".join(forwarded))
        return _doctor(args)
    if args.command == "evaluate":
        return _evaluate(args, forwarded)
    if args.command == "physicalrsi":
        return _physicalrsi(forwarded)
    if args.command == "evolve":
        bootstrap()
        from dsr.research.experiment import main as evolve_main
        return evolve_main(forwarded)
    if args.command == "dashboard":
        from dsr.visualization.server import main as dashboard_main
        return dashboard_main(forwarded)
    return 2
