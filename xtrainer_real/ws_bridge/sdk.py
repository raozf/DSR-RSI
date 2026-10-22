"""Read-only imports and RPC client from the unmodified, pinned site SDK."""
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

from dsr_xtrainer.config import KIT_ROOT, kit_path, read_json
from dsr_xtrainer.hardware import StrictTransport
from dsr_xtrainer.model import PreparationError


def verify_sdk(root):
    root = kit_path(root).resolve()
    lock = read_json(KIT_ROOT/"ws-sdk.lock.json")
    try:
        rev = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                      text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreparationError("XTrainer SDK missing; run scripts/install_ws.sh") from exc
    if rev != lock["commit"]:
        raise PreparationError("XTrainer SDK revision differs from ws-sdk.lock.json")
    for relative, expected in lock["files_sha256"].items():
        path = root/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise PreparationError("Pinned SDK source missing/modified: "+relative)
    return root


def load_sdk(root):
    root = verify_sdk(root)
    sys.path.insert(0, str(root))
    try:
        api = importlib.import_module("dobot_control.robots.dobot_api")
        node = importlib.import_module("dobot_control.robots.robot_node")
        util = importlib.import_module("scripts.function_util")
    except ImportError as exc:
        raise PreparationError("Install kit requirements (and python3-tk if absent): "+str(exc)) from exc
    for module, relative in [(api, "dobot_control/robots/dobot_api.py"),
                             (node, "dobot_control/robots/robot_node.py"),
                             (util, "scripts/function_util.py")]:
        if Path(module.__file__).resolve() != (root/relative).resolve():
            raise PreparationError("A different SDK was imported; launch in a fresh process")
    if api.MyType.itemsize != 1440:
        raise PreparationError("Unsupported feedback packet layout")

    class ReadOnlyDashboard(StrictTransport, api.DobotApiDashboard):
        def sendRecvMsg(self, command):
            if command.partition("(")[0] not in {
                    "GetAngle", "GetPose", "GetErrorID", "RobotMode",
                    "InverseSolution", "PositiveSolution"}:
                raise PreparationError("Adapter dashboard is read-only: "+command)
            self.last_command = command
            return super().sendRecvMsg(command)

    # Never instantiate DobotRobot or launch_nodes: they enable and configure hardware.
    return ReadOnlyDashboard, node.ZMQClientRobot, api.MyType
