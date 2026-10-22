import ipaddress
import json
from pathlib import Path
import numpy as np
from .geometry import matrix, vector
from .model import PreparationError

KIT_ROOT = Path(__file__).resolve().parents[1]
SDK_REVISION = "04cbec5f57aee6355b70c3933d84962376029f1d"


def read_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    except (OSError, ValueError) as exc:
        raise PreparationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PreparationError(f"{path}: top level must be an object")
    return data


def kit_path(value):
    if not isinstance(value, str) or not value:
        raise PreparationError("A required file path is empty; fill the local configuration")
    p = Path(value).expanduser()
    return p if p.is_absolute() else KIT_ROOT / p


def bounded(value, low, high, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
        raise PreparationError(f"{name}: expected {'integer' if integer else 'number'} in [{low}, {high}]")
    if integer and int(value) != value:
        raise PreparationError(f"{name}: must be an integer")
    return value


class Config:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.data = read_json(self.path)
        if self.data.get("schema") != "dsr.xtrainer-config/v1":
            raise PreparationError("Unsupported configuration schema")
        self.calibration_path = kit_path(self.data.get("calibration_file"))
        self.calibration = read_json(self.calibration_path)
        if self.calibration.get("schema") != "dsr.xtrainer-calibration/v1":
            raise PreparationError("Unsupported calibration schema")
        if not self.calibration.get("calibration_id"):
            raise PreparationError("Fill calibration_id and measured calibration values")
        if self.calibration.get("rotation_convention") != "RzRyRx_degrees":
            raise PreparationError("rotation_convention must be RzRyRx_degrees")
        self.arms = self.data.get("arms", {})
        self.frames = self.calibration.get("arms", {})
        if set(self.arms) != {"left", "right"} or set(self.frames) != {"left", "right"}:
            raise PreparationError("Both left and right arms must be configured")
        self.safety = self.data.get("safety", {})
        for key, low, high in [("speed_percent", 1, 20), ("acceleration_percent", 1, 20),
                               ("max_step_mm", 0.1, 25), ("max_rotation_step_deg", 0.1, 10),
                               ("tcp_padding_mm", 0, 200), ("min_tcp_separation_mm", 50, 1000),
                               ("position_tolerance_mm", 0.1, 5), ("rotation_tolerance_deg", 0.1, 5),
                               ("park_joint_tolerance_deg", 0.1, 3),
                               ("park_drift_tolerance_mm", 0.1, 5), ("motion_timeout_s", 1, 120),
                               ("connect_timeout_s", 0.1, 5), ("reply_timeout_s", 0.1, 5),
                               ("feedback_timeout_s", 0.1, 2), ("scene_max_age_s", 1, 600)]:
            bounded(self.safety.get(key), low, high, "safety."+key, key in {"speed_percent", "acceleration_percent"})
        bounded(self.safety.get("checkpoint_timeout_s"), 10, 3600, "safety.checkpoint_timeout_s")
        if self.safety["max_step_mm"] <= 2*self.safety["position_tolerance_mm"]:
            raise PreparationError("max_step_mm must exceed twice the position tolerance")
        self.safe_z = bounded(self.data.get("tasks", {}).get("safe_transit_z_mm"), 20, 2000, "tasks.safe_transit_z_mm")
        for side in self.arms:
            f, arm = self.frames[side], self.arms[side]
            matrix(f.get("world_T_base"), side+".world_T_base")
            park = vector(f.get("park_tcp_world_mm_deg"), 6, side+".park_tcp_world_mm_deg")
            if park[2] < self.safe_z:
                raise PreparationError(side+": park TCP must be at or above safe_transit_z_mm")
            bounds = f.get("workspace_world_mm", {})
            lo = vector(bounds.get("min"), 3, side+".workspace.min")
            hi = vector(bounds.get("max"), 3, side+".workspace.max")
            if np.any(hi <= lo):
                raise PreparationError(side+": workspace min must be below max")
            for key in ["joint_min_deg", "joint_max_deg"]:
                vector(f.get(key), 6, side+"."+key)
            vector(f.get("park_joints_deg"), 6, side+".park_joints_deg")
            if np.any(np.asarray(f["joint_min_deg"]) >= f["joint_max_deg"]):
                raise PreparationError(side+": invalid joint bounds")
            if np.any(np.asarray(f["park_joints_deg"]) < f["joint_min_deg"]) or np.any(np.asarray(f["park_joints_deg"]) > f["joint_max_deg"]):
                raise PreparationError(side+": park joints exceed configured joint bounds")
            bounded(arm.get("tool_index"), 0, 9, side+".tool_index", True)
            if arm.get("user_index") != 0:
                raise PreparationError(side+": user_index must be 0 (base); transform handles the common world frame")
            for key in ["dashboard_port", "move_port", "feedback_port"]:
                bounded(arm.get(key), 1, 65535, side+"."+key, True)
            g = arm.get("gripper", {})
            bounded(g.get("id"), 1, 253, side+".gripper.id", True)
            bounded(g.get("baudrate"), 9600, 1000000, side+".gripper.baudrate", True)
            bounded(g.get("speed"), 1, 1000, side+".gripper.speed", True)
            bounded(g.get("acceleration"), 1, 100, side+".gripper.acceleration", True)
            bounded(g.get("settle_s"), 0.1, 5, side+".gripper.settle_s")
            bounded(g.get("position_tolerance_ticks"), 1, 500, side+".gripper.position_tolerance_ticks", True)
            bounded(g.get("torque_limit"), 1, 300, side+".gripper.torque_limit", True)
            for k in ["open_ticks", "closed_ticks"]:
                bounded(g.get(k), 0, 4095, side+".gripper."+k, True)
            if g["open_ticks"] == g["closed_ticks"]:
                raise PreparationError(side+": gripper open and closed calibration must differ")
        for box in self.safety.get("forbidden_boxes_world_mm", []):
            if np.any(vector(box.get("min"), 3, "forbidden box.min") >= vector(box.get("max"), 3, "forbidden box.max")):
                raise PreparationError("Invalid forbidden box")

    def park(self, side):
        return vector(self.frames[side]["park_tcp_world_mm_deg"], 6, side+".park")

    def require_hardware(self, execute=False):
        if self.data.get("example_only") is not False or self.calibration.get("example_only") is not False:
            raise PreparationError("Demo/template configuration cannot connect to hardware. Fill local/ site calibration first.")
        for side, arm in self.arms.items():
            try:
                ip = ipaddress.ip_address(arm.get("ip", ""))
            except ValueError as exc:
                raise PreparationError(side+": fill the actual robot IP") from exc
            if ip.is_unspecified or ip.is_multicast:
                raise PreparationError(side+": invalid robot IP")
        if self.arms["left"]["ip"] == self.arms["right"]["ip"]:
            raise PreparationError("Left and right robot IPs must differ")
        if execute:
            for key in ["tcp_and_frames_measured", "workspace_and_paths_reviewed", "grippers_calibrated"]:
                if self.calibration.get(key) is not True:
                    raise PreparationError(f"calibration.{key} must reflect completed on-site calibration")
            ports = []
            for side, arm in self.arms.items():
                vector(self.frames[side].get("tool_tcp_controller_mm_deg"), 6, side+".tool_tcp_controller_mm_deg")
                port = arm["gripper"].get("port")
                if not isinstance(port, str) or not port.startswith("/dev/"):
                    raise PreparationError(side+": configure the actual gripper /dev/serial/by-id/... port")
                ports.append(str(Path(port).resolve()))
            if len(set(ports)) != 2:
                raise PreparationError("Left and right gripper serial devices must differ")
