"""JSON /ws envelope, with explicit Cartesian skill plans instead of fake joints.

The upstream Motus health envelope is preserved. DSR responses deliberately use
action_space=tcp_skill_plan: TCP millimetres/degrees must never be interpreted as
the upstream predicted_actions [T,14] radians. Conversion belongs to the client.
"""
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

from dsr_xtrainer.config import Config
from dsr_xtrainer.model import Plan, PreparationError, Step
from dsr_xtrainer.perception import age_seconds, load_scene
from dsr_xtrainer.safety import validate_plan
from dsr_xtrainer.tasks import build_plan

SCHEMA = "dsr.xtrainer-ws/v1"
ACTION_SPACE = "tcp_skill_plan_world_mm_deg"
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
TASKS = ("align_blocks", "cover_blocks")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def loads(value):
    if not isinstance(value, str) or len(value.encode()) > MAX_MESSAGE_BYTES:
        raise PreparationError("Expected a JSON text frame within the message size limit")
    result = json.loads(value, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    if not isinstance(result, dict):
        raise PreparationError("WebSocket message must be a JSON object")
    return result


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def planning_projection(config):
    """Send only planning inputs. Site addresses, serial devices and SDK stay local."""
    return {
        "schema": config.data["schema"], "example_only": config.data["example_only"],
        "arms": {side: {k: arm[k] for k in ("user_index", "tool_index")}
                 for side, arm in config.arms.items()},
        "safety": config.safety, "tasks": config.data["tasks"],
        "calibration": config.calibration,
    }


def projected_config(projection, directory):
    """Reuse planning validation; inert legacy I/O fields are never connected."""
    if set(projection) != {"schema", "example_only", "arms", "safety", "tasks", "calibration"}:
        raise PreparationError("Invalid planning configuration projection")
    data = {k: projection[k] for k in ("schema", "example_only", "safety", "tasks")}
    data["arms"] = {}
    for side in ("left", "right"):
        if set(projection["arms"][side]) != {"user_index", "tool_index"}:
            raise PreparationError("Only User/Tool indices belong in projected arm configuration")
        data["arms"][side] = dict(projection["arms"][side],
            dashboard_port=29999, move_port=30003, feedback_port=30004,
            gripper={"id": 1, "baudrate": 1000000, "speed": 100, "acceleration": 1,
                     "settle_s": 0.5, "position_tolerance_ticks": 50, "torque_limit": 100,
                     "open_ticks": 0, "closed_ticks": 4095})
    root = Path(directory)
    (root/"calibration.json").write_text(dumps(projection["calibration"]))
    data["calibration_file"] = str(root/"calibration.json")
    (root/"site.json").write_text(dumps(data))
    return Config(root/"site.json")


def make_request(config, scene, task, state=None, images=None, execution=None):
    request = {"type": "inference", "schema": SCHEMA, "request_id": uuid.uuid4().hex,
            "sequence": 0, "captured_at": datetime.now(timezone.utc).isoformat(),
            "instruction": task, "images": images or [],
            "proprio_data": [] if state is None else [list(state)],
            "dsr": {"task": task, "planning_config": planning_projection(config), "scene": scene}}
    if execution is not None:
        request["execution"] = execution
    return request


def compile_request(request):
    if request.get("type") != "inference" or request.get("schema") != SCHEMA or type(request.get("sequence")) is not int or request["sequence"] != 0:
        raise PreparationError("Unsupported DSR protocol/schema or sequence")
    if not isinstance(request.get("request_id"), str) or len(request["request_id"]) != 32:
        raise PreparationError("Missing request identity")
    if age_seconds(request.get("captured_at")) > 600:
        raise PreparationError("Stale planning request")
    body = request["dsr"]
    if body["task"] not in TASKS or request.get("instruction") != body["task"]:
        raise PreparationError("Select an implemented DSR task")
    images = request.get("images", [])
    if not isinstance(images, list) or (images and (len(images) != 3 or not all(isinstance(x, str) for x in images))):
        raise PreparationError("Images must be top/left_wrist/right_wrist in that order")
    if request.get("proprio_data"):
        from dsr_xtrainer.geometry import vector
        if not isinstance(request["proprio_data"], list) or len(request["proprio_data"]) != 1:
            raise PreparationError("Expected exactly one 14D state")
        vector(request["proprio_data"][0], 14, "14D absolute joint state")
    with tempfile.TemporaryDirectory(prefix="dsr-ws-plan-") as tmp:
        config = projected_config(body["planning_config"], tmp)
        path = Path(tmp)/"scene.json"
        path.write_text(dumps(body["scene"]))
        scene = load_scene(path, config, body["task"])
        plan = build_plan(config, scene, body["task"]).to_dict()
    return {"type": "inference", "schema": SCHEMA, "request_id": request["request_id"],
            "sequence": 0, "action_space": ACTION_SPACE, "request_sha256": digest(request),
            "plan_sha256": digest(plan), "plan": plan,
            "perception": "measured_scene_json; optional images are metadata only, no object detection",
            "hardware_motion_tested": False}


def accept_plan(response, request, config, scene):
    for key, expected in [("type", "inference"), ("schema", SCHEMA),
                          ("request_id", request["request_id"]), ("sequence", 0),
                          ("action_space", ACTION_SPACE), ("request_sha256", digest(request))]:
        if response.get(key) != expected:
            raise PreparationError("Invalid server response identity or action space: "+key)
    data = response.get("plan", {})
    if response.get("plan_sha256") != digest(data):
        raise PreparationError("Plan checksum mismatch")
    if set(data) != {f.name for f in fields(Plan)} or not isinstance(data["steps"], list) or not 0 < len(data["steps"]) <= 10000:
        raise PreparationError("Invalid task plan shape")
    steps = []
    for step in data["steps"]:
        if not isinstance(step, dict) or set(step) != {f.name for f in fields(Step)}:
            raise PreparationError("Invalid skill step")
        steps.append(Step(**step))
    plan = Plan(task=data["task"], steps=steps, memory=data["memory"], schema=data["schema"], controller=data["controller"])
    validate_plan(config, plan)
    # The server cannot broaden the two qualified task contracts/checkpoints.
    expected = build_plan(config, scene, request["dsr"]["task"]).to_dict()
    if digest(expected) != digest(data):
        raise PreparationError("Remote plan differs from the locally qualified task/scene contract")
    return plan
