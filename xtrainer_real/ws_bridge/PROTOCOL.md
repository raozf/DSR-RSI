# DSR XTrainer WebSocket protocol

Endpoint: `ws://127.0.0.1:18000/ws`. Use the matching `run_ws_server.sh` and
`run_ws_client.sh` launchers, with an SSH tunnel for a remote server. Messages are
JSON text objects, finite numbers only, with a 16 MiB limit. The server never opens
the robot SDK. All SDK calls and Cartesian-to-joint conversion run on the robot PC.

| Field | Value / meaning |
| --- | --- |
| `schema` | `dsr.xtrainer-ws/v1` |
| `action_space` | `tcp_skill_plan_world_mm_deg` |
| execution mode | `feedback_skill_stream/v1` |
| TCP pose | `[x,y,z,rx,ry,rz]`, world-frame mm / degrees; `Rz @ Ry @ Rx` |
| `proprio_data` | One row: left 6 joint radians, left closure command, right 6 radians, right closure command |
| skill `opening` | `1=open`, `0=closed`; client maps to SDK closure (`0=open`, `1=closed` by default) |
| digest | SHA-256 of UTF-8 JSON with sorted keys, no whitespace, `ensure_ascii=False`, `allow_nan=False` |

`protocol.py` defines the envelope and hashes; `stream.py` defines execution state;
`robot_client.py` owns SDK conversion. Tasks are `align_blocks` and `cover_blocks`.

## Health and preview

Send `{"type":"health"}`. A compatible response has `type="health"`,
`status="healthy"`, the schema/action space above, supported tasks and
`execution_modes=["plan_preview","feedback_skill_stream/v1"]`. Health checks can
continue during an episode; the client serializes health and inference exchanges.

`make_request()` constructs a preview request with `type="inference"`, schema,
a 32-character `request_id`, `sequence=0`, timezone-aware `captured_at`,
`instruction=<task>`, `images=[]`, `proprio_data=[]`, and:

```text
dsr:
  task: align_blocks | cover_blocks
  planning_config: schema, example_only, arms (User/Tool indices), safety, tasks, calibration
  scene: measured task geometry, calibration identity and acquisition time
```

The projection excludes IP addresses, serial ports and SDK paths. Optional `images`
contains three base64 strings in top / left wrist / right wrist order. The initial
request is archived; these images do not produce object poses or affect the policy.
The preview request must be no older than 600 seconds. Hardware mode also requires
measured configuration and a scene within the configured age limit (default 120 s).

The response echoes the schema/request identity/sequence and adds `action_space`,
`request_sha256`, `plan_sha256`, and `plan`. The client checks the complete plan
against its own task/scene contract before opening any hardware connection.
Without `--mock` or `--execute`, this preview is the whole operation.

## Feedback execution

For execution, the initial request also contains:

```json
{"execution":{"mode":"feedback_skill_stream/v1","run_id":"<journal run>","mock":true}}
```

Use `mock=false` for hardware. The preview response adds
`execution={mode,stream_id,run_id}`. One WebSocket connection owns one episode.
This is a measured-scene Code Policy/Skills port: feedback gates advancement of the
precompiled plan, with external observations at checkpoints. It does not implement
the full DSR Agent/RSI or online visual replanning.

Before step `i` (starting at zero), the client waits for feedback sequences from
**both** arms to advance, then sends:

```text
type: inference
schema: dsr.xtrainer-ws/v1
phase: step
stream_id: <negotiated stream>
parent_request_id: <preview request_id>
request_id: <new 32-character identity>
sequence: i
captured_at: <fresh timestamp>
proprio_data: [[<14 SDK-order values>]]
acknowledgement: null for i=0; exact receipt for step i-1 thereafter
observation:
  source: mock | controller_feedback_with_commanded_grippers
  gripper_position_measured: false
  robot_state:
    left: mode, error, drag, paused, joints_deg, pose_world_mm_deg, sequence
    right: mode, error, drag, paused, joints_deg, pose_world_mm_deg, sequence
```

The server checks identity, order, timestamp (at most 4 s old), measured joint
units/order, arm status, TCP arrival at the previous target, and the stationary
arm's park state. Initial and final states require both arms at park. Reusing a
feedback sequence, receipt, request sequence or another connection's stream fails.

Each accepted response echoes the request identity/sequence and contains
`execution_mode`, `stream_id`, `action_space`, `request_sha256`, `plan_sha256`,
`skill_actions=[<one Step>]`, `done=false`. A Step has these fields (unused
values are null / empty strings / empty objects):

```text
kind: move | gripper | checkpoint
arm: left | right | null
pose_world_mm_deg: [x,y,z,rx,ry,rz] | null
opening: 0..1 | null
stage: task stage identifier
expected: checkpoint predicate | {}
instruction: checkpoint instruction
```

The client rejects a skill that differs from the validated plan. A move is converted
through the controller's read-only IK/FK, checked for limits and interpolation path,
then sent through the unchanged SDK's `command_joint_state` with only the active
arm's `flag_in` set. Gripper values in observations are **commanded**, not measured;
the upstream node returns fixed gripper values. A grasp needs its external checkpoint.

After completing an issued step, the next request carries:

```text
acknowledgement:
  index: i
  step_sha256: <digest of the exact issued Step>
  completed: true
  actual: <execution result object>
  checkpoint: <accepted observation record, only for checkpoint steps>
```

Gripper results report `opening` in mock mode or `opening_commanded` on hardware.
Move completion also needs fresh measured TCP arrival in the next request;
`completed=true` alone is insufficient. Checkpoint records bind run/stage/calibration,
must be fresh and acquired after the request (2 s clock tolerance), and must satisfy
the task predicate. Hardware records require `source=manual|vision`, `simulated=false`,
`approved=true`; mock records use `source=mock`, `simulated=true`.

After the final skill, the client sends one more fresh-feedback request containing
its receipt. Only a valid response with `skill_actions=[]`, `done=true` completes
the run. Thus N skills require N+1 accepted step responses; preview and health
responses are excluded from `summary.json`'s `remote_policy_calls` count.

## Errors and journals

Errors return `type="error"`, the schema and `detail`. An episode is permanently
closed after a protocol error or completion. The matching client aborts on a stale
response, rejection, disconnect, lost final confirmation or malformed reply. It
does not replay requests, reconnect, retry robot commands or continue locally.
The execution monitor attempts the existing two-arm stop path on failure.

`events.jsonl` records each `remote_inference` request/response. `summary.json`
records completed steps, accepted remote response count, execution mode and final
acknowledgement. `plan_sha256` in wire messages hashes canonical JSON; the existing
run summary's identically named field hashes the pretty-printed `plan.json` file,
so these two digests use different representations.

The upstream `dobot_client/client_ws.py --check-only` remains compatible with
health. Stock Motus/OpenPI clients expect `predicted_actions` chunks of 14 joint
values and cannot execute these typed Cartesian skills. Use this kit's client;
no SDK source edit is needed.
