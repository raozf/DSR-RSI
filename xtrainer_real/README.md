# DSR XTrainer real-robot adapter

This directory provides a WebSocket DSR server and a client for the Dobot
XTrainer SDK. The SDK source is fetched at the pinned commit in
`ws-sdk.lock.json`; the adapter leaves the SDK unchanged.

## Install and dry run

```bash
./scripts/install_ws.sh
./ws_dry_run.sh
./run_ws_server.sh --host 127.0.0.1 --port 18000
./run_ws_client.sh --check-only
```

The dry run uses a local MockRig and does not connect to a robot. The WebSocket
wire format is documented in `ws_bridge/PROTOCOL.md`.

## Robot operation

Fill the measured files in `local/` using the templates and run the read-only
preflight before requesting a plan:

```bash
./run_ws_client.sh --preflight --config local/site.json \
  --transport local/transport.json
./run_ws_align_blocks.sh --config local/site.json \
  --scene local/align_blocks.scene.json
```

After the existing SDK `robot_node` is running, calibration and emergency-stop
checks are complete, execute explicitly:

```bash
./run_ws_align_blocks.sh --execute --config local/site.json \
  --transport local/transport.json --scene local/align_blocks.scene.json
./run_ws_cover_blocks.sh --execute --config local/site.json \
  --transport local/transport.json --scene local/cover_blocks.scene.json
```

The adapter supports the two precompiled Code Policy plans above. It converts
world-frame Cartesian targets to controller IK/FK and sends validated 14D joint
commands through the existing node. It does not start, enable, or calibrate the
robot and does not infer scene geometry from images. Keep the physical stop
button available throughout operation.

`local/site.json`, `calibration.json`, and `transport.json` must contain current
measured values. Scene files use millimetres and degrees in a shared world frame;
the protocol file defines the message identities and receipt rules. Do not expose
the node's local pickle interface outside the robot control computer.
