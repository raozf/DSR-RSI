# DSR XTrainer 双臂适配器

本目录提供 DSR WebSocket 服务端、机器人控制端客户端和越疆 XTrainer
SDK 适配。SDK 按 `ws-sdk.lock.json` 中的提交获取，适配器不修改 SDK。

## 安装与回环检查

```bash
./scripts/install_ws.sh
./ws_dry_run.sh
./run_ws_server.sh --host 127.0.0.1 --port 18000
./run_ws_client.sh --check-only
```

回环检查使用本地 MockRig，不连接真机。消息格式见
`ws_bridge/PROTOCOL.md`。

## 真机流程

根据现场实测填写 `local/` 下的配置，先运行只读预检并校验计划：

```bash
./run_ws_client.sh --preflight --config local/site.json \
  --transport local/transport.json
./run_ws_align_blocks.sh --config local/site.json \
  --scene local/align_blocks.scene.json
```

确认已有 SDK `robot_node`、标定、停靠姿态和急停均已检查后，显式加入
`--execute`：

```bash
./run_ws_align_blocks.sh --execute --config local/site.json \
  --transport local/transport.json --scene local/align_blocks.scene.json
./run_ws_cover_blocks.sh --execute --config local/site.json \
  --transport local/transport.json --scene local/cover_blocks.scene.json
```

适配器支持上述两个预编译 Code Policy 任务，把共享世界坐标系中的笛卡尔
目标转换为控制器 IK/FK，再通过已有 node 发送经过校验的 14 维关节指令。
适配器不会启动、使能或标定机器人，也不会根据图像推断场景。执行期间保持
实体急停可用。

`site.json`、`calibration.json`、`transport.json` 必须使用当前实测值。场景
文件的坐标单位为毫米和度，协议、回执和消息身份见 `ws_bridge/PROTOCOL.md`。
不要把 node 的本地 pickle 接口暴露到控制电脑以外。

每臂需要实测 `world_T_base`、工具 TCP、关节范围，以及同一姿态下的
`park_tcp_world_mm_deg` 和 `park_joints_deg`。TCP 格式为
`[x,y,z,rx,ry,rz]`，旋转顺序为 `Rz @ Ry @ Rx`。在 `transport.json`
核对夹爪映射与初始开度；完成标定后再将 `example_only` 设为 `false`，
填写标定确认字段。场景 `captured_at` 使用带时区的真实采集时间，默认
有效期为 120 秒。

任务检查点使用 `examples/*.observation.json` 的对应格式提交实测观测：

```bash
./.venv/bin/python -m dsr_xtrainer.cli checkpoint \
  --request runs/<run>/checkpoints/<stage>.request.json \
  --observation local/measured-observation.json
```

观测需填写 `source=manual/vision`、`simulated=false`、`approved=true`
和真实时间，程序同时校验 run、stage、calibration 及任务条件。
`--checkpoint-dir` 可接外部观测程序。独立急停入口为
`./emergency_stop.sh --config local/site.json`。
