"""A real loopback WebSocket exchange and full mock execution of both tasks."""
import argparse
import json
import subprocess
import sys
import threading

from dsr_xtrainer.config import KIT_ROOT
from .server_adapter import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(KIT_ROOT/"runs"))
    args = parser.parse_args()
    with make_server(port=0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"ws://127.0.0.1:{server.socket.getsockname()[1]}/ws"
        try:
            for task in ("align_blocks", "cover_blocks"):
                result = subprocess.run([sys.executable, "-m", "ws_bridge.client_adapter", "--url", url,
                                         "--task", task, "--mock", "--output-root", args.output_root])
                if result.returncode:
                    return result.returncode
            print(json.dumps({"websocket_mock_tasks_completed": 2, "hardware_motion_tested": False}))
        finally:
            server.shutdown()
            thread.join(timeout=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
