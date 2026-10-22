"""Remote DSR code-policy planner. Never imports or opens a robot SDK."""
import argparse
import json
import sys

from websockets.sync.server import serve
from dsr_xtrainer.model import PreparationError
from .protocol import ACTION_SPACE, MAX_MESSAGE_BYTES, SCHEMA, compile_request, dumps, loads
from .stream import STREAM_MODE, ServerEpisode


def handler(connection):
    if connection.request.path != "/ws":
        connection.close(1008, "Use /ws")
        return
    episode = None
    for raw in connection:
        try:
            request = loads(raw)
            if request.get("type") == "health":
                response = {"type": "health", "status": "healthy", "schema": SCHEMA,
                            "backend": "dsr-calibrated-code-policy", "action_space": ACTION_SPACE,
                            "tasks": ["align_blocks", "cover_blocks"],
                            "execution_modes": ["plan_preview", STREAM_MODE]}
            elif request.get("type") == "inference":
                if request.get("phase") == "step":
                    if episode is None:
                        raise PreparationError("No active DSR feedback stream on this WebSocket connection")
                    response = episode.infer(request)
                else:
                    if episode is not None:
                        raise PreparationError("One execution stream is allowed per WebSocket connection")
                    response = compile_request(request)
                    if "execution" in request:
                        episode = ServerEpisode(request, response)
                        response["execution"] = episode.metadata
            else:
                raise PreparationError("Supported types: health, inference (DSR schema required)")
        except (PreparationError, KeyError, TypeError, ValueError, OSError) as exc:
            if episode is not None:
                episode.closed = True
            response = {"type": "error", "schema": SCHEMA, "detail": str(exc)}
        connection.send(dumps(response))


def make_server(host="127.0.0.1", port=18000):
    return serve(handler, host, port, max_size=MAX_MESSAGE_BYTES, compression=None,
                 open_timeout=5, close_timeout=2, ping_interval=20, ping_timeout=10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Use an SSH tunnel for remote access")
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args(argv)
    try:
        with make_server(args.host, args.port) as server:
            print(json.dumps({"ready": True, "url": f"ws://{args.host}:{server.socket.getsockname()[1]}/ws",
                              "backend": "DSR code policy; no hardware connection"}), flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
