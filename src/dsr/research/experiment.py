"""DSR CLI entry for the bounded numerical source-evolution experiment."""

import argparse
import json
from pathlib import Path

from .control_evolution import run, HarnessState


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--rollback', action='store_true')
    args = parser.parse_args(argv)
    if args.rollback:
        state = HarnessState(Path(args.workspace).resolve() / 'state')
        result = state.rollback(state.resolve()['revision'])
    else:
        result = run(args.workspace, rounds=args.rounds)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
