"""Full native DSR-RSI evaluation using four policy/simulator GPU pairs."""
import sys
from pathlib import Path
from dsr.paths import repository_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_robodojo_agent_eval import main as full_main


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    return full_main([
        '--policy-name', 'DSRRSI', '--agent-mode', 'frozen',
        '--output-root', str(repository_root()/'results/dsr_robodojo'),
        *argv,
    ])


if __name__ == '__main__':
    raise SystemExit(main())
