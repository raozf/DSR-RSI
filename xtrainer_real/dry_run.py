#!/usr/bin/env python3
from dsr_xtrainer.cli import main
from dsr_xtrainer.config import KIT_ROOT
import sys

if __name__ == "__main__":
    raise SystemExit(main(["run", "--mock", "--config", str(KIT_ROOT/"examples/site.demo.json"), *sys.argv[1:]]))
