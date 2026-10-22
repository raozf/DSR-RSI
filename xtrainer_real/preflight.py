#!/usr/bin/env python3
from dsr_xtrainer.cli import main
import sys

if __name__ == "__main__":
    raise SystemExit(main(["preflight", *sys.argv[1:]]))
