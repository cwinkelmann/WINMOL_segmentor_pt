#!/usr/bin/env python
"""evaluate — see `python evaluate.py --help`.

Thin wrapper. The implementation is in winmol_unet/cli/evaluate.py so that the installed
console script `winmol-evaluate` and this file run exactly the same code.
"""
import sys

from winmol_unet.cli.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
