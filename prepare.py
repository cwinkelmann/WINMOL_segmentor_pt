#!/usr/bin/env python
"""prepare — see `python prepare.py --help`.

Thin wrapper. The implementation is in winmol_unet/cli/prepare.py so that the installed
console script `winmol-prepare` and this file run exactly the same code.
"""
import sys

from winmol_unet.cli.prepare import main

if __name__ == "__main__":
    sys.exit(main())
