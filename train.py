#!/usr/bin/env python
"""train — see `python train.py --help`.

Thin wrapper. The implementation is in winmol_unet/cli/train.py so that the installed
console script `winmol-train` and this file run exactly the same code.
"""
import sys

from winmol_unet.cli.train import main

if __name__ == "__main__":
    sys.exit(main())
