#!/usr/bin/env python
"""infer — see `python infer.py --help`.

Thin wrapper. The implementation is in winmol_unet/cli/infer.py so that the installed
console script `winmol-infer` and this file run exactly the same code.
"""
import sys

from winmol_unet.cli.infer import main

if __name__ == "__main__":
    sys.exit(main())
