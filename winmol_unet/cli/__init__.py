"""Command-line entry points.

The four root-level scripts (`prepare.py`, `train.py`, `infer.py`, `evaluate.py`) are thin
wrappers around the `main(argv)` functions here. Keeping the implementations inside the
package is what lets the same commands work after `pip install` via the console scripts
declared in pyproject.toml — a bare script at the repo root is not importable once the
package is installed somewhere else.

Nothing is imported here at module level: `import winmol_unet.cli` must stay as cheap as
`import winmol_unet`, since the analyzer's install has neither torch nor GDAL. Each
submodule imports its own heavy dependencies inside `main()`.
"""
