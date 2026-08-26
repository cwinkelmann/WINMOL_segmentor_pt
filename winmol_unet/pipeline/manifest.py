"""Every pipeline stage writes one of these beside its outputs.

A stage is *stale* when the inputs or parameters recorded here differ from the ones
now being asked for. That is the whole mechanism behind resuming a run: the stages
are separate directories either way, but without this they would have to be re-run
in full every time.

Large rasters are identified by (size, mtime) rather than a hash. Hashing a 19 GB
orthomosaic costs more than re-running the stage the hash is there to guard.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

MANIFEST_NAME = "_manifest.json"
TOOL_VERSION = 1
HASH_LIMIT = 64 * 1024 * 1024


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() or None
    except Exception:
        return None


def input_identity(path, role):
    """How a stage recognises one of its inputs across runs."""
    st = os.stat(path)
    return {"path": os.path.abspath(path), "role": role, "size": st.st_size,
            "mtime": st.st_mtime,
            "sha256": _sha256(path) if st.st_size <= HASH_LIMIT else None}


def write_manifest(stage_dir, stage, params, inputs, outputs, counts):
    os.makedirs(stage_dir, exist_ok=True)
    path = os.path.join(stage_dir, MANIFEST_NAME)
    doc = {"stage": stage, "tool_version": TOOL_VERSION, "git_commit": _git_commit(),
           "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "params": params, "inputs": inputs, "outputs": outputs, "counts": counts}
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    return path


def read_manifest(stage_dir):
    path = os.path.join(stage_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _comparable(ident):
    """The parts of an input identity that decide staleness.

    mtime is deliberately excluded when a hash is available: copying a file or
    checking it out again changes mtime without changing content, and re-running a
    multi-hour resample for that would make the manifest a nuisance rather than a
    tool.
    """
    if ident.get("sha256"):
        return (ident["path"], ident["role"], ident["sha256"])
    return (ident["path"], ident["role"], ident["size"], ident["mtime"])


def is_stale(stage_dir, stage, params, inputs):
    prev = read_manifest(stage_dir)
    if prev is None:
        return True
    if prev.get("stage") != stage or prev.get("tool_version") != TOOL_VERSION:
        return True
    if prev.get("params") != params:
        return True
    return ([_comparable(i) for i in prev.get("inputs", [])]
            != [_comparable(i) for i in inputs])
