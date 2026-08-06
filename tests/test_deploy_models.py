import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from deploy_models_to_release import sha256_file, build_manifest


def test_sha256_file_matches_known_digest(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"winmol")
    # sha256("winmol")
    import hashlib
    assert sha256_file(str(p)) == hashlib.sha256(b"winmol").hexdigest()


def test_build_manifest_lists_every_asset_with_hash_and_size():
    entries = [
        {"asset": "unet_fp32.onnx", "title": "Original UNet (fp32)", "backend": "any",
         "sha256": "abc123", "size_mb": 118.4, "notes": "reference, F1 0.760"},
        {"asset": "unet_w05_int8_cpu.onnx", "title": "CPU-optimised", "backend": "CPU",
         "sha256": "def456", "size_mb": 7.5, "notes": "10x, F1 0.760"},
    ]
    md = build_manifest("models-v1", entries)
    assert "models-v1" in md
    for e in entries:
        assert e["asset"] in md
        assert e["sha256"] in md
        assert e["notes"] in md
    # a markdown table header is present
    assert "| asset " in md.lower() or "| asset|" in md.lower().replace(" ", "")
