"""Release-manifest hashing and exclusion rules."""

import hashlib
import json

from scripts.release_manifest import build_manifest, sha256_file


def test_manifest_hashes_source_and_skips_artifacts(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("hello\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cache.bin").write_text("ignored")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "model.py").write_text("ignored")

    manifest = build_manifest(tmp_path, dataset_revision="rev-1")

    assert manifest["file_count"] == 2
    assert set(manifest["files"]) == {"README.md", "src/model.py"}
    assert manifest["files"]["src/model.py"]["sha256"] == sha256_file(tmp_path / "src" / "model.py")
    assert manifest["dataset"]["hub_revision"] == "rev-1"


def test_manifest_records_checkpoint_digest(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    (tmp_path / "a.py").write_text("pass\n")

    manifest = build_manifest(tmp_path, checkpoint=checkpoint)

    expected = hashlib.sha256(b"weights").hexdigest()
    assert manifest["checkpoint"]["sha256"] == expected
    assert manifest["checkpoint"]["bytes"] == 7


def test_manifest_is_json_serialisable(tmp_path):
    (tmp_path / "a.py").write_text("pass\n")
    payload = json.dumps(build_manifest(tmp_path))
    assert '"file_count": 1' in payload
