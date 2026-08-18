from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest


def test_local_artifact_store_round_trips_bytes_and_hash(tmp_path):
    from qc.artifact_store import LocalArtifactStore

    store = LocalArtifactStore(tmp_path / "artifacts")
    reference = store.put_bytes(
        "batch/B-1/1/transcript.json",
        b'[{"turnId":"T1"}]',
        mime_type="application/json",
    )

    assert reference.uri == "batch/B-1/1/transcript.json"
    assert reference.sha256 == hashlib.sha256(b'[{"turnId":"T1"}]').hexdigest()
    assert reference.size == len(b'[{"turnId":"T1"}]')
    assert store.read_bytes(reference.uri) == b'[{"turnId":"T1"}]'
    assert store.verify_sha256(reference.uri, reference.sha256) is True
    assert store.resolve_for_read(reference.uri).is_relative_to(tmp_path / "artifacts")


@pytest.mark.parametrize(
    "key",
    [
        "",
        "../outside.txt",
        "batch/../../outside.txt",
        "/absolute.txt",
        r"C:\outside.txt",
        r"batch\..\outside.txt",
        "nul\x00byte",
    ],
)
def test_artifact_store_rejects_unsafe_keys_before_opening_files(tmp_path, key):
    from qc.artifact_store import LocalArtifactStore

    store = LocalArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError):
        store.put_bytes(key, b"unsafe")


def test_failed_atomic_replace_does_not_expose_partial_final_file(tmp_path):
    from qc.artifact_store import LocalArtifactStore

    class FailingStore(LocalArtifactStore):
        def _atomic_replace(self, source: Path, target: Path) -> None:
            raise OSError("injected replace failure")

    store = FailingStore(tmp_path / "artifacts")
    with pytest.raises(OSError, match="injected"):
        store.put_bytes("exports/B-1/report.json", b"partial")

    assert store.exists("exports/B-1/report.json") is False


def test_tampered_artifact_fails_hash_verification(tmp_path):
    from qc.artifact_store import LocalArtifactStore

    store = LocalArtifactStore(tmp_path / "artifacts")
    reference = store.put_bytes("audio/A-1/source.wav", b"original")
    store.resolve_for_read(reference.uri).write_bytes(b"tampered")

    assert store.verify_sha256(reference.uri, reference.sha256) is False


def test_symlink_cannot_escape_artifact_root_when_supported(tmp_path):
    from qc.artifact_store import LocalArtifactStore

    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "escape"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    store = LocalArtifactStore(root)
    with pytest.raises(ValueError):
        store.put_bytes("escape/created.txt", b"unsafe")
    assert not (outside / "created.txt").exists()
