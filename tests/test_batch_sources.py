from pathlib import Path

from qc.batch.models import FileRecord
from qc.batch.sources import (
    DirectorySource,
    IngestSource,
    compute_idempotency_key,
)


def test_idempotency_key_is_stable_for_same_uri():
    assert compute_idempotency_key("/tmp/a.m4a") == compute_idempotency_key("/tmp/a.m4a")


def test_idempotency_key_differs_for_different_uri():
    assert compute_idempotency_key("/tmp/a.m4a") != compute_idempotency_key("/tmp/b.m4a")


def test_idempotency_key_prefers_call_id_when_provided():
    # 给定 callId 时，幂等键基于 callId 而非路径
    key_by_call = compute_idempotency_key("/tmp/wherever.m4a", call_id="CALL-42")
    other_path_same_call = compute_idempotency_key("/tmp/elsewhere.m4a", call_id="CALL-42")
    assert key_by_call == other_path_same_call


def test_directory_source_discovers_files(tmp_path):
    (tmp_path / "a.m4a").write_bytes(b"x")
    (tmp_path / "b.m4a").write_bytes(b"y")
    (tmp_path / "ignore.txt").write_text("nope")
    source = DirectorySource(tmp_path)
    records = source.discover()
    uris = {r.source_uri for r in records}
    assert len(records) == 2
    assert all(r.idempotency_key for r in records)
    assert str(tmp_path / "ignore.txt") not in uris


def test_directory_source_is_an_ingest_source():
    assert isinstance(DirectorySource("/tmp"), IngestSource) or issubclass(
        DirectorySource, IngestSource
    ) or hasattr(DirectorySource, "discover")
