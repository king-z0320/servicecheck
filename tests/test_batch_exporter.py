import csv
import hashlib
import json

import pytest

from qc.artifact_store import LocalArtifactStore
from qc.batch.exporter import Exporter
from qc.batch.models import BatchMeta, BatchFileStatus, FileRecord
from qc.batch.store import BatchStore


def seed_store(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(BatchMeta(batch_id="B-1", source="directory", total=2))
    store.add_file("B-1", FileRecord(source_uri="/a", idempotency_key="k1", callId="CALL-1"))
    store.add_file("B-1", FileRecord(source_uri="/b", idempotency_key="k2", callId="CALL-2"))
    fids = [f["file_id"] for f in store.list_files("B-1")]
    assert store.claim_file(fids[0], BatchFileStatus.PENDING) is True
    store.finalize_file(
        fids[0],
        BatchFileStatus.DONE,
        "{}",
        '{"report":{"score":80,"disposition":"AUTO_VIOLATION"}}',
    )
    assert store.claim_file(fids[1], BatchFileStatus.PENDING) is True
    store.finalize_file(
        fids[1],
        BatchFileStatus.FAILED_FINAL,
        "{}",
        "",
        failed_reason="ASR 超时",
    )
    return store


def test_export_json_contains_all_files(tmp_path):
    store = seed_store(tmp_path)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    uri = "exports/B-1/out.json"
    out = Exporter(store, artifacts).export_json("B-1", uri)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["batch_id"] == "B-1"
    assert len(data["files"]) == 2
    record = store.get_export_record("B-1", "json", uri)
    assert record["status"] == "DONE"
    assert record["artifact_uri"] == uri
    assert record["producer_version"] == "batch-json-export-v1"
    assert record["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_export_csv_has_summary_columns_and_failed_final(tmp_path):
    store = seed_store(tmp_path)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    uri = "exports/B-1/out.csv"
    out = Exporter(store, artifacts).export_csv("B-1", uri)
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert {r["callId"] for r in rows} == {"CALL-1", "CALL-2"}
    statuses = {r["status"] for r in rows}
    assert "DONE" in statuses and "FAILED_FINAL" in statuses
    done = [r for r in rows if r["status"] == "DONE"][0]
    assert done["score"] == "80"
    assert done["disposition"] == "AUTO_VIOLATION"
    dead = [r for r in rows if r["status"] == "FAILED_FINAL"][0]
    assert dead["failed_reason"] == "ASR 超时"
    assert dead["score"] == ""
    record = store.get_export_record("B-1", "csv", uri)
    assert record["status"] == "DONE"
    assert record["producer_version"] == "batch-csv-export-v1"


def test_explicit_export_rewrites_file_and_updates_record(tmp_path):
    store = seed_store(tmp_path)
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    uri = "exports/B-1/out.json"
    artifacts.put_bytes(uri, b"stale")

    first = Exporter(store, artifacts).export_json("B-1", uri)
    expected = first.read_bytes()
    artifacts.put_bytes(uri, b"stale-again")
    second = Exporter(store, artifacts).export_json("B-1", uri)

    assert second.read_bytes() == expected
    record = store.get_export_record("B-1", "json", uri)
    assert record["sha256"] == hashlib.sha256(expected).hexdigest()


def test_export_failure_records_error_without_changing_file_status(tmp_path):
    store = seed_store(tmp_path)
    statuses_before = [row["status"] for row in store.list_files("B-1")]

    class FailingArtifactStore(LocalArtifactStore):
        def put_bytes(self, uri, content, *, mime_type=None):
            raise OSError("injected artifact write failure")

    artifacts = FailingArtifactStore(tmp_path / "artifacts")
    uri = "exports/B-1/out.json"

    with pytest.raises(OSError, match="injected"):
        Exporter(store, artifacts).export_json("B-1", uri)

    record = store.get_export_record("B-1", "json", uri)
    assert record["status"] == "FAILED"
    assert record["error_code"] == "EXPORT_FAILED"
    assert [row["status"] for row in store.list_files("B-1")] == statuses_before


def test_legacy_dead_letter_is_exported_without_score(tmp_path):
    store = seed_store(tmp_path)
    file_id = store.list_files("B-1")[0]["file_id"]
    with store._connect() as db:
        db.execute(
            "UPDATE batch_files SET status='DEAD_LETTER', result_json=NULL WHERE file_id=?",
            (file_id,),
        )

    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    out = Exporter(store, artifacts).export_csv(
        "B-1", "exports/B-1/legacy.csv"
    )
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    legacy = [row for row in rows if row["status"] == "DEAD_LETTER"][0]
    assert legacy["score"] == ""
    assert legacy["disposition"] == ""
