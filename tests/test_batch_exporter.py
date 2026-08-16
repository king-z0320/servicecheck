import csv
import hashlib
import json

import pytest

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
    out = Exporter(store).export_json("B-1", tmp_path / "out.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["batch_id"] == "B-1"
    assert len(data["files"]) == 2
    record = store.get_export_record("B-1", "json", out)
    assert record["status"] == "DONE"
    assert record["producer_version"] == "batch-json-export-v1"
    assert record["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_export_csv_has_summary_columns_and_failed_final(tmp_path):
    store = seed_store(tmp_path)
    out = Exporter(store).export_csv("B-1", tmp_path / "out.csv")
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
    record = store.get_export_record("B-1", "csv", out)
    assert record["status"] == "DONE"
    assert record["producer_version"] == "batch-csv-export-v1"


def test_explicit_export_rewrites_file_and_updates_record(tmp_path):
    store = seed_store(tmp_path)
    out_path = tmp_path / "out.json"
    out_path.write_text("stale", encoding="utf-8")

    first = Exporter(store).export_json("B-1", out_path)
    expected = first.read_bytes()
    out_path.write_text("stale-again", encoding="utf-8")
    second = Exporter(store).export_json("B-1", out_path)

    assert second.read_bytes() == expected
    record = store.get_export_record("B-1", "json", out_path)
    assert record["sha256"] == hashlib.sha256(expected).hexdigest()


def test_export_failure_records_error_without_changing_file_status(tmp_path):
    store = seed_store(tmp_path)
    statuses_before = [row["status"] for row in store.list_files("B-1")]
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("x", encoding="utf-8")
    out_path = blocked_parent / "out.json"

    with pytest.raises(OSError):
        Exporter(store).export_json("B-1", out_path)

    record = store.get_export_record("B-1", "json", out_path)
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

    out = Exporter(store).export_csv("B-1", tmp_path / "legacy.csv")
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    legacy = [row for row in rows if row["status"] == "DEAD_LETTER"][0]
    assert legacy["score"] == ""
    assert legacy["disposition"] == ""
