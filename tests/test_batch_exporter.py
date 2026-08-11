import csv
import json

from qc.batch.exporter import Exporter
from qc.batch.models import BatchMeta, BatchFileStatus, FileRecord
from qc.batch.store import BatchStore


def seed_store(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(BatchMeta(batch_id="B-1", source="directory", total=2))
    store.add_file("B-1", FileRecord(source_uri="/a", idempotency_key="k1", callId="CALL-1"))
    store.add_file("B-1", FileRecord(source_uri="/b", idempotency_key="k2", callId="CALL-2"))
    fids = [f["file_id"] for f in store.list_files("B-1")]
    store.save_file_report(
        fids[0], BatchFileStatus.DONE,
        "{}", '{"report":{"score":80,"disposition":"AUTO_VIOLATION"}}',
    )
    store.set_file_status(fids[1], BatchFileStatus.DEAD_LETTER, failed_reason="ASR 超时")
    return store


def test_export_json_contains_all_files(tmp_path):
    store = seed_store(tmp_path)
    out = Exporter(store).export_json("B-1", tmp_path / "out.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["batch_id"] == "B-1"
    assert len(data["files"]) == 2


def test_export_csv_has_summary_columns_and_dead_letter(tmp_path):
    store = seed_store(tmp_path)
    out = Exporter(store).export_csv("B-1", tmp_path / "out.csv")
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert {r["callId"] for r in rows} == {"CALL-1", "CALL-2"}
    statuses = {r["status"] for r in rows}
    assert "DONE" in statuses and "DEAD_LETTER" in statuses
    done = [r for r in rows if r["status"] == "DONE"][0]
    assert done["score"] == "80"
    assert done["disposition"] == "AUTO_VIOLATION"
    dead = [r for r in rows if r["status"] == "DEAD_LETTER"][0]
    assert dead["failed_reason"] == "ASR 超时"
    assert dead["score"] == ""  # 死信无 result_json，提取应安全返回空
