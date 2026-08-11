from qc.batch.models import (
    BatchFileStatus,
    BatchMeta,
    FileRecord,
    StageName,
    StageRecord,
)
from qc.batch.store import BatchStore


def make_meta(batch_id="B-1", total=2):
    return BatchMeta(batch_id=batch_id, source="directory", total=total)


def make_record(uri="/tmp/a.m4a", key="abc123", call_id="CALL-001"):
    return FileRecord(source_uri=uri, idempotency_key=key, callId=call_id)


def test_create_batch_and_add_file(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta())
    added = store.add_file("B-1", make_record())
    assert added is True
    files = store.list_files("B-1")
    assert len(files) == 1
    assert files[0]["status"] == BatchFileStatus.PENDING.value
    assert files[0]["source_uri"] == "/tmp/a.m4a"


def test_duplicate_idempotency_key_is_ignored(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta())
    assert store.add_file("B-1", make_record()) is True
    # 同一幂等键再次添加 → 返回 False，不重复
    assert store.add_file("B-1", make_record()) is False
    assert len(store.list_files("B-1")) == 1


def test_record_stage_persists_and_is_readable(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta())
    store.add_file("B-1", make_record())
    file_id = store.list_files("B-1")[0]["file_id"]
    store.record_stage(file_id, StageRecord(stage=StageName.ASR, status="DONE", duration_ms=6800))
    record = store.get_file(file_id)
    stages = record["stages"]
    assert any(s["stage"] == "ASR" and s["status"] == "DONE" for s in stages)


def test_set_file_status_and_failed_reason(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta())
    store.add_file("B-1", make_record())
    file_id = store.list_files("B-1")[0]["file_id"]
    store.set_file_status(file_id, BatchFileStatus.DEAD_LETTER, failed_reason="ASR 超时")
    record = store.get_file(file_id)
    assert record["status"] == "DEAD_LETTER"
    assert record["failed_reason"] == "ASR 超时"


def test_state_survives_store_recreation(tmp_path):
    db = tmp_path / "batch.db"
    store = BatchStore(db)
    store.create_batch(make_meta())
    store.add_file("B-1", make_record())
    reloaded = BatchStore(db)
    assert len(reloaded.list_files("B-1")) == 1


def test_resume_candidates_returns_pending_and_interrupted(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta(total=3))
    store.add_file("B-1", FileRecord(source_uri="/a", idempotency_key="k1"))
    store.add_file("B-1", FileRecord(source_uri="/b", idempotency_key="k2"))
    store.add_file("B-1", FileRecord(source_uri="/c", idempotency_key="k3"))
    # 第二个文件跑到一半（RUNNING），第三个已完成
    fids = [f["file_id"] for f in store.list_files("B-1")]
    store.set_file_status(fids[1], BatchFileStatus.RUNNING)
    store.set_file_status(fids[2], BatchFileStatus.DONE)
    candidates = store.resume_candidates("B-1")
    statuses = {c["status"] for c in candidates}
    assert "DONE" not in statuses
    assert {c["file_id"] for c in candidates} == {fids[0], fids[1]}


def test_mark_interrupted_running_clears_dirty_state(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta(total=2))
    store.add_file("B-1", FileRecord(source_uri="/a", idempotency_key="k1"))
    store.add_file("B-1", FileRecord(source_uri="/b", idempotency_key="k2"))
    fids = [f["file_id"] for f in store.list_files("B-1")]
    store.set_file_status(fids[0], BatchFileStatus.RUNNING)
    count = store.mark_interrupted_running()
    assert count == 1
    assert store.get_file(fids[0])["status"] == "INTERRUPTED"


def test_last_completed_stage(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta())
    store.add_file("B-1", make_record())
    file_id = store.list_files("B-1")[0]["file_id"]
    assert store.last_completed_stage(file_id) is None
    store.record_stage(file_id, StageRecord(stage=StageName.TRANSCODE, status="DONE"))
    store.record_stage(file_id, StageRecord(stage=StageName.ASR, status="DONE"))
    assert store.last_completed_stage(file_id) == StageName.ASR


def test_increment_stage_attempts(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta())
    store.add_file("B-1", make_record())
    file_id = store.list_files("B-1")[0]["file_id"]
    store.record_stage(file_id, StageRecord(stage=StageName.ASR, status="FAILED", attempts=1))
    next_attempts = store.increment_stage_attempts(file_id, StageName.ASR)
    assert next_attempts == 2
