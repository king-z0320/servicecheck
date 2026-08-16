from concurrent.futures import ThreadPoolExecutor

import pytest

from qc.batch import store as store_module
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
    store.record_stage(
        file_id,
        StageRecord(
            stage=StageName.ASR,
            status="DONE",
            duration_ms=6800,
            attempts=1,
            artifact_uri=str(tmp_path / "transcript.json"),
            sha256="abc123",
            producer_version="fake-asr-v1",
        ),
    )
    record = store.get_file(file_id)
    stages = record["stages"]
    checkpoint = [s for s in stages if s["stage"] == "ASR"][-1]
    assert checkpoint["status"] == "DONE"
    assert checkpoint["attempts"] == 1
    assert checkpoint["sha256"] == "abc123"
    assert checkpoint["producer_version"] == "fake-asr-v1"


def test_finalize_file_writes_failure_atomically_and_cannot_be_overwritten(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta())
    store.add_file("B-1", make_record())
    file_id = store.list_files("B-1")[0]["file_id"]
    assert store.claim_file(file_id, BatchFileStatus.PENDING) is True
    store.finalize_file(
        file_id,
        BatchFileStatus.FAILED_FINAL,
        '{"request":1}',
        "",
        failed_reason="ASR 超时",
    )
    record = store.get_file(file_id)
    assert record["status"] == "FAILED_FINAL"
    assert record["failed_reason"] == "ASR 超时"
    assert record["request_json"] == '{"request":1}'
    assert record["result_json"] == ""
    with pytest.raises(store_module.StateConflictError):
        store.finalize_file(
            file_id,
            BatchFileStatus.DONE,
            '{"request":2}',
            '{"report":2}',
        )
    assert store.get_file(file_id)["request_json"] == '{"request":1}'


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
    # 第二个文件跑到一半后被恢复标记为 INTERRUPTED，第三个已完成。
    fids = [f["file_id"] for f in store.list_files("B-1")]
    assert store.claim_file(fids[1], BatchFileStatus.PENDING) is True
    store.mark_interrupted_running("B-1")
    assert store.claim_file(fids[2], BatchFileStatus.PENDING) is True
    store.finalize_file(fids[2], BatchFileStatus.DONE, "{}", "{}")
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
    assert store.claim_file(fids[0], BatchFileStatus.PENDING) is True
    count = store.mark_interrupted_running("B-1")
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
    assert store.get_stage_checkpoint(file_id, StageName.ASR)["attempts"] == 1


def test_stage_attempts_and_error_update_current_real_stage(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta())
    store.add_file("B-1", make_record())
    file_id = store.list_files("B-1")[0]["file_id"]
    assert store.begin_stage(file_id, StageName.ASR) == 1
    store.fail_stage(
        file_id,
        StageName.ASR,
        error_code="UPSTREAM_TIMEOUT",
        retryable=True,
        error="temporary timeout",
        duration_ms=10,
    )
    failed = store.get_stage_checkpoint(file_id, StageName.ASR)
    assert failed["error_code"] == "UPSTREAM_TIMEOUT"
    assert failed["retryable"] is True
    assert failed["attempts"] == 1
    assert store.begin_stage(file_id, StageName.ASR) == 2
    store.complete_stage(
        file_id,
        StageName.ASR,
        artifact_uri=str(tmp_path / "transcript.json"),
        sha256="hash",
        producer_version="fake-asr-v1",
        duration_ms=8,
    )
    checkpoint = store.get_stage_checkpoint(file_id, StageName.ASR)
    assert checkpoint["status"] == "DONE"
    assert checkpoint["attempts"] == 2
    assert checkpoint["error_code"] is None
    assert checkpoint["retryable"] is None
    assert store.get_stage_checkpoint(file_id, StageName.QC) is None


def test_claim_file_is_compare_and_set(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta(total=1))
    store.add_file("B-1", make_record())
    file_id = store.list_files("B-1")[0]["file_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: store.claim_file(file_id, BatchFileStatus.PENDING),
                range(2),
            )
        )

    assert sorted(results) == [False, True]
    assert store.get_file(file_id)["status"] == "RUNNING"


def test_illegal_file_transition_is_rejected(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(make_meta(total=1))
    store.add_file("B-1", make_record())
    file_id = store.list_files("B-1")[0]["file_id"]

    with pytest.raises(store_module.StateConflictError):
        store.set_file_status(file_id, BatchFileStatus.DONE)

    assert store.get_file(file_id)["status"] == "PENDING"


def test_mark_interrupted_running_only_changes_target_batch(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    for batch_id, key in (("B-1", "k1"), ("B-2", "k2")):
        store.create_batch(make_meta(batch_id=batch_id, total=1))
        store.add_file(batch_id, make_record(key=key))
        file_id = store.list_files(batch_id)[0]["file_id"]
        assert store.claim_file(file_id, BatchFileStatus.PENDING) is True

    assert store.mark_interrupted_running("B-1") == 1
    assert store.list_files("B-1")[0]["status"] == "INTERRUPTED"
    assert store.list_files("B-2")[0]["status"] == "RUNNING"


def test_export_record_keeps_only_minimum_checkpoint_fields(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    out_path = tmp_path / "B-1.json"
    store.begin_export("B-1", "json", out_path, "batch-json-export-v1")
    store.complete_export(
        "B-1",
        "json",
        out_path,
        "batch-json-export-v1",
        "abc123",
    )
    record = store.get_export_record("B-1", "json", out_path)
    assert record == {
        "export_id": record["export_id"],
        "batch_id": "B-1",
        "format": "json",
        "artifact_uri": str(out_path),
        "status": "DONE",
        "sha256": "abc123",
        "producer_version": "batch-json-export-v1",
        "error_code": None,
    }
