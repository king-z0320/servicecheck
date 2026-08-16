from qc.batch.models import BatchMeta, BatchFileStatus, FileRecord, StageName, StageRecord
from qc.batch.report import render_progress
from qc.batch.store import BatchStore


def test_render_progress_shows_counts_and_top_failure(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(BatchMeta(batch_id="B-1", source="directory", total=3))
    store.add_file("B-1", FileRecord(source_uri="/a", idempotency_key="k1", callId="C1"))
    store.add_file("B-1", FileRecord(source_uri="/b", idempotency_key="k2", callId="C2"))
    store.add_file("B-1", FileRecord(source_uri="/c", idempotency_key="k3", callId="C3"))
    fids = [f["file_id"] for f in store.list_files("B-1")]
    assert store.claim_file(fids[0], BatchFileStatus.PENDING) is True
    store.finalize_file(fids[0], BatchFileStatus.DONE, "{}", "{}")
    # 旧 DEAD_LETTER 仅作为历史数据兼容场景，不由新状态机产生。
    with store._connect() as db:
        db.execute(
            "UPDATE batch_files SET status='DEAD_LETTER', failed_reason=? WHERE file_id=?",
            ("ASR 超时", fids[1]),
        )
    text = render_progress(store, "B-1")
    assert "B-1" in text
    assert "DONE" in text
    assert "DEAD_LETTER" in text
    assert "ASR 超时" in text


def test_render_progress_handles_empty_batch(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(BatchMeta(batch_id="B-1", source="directory", total=0))
    text = render_progress(store, "B-1")
    assert "total" in text.lower() or "总计" in text


def test_render_progress_shows_stage_durations_and_throughput(tmp_path):
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(BatchMeta(batch_id="B-2", source="directory", total=2))
    store.add_file("B-2", FileRecord(source_uri="/a", idempotency_key="k1", callId="C1"))
    store.add_file("B-2", FileRecord(source_uri="/b", idempotency_key="k2", callId="C2"))
    fids = [f["file_id"] for f in store.list_files("B-2")]
    # 两个 DONE 文件 + 阶段耗时记录
    for fid in fids:
        store.record_stage(
            fid,
            StageRecord(
                stage=StageName.TRANSCODE,
                status="DONE",
                duration_ms=1200.0,
            ),
        )
        store.record_stage(
            fid,
            StageRecord(stage=StageName.ASR, status="DONE", duration_ms=6800.0),
        )
        assert store.claim_file(fid, BatchFileStatus.PENDING) is True
        store.finalize_file(fid, BatchFileStatus.DONE, "{}", "{}")
    text = render_progress(store, "B-2")
    assert "阶段耗时" in text
    assert "TRANSCODE" in text
    assert "ASR" in text
    # 吞吐行应该出现（started_at 由 create_batch 写入）
    assert "吞吐" in text
    assert "文件/分钟" in text
