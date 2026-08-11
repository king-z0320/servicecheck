import pytest
from pydantic import ValidationError

from qc.batch.models import (
    BatchConfig,
    BatchFileStatus,
    BatchMeta,
    FileRecord,
    StageName,
    StageRecord,
)


def test_file_record_requires_idempotency_key():
    with pytest.raises(ValidationError):
        FileRecord(source_uri="/tmp/a.m4a")


def test_file_record_accepts_optional_callid():
    record = FileRecord(
        source_uri="/tmp/a.m4a",
        idempotency_key="abc123",
        callId="CALL-001",
    )
    assert record.callId == "CALL-001"
    assert record.idempotency_key == "abc123"


def test_batch_file_status_has_dead_letter_and_human_review():
    values = {s.value for s in BatchFileStatus}
    assert {"DEAD_LETTER", "HUMAN_REVIEW", "INTERRUPTED", "DONE"} <= values


def test_stage_name_covers_full_pipeline():
    names = {s.value for s in StageName}
    assert {"TRANSCODE", "ASR", "EMOTION", "EVENT_EXTRACT", "RAG", "AUDIT", "QC"} <= names


def test_stage_record_defaults():
    record = StageRecord(stage=StageName.ASR)
    assert record.status == "PENDING"
    assert record.attempts == 0


def test_batch_config_has_tunable_concurrency_with_defaults():
    config = BatchConfig()
    assert config.cpu_workers >= 1
    assert config.gpu_workers == 1
    assert config.llm_rpm > 0
    assert config.max_attempts == 3


def test_batch_meta_records_total():
    meta = BatchMeta(batch_id="B-1", source="directory", total=100)
    assert meta.total == 100
    assert meta.created_at is not None
