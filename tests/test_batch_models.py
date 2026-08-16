import pytest
from pydantic import ValidationError

from qc.batch import models as batch_models
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


def test_batch_file_status_has_failed_final_and_legacy_dead_letter():
    values = {s.value for s in BatchFileStatus}
    assert {
        "FAILED_FINAL",
        "DEAD_LETTER",
        "HUMAN_REVIEW",
        "INTERRUPTED",
        "DONE",
    } <= values


def test_stage_name_covers_full_pipeline():
    names = {s.value for s in StageName}
    assert {"TRANSCODE", "ASR", "EMOTION", "EVENT_EXTRACT", "RAG", "AUDIT", "QC"} <= names


def test_stage_record_defaults():
    record = StageRecord(stage=StageName.ASR)
    assert record.status == "PENDING"
    assert record.attempts == 0


def test_stage_record_contains_artifact_checkpoint_fields():
    record = StageRecord(
        stage=StageName.ASR,
        status="DONE",
        attempts=1,
        artifact_uri="E:/artifacts/transcript.json",
        sha256="abc123",
        producer_version="fake-asr-v1",
        error_code=None,
        retryable=None,
    )
    assert record.artifact_uri.endswith("transcript.json")
    assert record.sha256 == "abc123"
    assert record.producer_version == "fake-asr-v1"
    assert record.error_code is None
    assert record.retryable is None


def test_stage_record_rejects_invalid_status_and_negative_attempts():
    with pytest.raises(ValidationError):
        StageRecord(stage=StageName.ASR, status="SKIPPED")
    with pytest.raises(ValidationError):
        StageRecord(stage=StageName.ASR, attempts=-1)


def test_file_status_transitions_are_centralized_and_terminal():
    transitions = batch_models.VALID_FILE_TRANSITIONS
    assert transitions[BatchFileStatus.PENDING] == {BatchFileStatus.RUNNING}
    assert transitions[BatchFileStatus.INTERRUPTED] == {BatchFileStatus.RUNNING}
    assert transitions[BatchFileStatus.RUNNING] == {
        BatchFileStatus.DONE,
        BatchFileStatus.HUMAN_REVIEW,
        BatchFileStatus.FAILED_FINAL,
        BatchFileStatus.INTERRUPTED,
    }
    assert transitions[BatchFileStatus.DONE] == set()
    assert transitions[BatchFileStatus.HUMAN_REVIEW] == set()
    assert transitions[BatchFileStatus.FAILED_FINAL] == set()


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
