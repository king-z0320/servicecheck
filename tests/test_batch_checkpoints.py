import hashlib
import json
from pathlib import Path

import pytest

from qc.batch.models import BatchMeta, FileRecord, StageName
from qc.batch.store import BatchStore
from qc.models import AnalysisResult, QualityReport, TranscriptTurn


def make_session(tmp_path):
    from qc.batch.checkpoints import FileCheckpointSession

    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(BatchMeta(batch_id="B-1", source="test", total=1))
    store.add_file(
        "B-1",
        FileRecord(
            source_uri=str(tmp_path / "a.m4a"),
            idempotency_key="k1",
            callId="CALL-1",
        ),
    )
    file_id = store.list_files("B-1")[0]["file_id"]
    return store, file_id, FileCheckpointSession(store, "B-1", file_id)


def save_stage(session, stage, value, version):
    assert session.begin(stage) >= 1
    return session.complete(stage, value, version, duration_ms=1.0)


def test_saves_and_loads_all_four_stage_artifacts(tmp_path):
    store, file_id, session = make_session(tmp_path)
    source_wav = tmp_path / "source.wav"
    source_wav.write_bytes(b"FAKEWAV")
    turns = [
        TranscriptTurn(
            turnId="T1",
            speaker="客户",
            text="测试",
            start=0,
            end=1,
        )
    ]
    emotion = {"customer": "neutral"}
    analysis = AnalysisResult(
        runId="RUN-1",
        status="COMPLETED",
        loopUsed=False,
        report=QualityReport(callId="CALL-1"),
    )

    canonical_wav = save_stage(
        session, StageName.TRANSCODE, source_wav, "fake-transcode-v1"
    )
    save_stage(session, StageName.ASR, turns, "fake-asr-v1")
    save_stage(session, StageName.EMOTION, emotion, "fake-emotion-v1")
    save_stage(session, StageName.QC, analysis, "batch-qc-v1")

    assert canonical_wav == session.load(StageName.TRANSCODE, "fake-transcode-v1")
    assert session.load(StageName.ASR, "fake-asr-v1") == turns
    assert session.load(StageName.EMOTION, "fake-emotion-v1") == emotion
    assert session.load(StageName.QC, "batch-qc-v1") == analysis
    for stage in (StageName.TRANSCODE, StageName.ASR, StageName.EMOTION, StageName.QC):
        checkpoint = store.get_stage_checkpoint(file_id, stage)
        artifact = checkpoint["artifact_uri"]
        assert checkpoint["attempts"] == 1
        assert checkpoint["status"] == "DONE"
        assert str(tmp_path / "batch_artifacts") in artifact
        assert checkpoint["sha256"] == hashlib.sha256(
            Path(artifact).read_bytes()
        ).hexdigest()


@pytest.mark.parametrize("invalid_kind", ["missing", "hash", "version", "schema"])
def test_invalid_asr_artifact_is_not_reused(tmp_path, invalid_kind):
    store, file_id, session = make_session(tmp_path)
    turns = [
        TranscriptTurn(
            turnId="T1",
            speaker="客户",
            text="测试",
            start=0,
            end=1,
        )
    ]
    save_stage(session, StageName.ASR, turns, "fake-asr-v1")
    checkpoint = store.get_stage_checkpoint(file_id, StageName.ASR)
    artifact = checkpoint["artifact_uri"]

    if invalid_kind == "missing":
        Path(artifact).unlink()
        requested_version = "fake-asr-v1"
    elif invalid_kind == "hash":
        Path(artifact).write_text("[]", encoding="utf-8")
        requested_version = "fake-asr-v1"
    elif invalid_kind == "version":
        requested_version = "fake-asr-v2"
    else:
        Path(artifact).write_text(json.dumps({"not": "a transcript"}), encoding="utf-8")
        from qc.batch.models import StageRecord

        store.record_stage(
            file_id,
            StageRecord(
                stage=StageName.ASR,
                status="DONE",
                attempts=1,
                artifact_uri=artifact,
                sha256=hashlib.sha256(Path(artifact).read_bytes()).hexdigest(),
                producer_version="fake-asr-v1",
            ),
        )
        requested_version = "fake-asr-v1"

    assert session.load(StageName.ASR, requested_version) is None
    assert store.get_stage_checkpoint(file_id, StageName.ASR)["attempts"] == 1


def test_valid_load_does_not_increment_attempts(tmp_path):
    store, file_id, session = make_session(tmp_path)
    turns = [TranscriptTurn(turnId="T1", speaker="客户", text="测试", end=1)]
    save_stage(session, StageName.ASR, turns, "fake-asr-v1")

    assert session.load(StageName.ASR, "fake-asr-v1") == turns
    assert session.load(StageName.ASR, "fake-asr-v1") == turns
    assert store.get_stage_checkpoint(file_id, StageName.ASR)["attempts"] == 1


def test_checkpoint_invalidation_uses_only_declared_dependencies(tmp_path):
    store, file_id, session = make_session(tmp_path)
    source_wav = tmp_path / "source.wav"
    source_wav.write_bytes(b"FAKEWAV")
    turns = [TranscriptTurn(turnId="T1", speaker="客户", text="测试", end=1)]
    analysis = AnalysisResult(
        runId="RUN-1",
        status="COMPLETED",
        loopUsed=False,
        report=QualityReport(callId="CALL-1"),
    )
    save_stage(session, StageName.TRANSCODE, source_wav, "fake-transcode-v1")
    save_stage(session, StageName.ASR, turns, "fake-asr-v1")
    save_stage(session, StageName.EMOTION, {"customer": "neutral"}, "fake-emotion-v1")
    save_stage(session, StageName.QC, analysis, "batch-qc-v1")

    session.invalidate_downstream(StageName.ASR)

    assert store.get_stage_checkpoint(file_id, StageName.TRANSCODE)["status"] == "DONE"
    assert store.get_stage_checkpoint(file_id, StageName.ASR)["status"] == "DONE"
    assert store.get_stage_checkpoint(file_id, StageName.EMOTION)["status"] == "DONE"
    assert store.get_stage_checkpoint(file_id, StageName.QC)["status"] == "PENDING"

    save_stage(session, StageName.QC, analysis, "batch-qc-v1")
    session.invalidate_downstream(StageName.TRANSCODE)

    assert store.get_stage_checkpoint(file_id, StageName.TRANSCODE)["status"] == "DONE"
    assert store.get_stage_checkpoint(file_id, StageName.ASR)["status"] == "PENDING"
    assert store.get_stage_checkpoint(file_id, StageName.EMOTION)["status"] == "PENDING"
    assert store.get_stage_checkpoint(file_id, StageName.QC)["status"] == "PENDING"


@pytest.mark.parametrize("invalid_kind", ["missing", "hash", "version"])
def test_invalid_asr_checkpoint_reexecutes_asr_once(tmp_path, invalid_kind):
    from qc.batch.pipeline import execute_stage

    store, file_id, session = make_session(tmp_path)
    original = [TranscriptTurn(turnId="T1", speaker="客户", text="旧文本", end=1)]
    replacement = [TranscriptTurn(turnId="T2", speaker="客户", text="新文本", end=1)]
    save_stage(session, StageName.ASR, original, "fake-asr-v1")
    checkpoint = store.get_stage_checkpoint(file_id, StageName.ASR)
    artifact = Path(checkpoint["artifact_uri"])
    requested_version = "fake-asr-v1"
    if invalid_kind == "missing":
        artifact.unlink()
    elif invalid_kind == "hash":
        artifact.write_text("[]", encoding="utf-8")
    else:
        requested_version = "fake-asr-v2"

    calls = 0

    def run_asr():
        nonlocal calls
        calls += 1
        return replacement

    restored = execute_stage(
        session,
        StageName.ASR,
        requested_version,
        run_asr,
        max_attempts=3,
    )

    assert restored == replacement
    assert calls == 1
    checkpoint = store.get_stage_checkpoint(file_id, StageName.ASR)
    assert checkpoint["status"] == "DONE"
    assert checkpoint["attempts"] == 2
    assert checkpoint["producer_version"] == requested_version


def test_resume_does_not_exceed_stage_retry_budget(tmp_path):
    from qc.batch.pipeline import BatchStageFailure, execute_stage

    store, file_id, session = make_session(tmp_path)
    for _ in range(2):
        session.begin(StageName.ASR)
        session.fail(
            StageName.ASR,
            "UPSTREAM_TIMEOUT",
            True,
            "上游调用超时",
            duration_ms=1,
        )
    calls = 0

    def run_asr():
        nonlocal calls
        calls += 1
        return []

    with pytest.raises(BatchStageFailure) as raised:
        execute_stage(
            session,
            StageName.ASR,
            "fake-asr-v1",
            run_asr,
            max_attempts=2,
        )

    assert raised.value.attempts == 2
    assert calls == 0
    assert store.get_stage_checkpoint(file_id, StageName.ASR)["attempts"] == 2
