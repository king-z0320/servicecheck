import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from qc.batch.models import (
    BatchConfig,
    BatchFileStatus,
    BatchMeta,
    FileRecord,
    StageName,
    StageRecord,
)
from qc.batch.orchestrator import BatchOrchestrator
from qc.batch.pipeline import FakeAudioStageRunner
from qc.batch.sources import DirectorySource
from qc.batch.store import BatchStore
from qc.models import AnalysisResult, QualityReport


class FakeQualityService:
    def __init__(self, fail_call_id=None):
        self.fail_call_id = fail_call_id

    def analyze(self, request):
        if self.fail_call_id and request.callId == self.fail_call_id:
            raise RuntimeError("LLM 超时")
        return AnalysisResult(
            runId="RUN-X",
            status="COMPLETED",
            loopUsed=False,
            report=QualityReport(callId=request.callId),
        )


def store(tmp_path):
    """Helper: open a fresh handle on the batch.db under tmp_path."""
    return BatchStore(tmp_path / "batch.db")


def test_ingest_discovers_and_dedups(tmp_path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "a.m4a").write_bytes(b"x")
    (tmp_path / "audio" / "b.m4a").write_bytes(b"y")
    orch = BatchOrchestrator(store(tmp_path), BatchConfig())
    orch.ingest("B-1", DirectorySource(tmp_path / "audio"), BatchMeta(batch_id="B-1", source="directory", total=0))
    assert len(store(tmp_path).list_files("B-1")) == 2
    # 再次 ingest 不重复
    orch.ingest("B-1", DirectorySource(tmp_path / "audio"), BatchMeta(batch_id="B-1", source="directory", total=0))
    assert len(store(tmp_path).list_files("B-1")) == 2


def test_run_batch_completes_all_files(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"x")
    (audio_dir / "b.m4a").write_bytes(b"y")
    s = store(tmp_path)
    orch = BatchOrchestrator(s, BatchConfig())
    orch.ingest("B-1", DirectorySource(audio_dir), BatchMeta(batch_id="B-1", source="directory", total=0))
    summary = orch.run_batch("B-1", FakeAudioStageRunner(tmp_path / "wav"), FakeQualityService())
    assert summary["by_status"].get("DONE") == 2


def test_multiple_files_complete_without_interference(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"x")
    (audio_dir / "b.m4a").write_bytes(b"y")
    s = store(tmp_path)
    orch = BatchOrchestrator(s, BatchConfig())
    orch.ingest("B-1", DirectorySource(audio_dir), BatchMeta(batch_id="B-1", source="directory", total=0))

    summary = orch.run_batch(
        "B-1", FakeAudioStageRunner(tmp_path / "wav"),
        FakeQualityService(fail_call_id=None),
    )
    assert summary["by_status"].get("DONE") == 2


def test_broken_file_becomes_failed_final_without_blocking_others(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "good.m4a").write_bytes(b"x")
    (audio_dir / "bad.m4a").write_bytes(b"y")
    s = store(tmp_path)
    orch = BatchOrchestrator(s, BatchConfig())
    orch.ingest("B-1", DirectorySource(audio_dir), BatchMeta(batch_id="B-1", source="directory", total=0))

    class HalfBrokenRunner(FakeAudioStageRunner):
        def transcode(self, file_record):
            if "bad" in file_record.source_uri:
                from qc.batch.pipeline import BatchStageFailure

                raise BatchStageFailure(
                    code="FILE_CORRUPT",
                    stage=StageName.TRANSCODE,
                    message="文件损坏",
                    retryable=False,
                )
            return super().transcode(file_record)

    summary = orch.run_batch("B-1", HalfBrokenRunner(tmp_path / "wav"), FakeQualityService())
    assert summary["by_status"].get("DONE") == 1
    assert summary["by_status"].get("FAILED_FINAL") == 1


def test_resume_picks_up_incomplete_after_interrupt(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"x")
    s = store(tmp_path)
    orch = BatchOrchestrator(s, BatchConfig())
    orch.ingest("B-1", DirectorySource(audio_dir), BatchMeta(batch_id="B-1", source="directory", total=0))
    # 模拟中断：把文件标 RUNNING（脏状态）
    file_id = s.list_files("B-1")[0]["file_id"]
    assert s.claim_file(file_id, BatchFileStatus.PENDING) is True
    summary = orch.resume("B-1", FakeAudioStageRunner(tmp_path / "wav"), FakeQualityService())
    assert summary["by_status"].get("DONE") == 1


def test_resume_reuses_valid_asr_artifact_without_calling_runner(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"x")
    s = store(tmp_path)
    orch = BatchOrchestrator(s, BatchConfig(llm_rpm=600_000))
    orch.ingest(
        "B-1",
        DirectorySource(audio_dir),
        BatchMeta(batch_id="B-1", source="directory", total=0),
    )
    file_id = s.list_files("B-1")[0]["file_id"]
    wav_reference = orch.artifact_store.put_bytes(
        f"batch/B-1/{file_id}/transcode.wav", b"FAKEWAV"
    )
    transcript_content = json.dumps(
            [
                {
                    "turnId": "T0001",
                    "speaker": "客户",
                    "text": "我已经还完了",
                    "start": 0.0,
                    "end": 2.0,
                }
            ],
            ensure_ascii=False,
        ).encode("utf-8")
    transcript_reference = orch.artifact_store.put_bytes(
        f"batch/B-1/{file_id}/transcript.json", transcript_content
    )
    s.record_stage(
        file_id,
        StageRecord(
            stage=StageName.TRANSCODE,
            status="DONE",
            attempts=1,
            artifact_uri=wav_reference.uri,
            sha256=wav_reference.sha256,
            producer_version="fake-transcode-v1",
        ),
    )
    s.record_stage(
        file_id,
        StageRecord(
            stage=StageName.ASR,
            status="DONE",
            attempts=1,
            artifact_uri=transcript_reference.uri,
            sha256=transcript_reference.sha256,
            producer_version="fake-asr-v1",
        ),
    )
    assert s.claim_file(file_id, BatchFileStatus.PENDING) is True

    class SpyRunner(FakeAudioStageRunner):
        def __init__(self, wav_root):
            super().__init__(wav_root)
            self.transcode_calls = 0
            self.asr_calls = 0

        def transcode(self, file_record):
            self.transcode_calls += 1
            return super().transcode(file_record)

        def run_asr(self, wav_path):
            self.asr_calls += 1
            return super().run_asr(wav_path)

    runner = SpyRunner(tmp_path / "wav")
    summary = orch.resume("B-1", runner, FakeQualityService())

    assert summary["by_status"].get("DONE") == 1
    assert runner.transcode_calls == 0
    assert runner.asr_calls == 0
    restored_asr = s.get_stage_checkpoint(file_id, StageName.ASR)
    assert restored_asr["artifact_uri"] == transcript_reference.uri
    assert restored_asr["sha256"] == transcript_reference.sha256


def test_retryable_asr_failure_updates_only_asr_attempts(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"x")
    s = store(tmp_path)
    orch = BatchOrchestrator(
        s, BatchConfig(max_attempts=2, llm_rpm=600_000)
    )
    orch.ingest(
        "B-1",
        DirectorySource(audio_dir),
        BatchMeta(batch_id="B-1", source="directory", total=0),
    )

    class RetryOnceAsr(FakeAudioStageRunner):
        def __init__(self, wav_root):
            super().__init__(wav_root)
            self.transcode_calls = 0
            self.asr_calls = 0

        def transcode(self, file_record):
            self.transcode_calls += 1
            return super().transcode(file_record)

        def run_asr(self, wav_path):
            self.asr_calls += 1
            if self.asr_calls == 1:
                raise TimeoutError("temporary ASR timeout")
            return super().run_asr(wav_path)

    runner = RetryOnceAsr(tmp_path / "wav")
    summary = orch.run_batch("B-1", runner, FakeQualityService())
    file_row = s.list_files("B-1")[0]
    latest_by_stage = {
        stage: [row for row in file_row["stages"] if row["stage"] == stage][-1]
        for stage in ("TRANSCODE", "ASR", "QC")
    }

    assert summary["by_status"].get("DONE") == 1
    assert runner.transcode_calls == 1
    assert runner.asr_calls == 2
    assert latest_by_stage["TRANSCODE"]["attempts"] == 1
    assert latest_by_stage["ASR"]["attempts"] == 2
    assert latest_by_stage["QC"]["error_code"] is None


def test_concurrent_duplicate_processing_has_one_effective_report(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"x")

    class CoordinatedStore(BatchStore):
        def __init__(self, path):
            self._candidate_barrier = threading.Barrier(2)
            self._candidate_reads = 0
            self._candidate_lock = threading.Lock()
            super().__init__(path)

        def list_files(self, batch_id, statuses=None):
            rows = super().list_files(batch_id, statuses)
            with self._candidate_lock:
                should_wait = statuses is None and self._candidate_reads < 2
                if should_wait:
                    self._candidate_reads += 1
            if should_wait:
                self._candidate_barrier.wait(timeout=2)
            return rows

    s = CoordinatedStore(tmp_path / "batch.db")
    first = BatchOrchestrator(s, BatchConfig(llm_rpm=600_000))
    second = BatchOrchestrator(s, BatchConfig(llm_rpm=600_000))
    first.ingest(
        "B-1",
        DirectorySource(audio_dir),
        BatchMeta(batch_id="B-1", source="directory", total=0),
    )

    class CountingRunner(FakeAudioStageRunner):
        def __init__(self, wav_root):
            super().__init__(wav_root)
            self.asr_calls = 0
            self._lock = threading.Lock()

        def run_asr(self, wav_path):
            with self._lock:
                self.asr_calls += 1
            time.sleep(0.05)
            return super().run_asr(wav_path)

    class CountingQuality(FakeQualityService):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self._lock = threading.Lock()

        def analyze(self, request):
            with self._lock:
                self.calls += 1
            return super().analyze(request)

    runner = CountingRunner(tmp_path / "wav")
    quality = CountingQuality()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                lambda orch: orch.run_batch("B-1", runner, quality),
                (first, second),
            )
        )

    file_row = s.list_files("B-1")[0]
    assert runner.asr_calls == 1
    assert quality.calls == 1
    assert file_row["status"] == "DONE"
    assert file_row["result_json"]
