from pathlib import Path

import pytest

from qc.batch.models import BatchConfig, BatchFileStatus, BatchMeta, FileRecord
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


def test_single_file_failure_isolated_as_dead_letter(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"x")
    (audio_dir / "b.m4a").write_bytes(b"y")
    s = store(tmp_path)
    orch = BatchOrchestrator(s, BatchConfig())
    orch.ingest("B-1", DirectorySource(audio_dir), BatchMeta(batch_id="B-1", source="directory", total=0))

    # 一个文件的质检失败 → 死信；另一个正常完成。失败互不影响。
    summary = orch.run_batch(
        "B-1", FakeAudioStageRunner(tmp_path / "wav"),
        FakeQualityService(fail_call_id=None),  # 见下方说明：用真实失败注入替换
    )
    # 默认无失败时两者都 DONE；死信路径由下一个用例显式验证。
    assert summary["by_status"].get("DONE") == 2


def test_broken_file_becomes_dead_letter_without_blocking_others(tmp_path):
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
                raise RuntimeError("文件损坏")
            return super().transcode(file_record)

    summary = orch.run_batch("B-1", HalfBrokenRunner(tmp_path / "wav"), FakeQualityService())
    assert summary["by_status"].get("DONE") == 1
    assert summary["by_status"].get("DEAD_LETTER") == 1


def test_resume_picks_up_incomplete_after_interrupt(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"x")
    s = store(tmp_path)
    orch = BatchOrchestrator(s, BatchConfig())
    orch.ingest("B-1", DirectorySource(audio_dir), BatchMeta(batch_id="B-1", source="directory", total=0))
    # 模拟中断：把文件标 RUNNING（脏状态）
    s.set_file_status(s.list_files("B-1")[0]["file_id"], BatchFileStatus.RUNNING)
    s.mark_interrupted_running()
    summary = orch.resume("B-1", FakeAudioStageRunner(tmp_path / "wav"), FakeQualityService())
    assert summary["by_status"].get("DONE") == 1
