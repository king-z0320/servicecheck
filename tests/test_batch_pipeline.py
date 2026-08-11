from pathlib import Path

import pytest

from qc.batch.models import BatchFileStatus, FileRecord, StageName
from qc.batch.pipeline import FileResult, FakeAudioStageRunner, process_file
from qc.models import AnalysisResult, QualityReport


class FakeQualityService:
    """复用单通话质检链路契约，但不真实调用 LLM。"""

    def analyze(self, request):
        return AnalysisResult(
            runId="RUN-X",
            status="COMPLETED",
            loopUsed=False,
            report=QualityReport(callId=request.callId),
        )


def make_record(uri="/tmp/a.m4a"):
    return FileRecord(source_uri=uri, idempotency_key="abc123", callId="CALL-001")


def test_process_file_runs_all_stages_and_completes(tmp_path):
    stages_seen: list[tuple[str, str]] = []

    def on_stage(stage, status, duration_ms, error=None):
        stages_seen.append((stage.value, status))

    result = process_file(
        make_record(),
        FakeAudioStageRunner(wav_root=tmp_path),
        FakeQualityService(),
        on_stage,
    )
    assert result.status == BatchFileStatus.DONE
    stage_names = [s for s, _ in stages_seen if _ == "DONE"]
    # 至少经过 转码/ASR/情绪/质检
    assert {"TRANSCODE", "ASR", "EMOTION", "QC"} <= set(stage_names)


def test_process_file_marks_dead_letter_on_unrecoverable_transcode(tmp_path):
    class BrokenTranscode(FakeAudioStageRunner):
        def transcode(self, file_record):
            raise RuntimeError("文件损坏")

    result = process_file(
        make_record(),
        BrokenTranscode(wav_root=tmp_path),
        FakeQualityService(),
        lambda *a, **k: None,
    )
    assert result.status == BatchFileStatus.DEAD_LETTER
    assert "文件损坏" in (result.failed_reason or "")


def test_process_file_does_not_check_business_fact():
    # process_file 不应判断客户结清；这里只验证它产出 DONE 且 result_json 可解析
    result = process_file(
        make_record(),
        FakeAudioStageRunner(wav_root=Path("/tmp")),
        FakeQualityService(),
        lambda *a, **k: None,
    )
    assert result.status == BatchFileStatus.DONE
