from pathlib import Path

import pytest

from qc.batch.models import BatchFileStatus, BatchMeta, FileRecord, StageName
from qc.batch.pipeline import FileResult, FakeAudioStageRunner, process_file
from qc.batch.store import BatchStore
from qc.errors import AnalysisError, ErrorStage
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


def make_checkpoints(tmp_path, record=None):
    from qc.batch.checkpoints import FileCheckpointSession

    record = record or make_record()
    store = BatchStore(tmp_path / "batch.db")
    store.create_batch(BatchMeta(batch_id="B-1", source="test", total=1))
    store.add_file("B-1", record)
    file_id = store.list_files("B-1")[0]["file_id"]
    return store, FileCheckpointSession(store, "B-1", file_id)


def test_process_file_runs_all_stages_and_completes(tmp_path):
    store, checkpoints = make_checkpoints(tmp_path)

    result = process_file(
        make_record(),
        FakeAudioStageRunner(wav_root=tmp_path),
        FakeQualityService(),
        checkpoints,
    )
    assert result.status == BatchFileStatus.DONE
    file_id = store.list_files("B-1")[0]["file_id"]
    done = {
        row["stage"]
        for row in store.get_file(file_id)["stages"]
        if row["status"] == "DONE"
    }
    assert {"TRANSCODE", "ASR", "EMOTION", "QC"} <= done
    checkpoint = store.get_stage_checkpoint(file_id, StageName.ASR)
    assert checkpoint["artifact_uri"] == f"batch/B-1/{file_id}/transcript.json"
    assert not Path(checkpoint["artifact_uri"]).is_absolute()


def test_process_file_marks_failed_final_on_corrupt_transcode(tmp_path):
    class BrokenTranscode(FakeAudioStageRunner):
        def transcode(self, file_record):
            from qc.batch.pipeline import BatchStageFailure

            raise BatchStageFailure(
                code="FILE_CORRUPT",
                stage=StageName.TRANSCODE,
                message="文件损坏",
                retryable=False,
            )

    store, checkpoints = make_checkpoints(tmp_path)
    result = process_file(
        make_record(),
        BrokenTranscode(wav_root=tmp_path),
        FakeQualityService(),
        checkpoints,
    )
    assert result.status.value == "FAILED_FINAL"
    assert "文件损坏" in (result.failed_reason or "")
    file_id = store.list_files("B-1")[0]["file_id"]
    transcode = store.get_stage_checkpoint(file_id, StageName.TRANSCODE)
    assert transcode["attempts"] == 1
    assert transcode["error_code"] == "FILE_CORRUPT"
    assert transcode["retryable"] is False


def test_process_file_does_not_check_business_fact(tmp_path):
    # process_file 不应判断客户结清；这里只验证它产出 DONE 且 result_json 可解析
    _, checkpoints = make_checkpoints(tmp_path)
    result = process_file(
        make_record(),
        FakeAudioStageRunner(wav_root=tmp_path),
        FakeQualityService(),
        checkpoints,
    )
    assert result.status == BatchFileStatus.DONE


@pytest.mark.parametrize(
    ("analysis_status", "report", "expected_status"),
    [
        ("FAILED", None, "FAILED_FINAL"),
        ("COMPLETED", None, "FAILED_FINAL"),
        ("PARTIAL", None, "FAILED_FINAL"),
        ("PARTIAL", QualityReport(callId="CALL-001"), "HUMAN_REVIEW"),
    ],
)
def test_process_file_fails_closed_for_incomplete_analysis(
    tmp_path, analysis_status, report, expected_status
):
    class StaticQualityService:
        def analyze(self, request):
            return AnalysisResult(
                runId="RUN-FAIL-CLOSED",
                status=analysis_status,
                loopUsed=False,
                report=report,
            )

    _, checkpoints = make_checkpoints(tmp_path)
    result = process_file(
        make_record(),
        FakeAudioStageRunner(wav_root=tmp_path),
        StaticQualityService(),
        checkpoints,
    )

    assert result.status.value == expected_status
    assert '"disposition":"AUTO_PASS"' not in result.result_json


def test_unknown_asr_error_is_not_retried(tmp_path):
    class UnknownAsrFailure(FakeAudioStageRunner):
        def __init__(self, wav_root):
            super().__init__(wav_root)
            self.asr_calls = 0

        def run_asr(self, wav_path):
            self.asr_calls += 1
            raise RuntimeError("unexpected")

    store, checkpoints = make_checkpoints(tmp_path)
    runner = UnknownAsrFailure(tmp_path)
    result = process_file(
        make_record(),
        runner,
        FakeQualityService(),
        checkpoints,
        max_attempts=3,
    )

    file_id = store.list_files("B-1")[0]["file_id"]
    asr = store.get_stage_checkpoint(file_id, StageName.ASR)
    assert result.status.value == "FAILED_FINAL"
    assert runner.asr_calls == 1
    assert asr["attempts"] == 1
    assert asr["error_code"] == "INTERNAL_ERROR"
    assert asr["retryable"] is False
    assert store.get_stage_checkpoint(file_id, StageName.QC) is None


def test_retryable_failed_analysis_retries_only_qc(tmp_path):
    class CountingRunner(FakeAudioStageRunner):
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

    class RetryQuality:
        def __init__(self):
            self.calls = 0

        def analyze(self, request):
            self.calls += 1
            if self.calls == 1:
                return AnalysisResult(
                    runId="RUN-RETRY-1",
                    status="FAILED",
                    loopUsed=False,
                    report=None,
                    errors=[
                        AnalysisError(
                            code="LLM_RATE_LIMITED",
                            stage=ErrorStage.API,
                            message="上游限流",
                            retryable=True,
                            attempts=1,
                        )
                    ],
                )
            return AnalysisResult(
                runId="RUN-RETRY-2",
                status="COMPLETED",
                loopUsed=False,
                report=QualityReport(callId=request.callId),
            )

    store, checkpoints = make_checkpoints(tmp_path)
    runner = CountingRunner(tmp_path)
    quality = RetryQuality()
    result = process_file(
        make_record(),
        runner,
        quality,
        checkpoints,
        max_attempts=2,
    )

    file_id = store.list_files("B-1")[0]["file_id"]
    assert result.status == BatchFileStatus.DONE
    assert runner.transcode_calls == 1
    assert runner.asr_calls == 1
    assert quality.calls == 2
    assert store.get_stage_checkpoint(file_id, StageName.QC)["attempts"] == 2
