from __future__ import annotations

from pathlib import Path
from time import monotonic
import time
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from qc.batch.checkpoints import FileCheckpointSession
from qc.batch.models import BatchFileStatus, FileRecord, StageName
from qc.errors import PipelineFailure
from qc.batch.retry_policy import RetryPolicy, classify_error
from qc.models import (
    AnalysisRequest,
    AnalysisResult,
    ReviewDisposition,
    TranscriptTurn,
)


class AudioStageRunner(Protocol):
    """Audio stage contract used by the batch pipeline."""

    def transcode(self, file_record: FileRecord) -> Path: ...

    def run_asr(self, wav_path: Path) -> list[TranscriptTurn]: ...

    def run_emotion(self, wav_path: Path) -> dict: ...

    def producer_version(self, stage: StageName) -> str: ...


class FakeAudioStageRunner:
    """Offline runner used by batch tests and the current batch skeleton."""

    _VERSIONS = {
        StageName.TRANSCODE: "fake-transcode-v1",
        StageName.ASR: "fake-asr-v1",
        StageName.EMOTION: "fake-emotion-v1",
    }

    def __init__(self, wav_root: Path):
        self.wav_root = wav_root

    def transcode(self, file_record: FileRecord) -> Path:
        self.wav_root.mkdir(parents=True, exist_ok=True)
        wav_path = self.wav_root / f"{Path(file_record.source_uri).stem}.wav"
        wav_path.write_bytes(b"FAKEWAV")
        return wav_path

    def run_asr(self, wav_path: Path) -> list[TranscriptTurn]:
        return [
            TranscriptTurn(
                turnId="T0001",
                speaker="客户",
                text="我已经还完了",
                start=0.0,
                end=2.0,
            )
        ]

    def run_emotion(self, wav_path: Path) -> dict:
        return {"agent": "neutral", "customer": "neutral"}

    def producer_version(self, stage: StageName) -> str:
        return self._VERSIONS[stage]


class FileResult(BaseModel):
    status: BatchFileStatus
    request_json: str
    result_json: str
    failed_reason: str | None = None
    llm_request_count: int = 0


class BatchStageFailure(RuntimeError):
    """Structured failure for one actual batch stage."""

    def __init__(
        self,
        code: str,
        stage: StageName,
        message: str,
        retryable: bool,
        attempts: int = 0,
    ):
        self.code = code
        self.stage = stage
        self.message = message
        self.retryable = retryable
        self.attempts = attempts
        super().__init__(f"{code}: {message}")


def _stage_failure(exc: Exception, stage: StageName) -> BatchStageFailure:
    if isinstance(exc, BatchStageFailure):
        return BatchStageFailure(
            code=exc.code,
            stage=stage,
            message=exc.message,
            retryable=exc.retryable,
            attempts=exc.attempts,
        )
    if isinstance(exc, PipelineFailure):
        return BatchStageFailure(
            code=str(exc.error.code),
            stage=stage,
            message=exc.error.message,
            retryable=exc.error.retryable,
        )
    if isinstance(exc, TimeoutError):
        return BatchStageFailure(
            code="UPSTREAM_TIMEOUT",
            stage=stage,
            message="上游调用超时",
            retryable=True,
        )
    return BatchStageFailure(
        code="INTERNAL_ERROR",
        stage=stage,
        message="批量阶段发生内部错误",
        retryable=False,
    )


def execute_stage(
    checkpoints: FileCheckpointSession,
    stage: StageName,
    producer_version: str,
    action,
    *,
    max_attempts: int,
    retry_policy: RetryPolicy | None = None,
    stage_timeout_seconds: float | None = None,
    run_deadline_at: float | None = None,
    monotonic_fn=monotonic,
):
    retry_policy = retry_policy or RetryPolicy(
        max_attempts=max_attempts,
        initial_delay=0,
        max_delay=0,
        jitter=0,
    )
    if run_deadline_at is not None and monotonic_fn() >= run_deadline_at:
        raise BatchStageFailure(
            code="RUN_DEADLINE_EXCEEDED",
            stage=stage,
            message="单文件处理超过总时间预算",
            retryable=False,
        )

    cached = checkpoints.load(stage, producer_version)
    if cached is not None:
        return cached

    previous = checkpoints.store.get_stage_checkpoint(checkpoints.file_id, stage)
    if previous is not None and previous["status"] == "FAILED":
        previous_attempts = int(previous["attempts"] or 0)
        previous_retryable = bool(previous["retryable"])
        if not previous_retryable or previous_attempts >= max_attempts:
            raise BatchStageFailure(
                code=previous["error_code"] or "INTERNAL_ERROR",
                stage=stage,
                message=previous["error"] or "批量阶段失败",
                retryable=previous_retryable,
                attempts=previous_attempts,
            )

    checkpoints.invalidate_downstream(stage)
    while True:
        attempts = checkpoints.begin(stage)
        started = monotonic_fn()
        try:
            if run_deadline_at is not None and started >= run_deadline_at:
                raise BatchStageFailure(
                    code="RUN_DEADLINE_EXCEEDED",
                    stage=stage,
                    message="单文件处理超过总时间预算",
                    retryable=False,
                )
            value = action()
            finished = monotonic_fn()
            duration_ms = (finished - started) * 1000
            if run_deadline_at is not None and finished >= run_deadline_at:
                raise BatchStageFailure(
                    code="RUN_DEADLINE_EXCEEDED",
                    stage=stage,
                    message="单文件处理超过总时间预算",
                    retryable=False,
                )
            if (
                stage_timeout_seconds is not None
                and finished - started > stage_timeout_seconds
            ):
                raise BatchStageFailure(
                    code="STAGE_TIMEOUT",
                    stage=stage,
                    message="阶段执行超过时间预算",
                    retryable=True,
                )
            return checkpoints.complete(
                stage,
                value,
                producer_version,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (monotonic_fn() - started) * 1000
            classified = classify_error(exc, stage)
            failure = BatchStageFailure(
                code=classified.code,
                stage=classified.stage,
                message=classified.message,
                retryable=classified.retryable,
            )
            failure.attempts = attempts
            checkpoints.fail(
                stage,
                failure.code,
                failure.retryable,
                failure.message,
                duration_ms=duration_ms,
            )
            if retry_policy.should_retry(classified, attempts):
                delay = retry_policy.delay_for(attempts)
                if delay > 0:
                    time.sleep(delay)
                continue
            raise failure


def _validate_quality_result(analysis: AnalysisResult) -> AnalysisResult:
    status = analysis.status.value
    if status in {"COMPLETED", "PARTIAL"}:
        if analysis.report is None:
            raise BatchStageFailure(
                code="QC_REPORT_MISSING",
                stage=StageName.QC,
                message="质检结果缺少报告",
                retryable=False,
            )
        if status == "PARTIAL":
            analysis = analysis.model_copy(deep=True)
            analysis.report.disposition = ReviewDisposition.HUMAN_REVIEW_REQUIRED
        return analysis

    if status == "FAILED":
        errors = analysis.errors
        retryable = bool(errors) and all(error.retryable for error in errors)
        code = errors[0].code if errors else "QC_FAILED"
        message = errors[0].message if errors else "质检分析失败"
        raise BatchStageFailure(
            code=str(code),
            stage=StageName.QC,
            message=message,
            retryable=retryable,
        )

    raise BatchStageFailure(
        code="QC_INVALID_STATUS",
        stage=StageName.QC,
        message="质检结果状态非法",
        retryable=False,
    )


def process_file(
    file_record: FileRecord,
    audio_runner: AudioStageRunner,
    quality_service: Any,
    checkpoints: FileCheckpointSession,
    *,
    max_attempts: int = 3,
    retry_policy: RetryPolicy | None = None,
    stage_timeout_seconds: float | None = None,
    run_deadline_seconds: float | None = None,
    monotonic_fn=monotonic,
) -> FileResult:
    """Run or restore TRANSCODE, ASR, EMOTION and QC for one file."""

    request_json = ""
    max_attempts = max(1, int(max_attempts))
    run_deadline_at = (
        monotonic_fn() + float(run_deadline_seconds)
        if run_deadline_seconds is not None
        else None
    )

    def run_stage(stage: StageName, producer_version: str, action):
        return execute_stage(
            checkpoints,
            stage,
            producer_version,
            action,
            max_attempts=max_attempts,
            retry_policy=retry_policy,
            stage_timeout_seconds=stage_timeout_seconds,
            run_deadline_at=run_deadline_at,
            monotonic_fn=monotonic_fn,
        )

    try:
        wav_path = run_stage(
            StageName.TRANSCODE,
            audio_runner.producer_version(StageName.TRANSCODE),
            lambda: audio_runner.transcode(file_record),
        )

        def run_asr():
            turns = audio_runner.run_asr(wav_path)
            try:
                AnalysisRequest(
                    caseId=file_record.callId or file_record.idempotency_key,
                    callId=file_record.callId or file_record.idempotency_key,
                    transcript=turns,
                )
            except ValidationError as exc:
                raise BatchStageFailure(
                    code="MODEL_INVALID_OUTPUT",
                    stage=StageName.ASR,
                    message="ASR 输出结构非法",
                    retryable=False,
                ) from exc
            return turns

        turns = run_stage(
            StageName.ASR,
            audio_runner.producer_version(StageName.ASR),
            run_asr,
        )
        run_stage(
            StageName.EMOTION,
            audio_runner.producer_version(StageName.EMOTION),
            lambda: audio_runner.run_emotion(wav_path),
        )

        request = AnalysisRequest(
            caseId=file_record.callId or file_record.idempotency_key,
            callId=file_record.callId or file_record.idempotency_key,
            transcript=turns,
        )
        request_json = request.model_dump_json()

        def run_quality():
            try:
                analysis = AnalysisResult.model_validate(quality_service.analyze(request))
            except ValidationError as exc:
                raise BatchStageFailure(
                    code="MODEL_INVALID_OUTPUT",
                    stage=StageName.QC,
                    message="质检模型输出结构非法",
                    retryable=False,
                ) from exc
            return _validate_quality_result(analysis)

        analysis = run_stage(
            StageName.QC,
            "batch-qc-v1",
            run_quality,
        )
    except BatchStageFailure as exc:
        return FileResult(
            status=BatchFileStatus.FAILED_FINAL,
            request_json=request_json,
            result_json="",
            failed_reason=f"{exc.stage.value}/{exc.code}: {exc.message}",
        )

    report = analysis.report
    if analysis.status.value == "PARTIAL":
        final_status = BatchFileStatus.HUMAN_REVIEW
    elif report is not None and report.disposition.value == "HUMAN_REVIEW_REQUIRED":
        final_status = BatchFileStatus.HUMAN_REVIEW
    else:
        final_status = BatchFileStatus.DONE

    return FileResult(
        status=final_status,
        request_json=request_json,
        result_json=analysis.model_dump_json(),
        llm_request_count=1,
    )
