from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any, Protocol

from pydantic import BaseModel

from qc.batch.models import BatchFileStatus, FileRecord, StageName
from qc.models import AnalysisRequest, TranscriptTurn


class AudioStageRunner(Protocol):
    """音频阶段运行器接口。

    生产实现常驻 FunASR + emotion2vec；测试用 FakeAudioStageRunner。
    """

    def transcode(self, file_record: FileRecord) -> Path: ...

    def run_asr(self, wav_path: Path) -> list[TranscriptTurn]: ...

    def run_emotion(self, wav_path: Path) -> dict: ...


class FakeAudioStageRunner:
    """不依赖 GPU 的占位实现：产出 WAV 占位文件 + 固定转录 + 默认情绪。"""

    def __init__(self, wav_root: Path):
        self.wav_root = wav_root

    def transcode(self, file_record: FileRecord) -> Path:
        self.wav_root.mkdir(parents=True, exist_ok=True)
        wav_path = self.wav_root / f"{Path(file_record.source_uri).stem}.wav"
        wav_path.write_bytes(b"FAKEWAV")  # 占位
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


class FileResult(BaseModel):
    status: BatchFileStatus
    request_json: str
    result_json: str
    failed_reason: str | None = None
    llm_request_count: int = 0


# 阶段回调签名：(stage, status, duration_ms, error=None) -> None
StageCallback = Any


def process_file(
    file_record: FileRecord,
    audio_runner: AudioStageRunner,
    quality_service: Any,
    on_stage: StageCallback,
    skip_stages: set[str] | None = None,
) -> FileResult:
    """跑完单文件管线：转码 → ASR → 情绪 → 质检（复用 qc/）。

    skip_stages: 已成功阶段名集合（如 {"TRANSCODE","ASR"}），用于断点续跑跳过。
    跳过 ASR 时仍会调用 transcode 拿 wav 路径（若 TRANSCODE 也跳过则重做 transcode
    仅生成路径，Fake/真实 runner 应幂等）。
    """
    request_json = ""
    skip_stages = set(skip_stages or set())

    def run(stage: StageName, action):
        if stage.value in skip_stages:
            on_stage(stage, "DONE", 0.0)
            return action()
        started = monotonic()
        try:
            value = action()
            duration = (monotonic() - started) * 1000
            on_stage(stage, "DONE", duration)
            return value
        except Exception as exc:
            duration = (monotonic() - started) * 1000
            on_stage(stage, "FAILED", duration, error=str(exc))
            raise

    try:
        wav_path = run(StageName.TRANSCODE, lambda: audio_runner.transcode(file_record))
        turns = run(StageName.ASR, lambda: audio_runner.run_asr(wav_path))
        run(StageName.EMOTION, lambda: audio_runner.run_emotion(wav_path))
    except Exception as exc:
        # 转码/ASR/情绪失败：首版按不可恢复处理为死信（重试预算在 orchestrator 层）。
        return FileResult(
            status=BatchFileStatus.DEAD_LETTER,
            request_json=request_json,
            result_json="",
            failed_reason=str(exc),
        )

    request = AnalysisRequest(
        caseId=file_record.callId or file_record.idempotency_key,
        callId=file_record.callId or file_record.idempotency_key,
        transcript=turns,
    )
    request_json = request.model_dump_json()

    try:
        analysis = run(StageName.QC, lambda: quality_service.analyze(request))
    except Exception as exc:
        # 安全网：正常情况下 quality_service.analyze 不会抛错（接口失败内部降级）；
        # 一旦抛错说明出现未预期的降级路径，按死信隔离。
        return FileResult(
            status=BatchFileStatus.DEAD_LETTER,
            request_json=request_json,
            result_json="",
            failed_reason=str(exc),
        )

    report = analysis.report
    # quality_service 已保证 businessFact.status == NOT_CHECKED。
    disposition = report.disposition.value if report else "AUTO_PASS"
    final_status = (
        BatchFileStatus.HUMAN_REVIEW
        if disposition == "HUMAN_REVIEW_REQUIRED"
        else BatchFileStatus.DONE
    )
    result_json = analysis.model_dump_json()
    return FileResult(
        status=final_status,
        request_json=request_json,
        result_json=result_json,
        llm_request_count=1,
    )
