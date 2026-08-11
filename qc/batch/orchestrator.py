from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from qc.batch.models import (
    BatchConfig,
    BatchFileStatus,
    BatchMeta,
    FileRecord,
    StageName,
    StageRecord,
)
from qc.batch.pipeline import FileResult, process_file
from qc.batch.store import BatchStore


class BatchOrchestrator:
    """编排器：CPU 线程池跑整文件，GPU 信号量只保护 ASR/情绪阶段。

    - max_attempts：可恢复失败（非明确文件损坏）时重试
    - llm_rpm：质检阶段简易限流（最小间隔）
    - 阶段续跑：若 store 显示 TRANSCODE/ASR/EMOTION 已 DONE，则跳过
      （由 process_file 的 skip_stages 参数实现）
    """

    def __init__(self, store: BatchStore, config: BatchConfig):
        self.store = store
        self.config = config
        self._llm_lock = threading.Lock()
        self._last_llm_at = 0.0

    def ingest(self, batch_id: str, source, meta: BatchMeta) -> int:
        self.store.create_batch_if_absent(meta)
        added = 0
        for record in source.discover():
            if self.store.add_file(batch_id, record):
                added += 1
        return added

    def run_batch(self, batch_id, audio_runner, quality_service) -> dict:
        return self._process(batch_id, audio_runner, quality_service, resume=False)

    def resume(self, batch_id, audio_runner, quality_service) -> dict:
        self.store.mark_interrupted_running()
        return self._process(batch_id, audio_runner, quality_service, resume=True)

    def _throttle_llm(self) -> None:
        """按 llm_rpm 做最小间隔限流。"""
        rpm = max(1, int(self.config.llm_rpm or 60))
        min_interval = 60.0 / rpm
        with self._llm_lock:
            now = time.monotonic()
            wait = self._last_llm_at + min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_llm_at = time.monotonic()

    def _completed_stages(self, file_id: int) -> set[str]:
        try:
            row = self.store.get_file(file_id)
        except KeyError:
            return set()
        done = set()
        for stage in row.get("stages") or []:
            if stage.get("status") == "DONE":
                done.add(stage.get("stage"))
        return done

    def _process(self, batch_id, audio_runner, quality_service, resume: bool) -> dict:
        candidates = (
            self.store.resume_candidates(batch_id)
            if resume
            else [
                f
                for f in self.store.list_files(batch_id)
                if f["status"] in {"PENDING", "INTERRUPTED"}
            ]
        )
        # GPU 只保护 ASR/情绪；转码与 LLM 不占 GPU 锁
        gpu_sem = threading.Semaphore(max(1, self.config.gpu_workers))
        max_workers = max(1, self.config.cpu_workers)
        max_attempts = max(1, int(self.config.max_attempts or 3))

        def handle(file_row):
            file_id = file_row["file_id"]
            self.store.set_file_status(file_id, BatchFileStatus.RUNNING)
            file_record = self._to_file_record(file_row)
            skip = self._completed_stages(file_id) if resume else set()

            def on_stage(stage: StageName, status: str, duration_ms, error=None):
                self.store.record_stage(
                    file_id,
                    StageRecord(
                        stage=stage,
                        status=status,
                        duration_ms=duration_ms,
                        error=error,
                    ),
                )

            class GpuGuardedRunner:
                """ASR/情绪获取 GPU 信号量；转码不占用。"""

                def __init__(self, inner, sem):
                    self.inner = inner
                    self.sem = sem

                def transcode(self, record):
                    return self.inner.transcode(record)

                def run_asr(self, wav_path):
                    with self.sem:
                        return self.inner.run_asr(wav_path)

                def run_emotion(self, wav_path):
                    with self.sem:
                        return self.inner.run_emotion(wav_path)

            class ThrottledQuality:
                def __init__(self, inner, throttle):
                    self.inner = inner
                    self.throttle = throttle

                def analyze(self, request):
                    self.throttle()
                    return self.inner.analyze(request)

            guarded_audio = GpuGuardedRunner(audio_runner, gpu_sem)
            throttled_quality = ThrottledQuality(quality_service, self._throttle_llm)

            last_result = None
            for attempt in range(1, max_attempts + 1):
                last_result = process_file(
                    file_record,
                    guarded_audio,
                    throttled_quality,
                    on_stage,
                    skip_stages=skip,
                )
                if last_result.status != BatchFileStatus.DEAD_LETTER:
                    break
                reason = (last_result.failed_reason or "").lower()
                # 明确损坏不重试
                if any(
                    k in reason
                    for k in ("损坏", "corrupt", "not found", "不存在", "format")
                ):
                    break
                self.store.increment_stage_attempts(file_id, StageName.QC)
                time.sleep(min(2 ** (attempt - 1) * 0.01, 0.5))  # 测试友好的短退避
                skip = self._completed_stages(file_id)

            assert last_result is not None
            self.store.save_file_report(
                file_id,
                last_result.status,
                last_result.request_json,
                last_result.result_json,
            )
            if last_result.failed_reason:
                self.store.set_file_status(
                    file_id,
                    last_result.status,
                    failed_reason=last_result.failed_reason,
                )
            else:
                self.store.set_file_status(file_id, last_result.status)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(handle, candidates))

        return self.store.batch_summary(batch_id)

    @staticmethod
    def _to_file_record(file_row: dict) -> FileRecord:
        return FileRecord(
            source_uri=file_row["source_uri"],
            idempotency_key=file_row["idempotency_key"],
            callId=file_row["call_id"],
        )
