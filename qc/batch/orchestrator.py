from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from qc.batch.checkpoints import FileCheckpointSession
from qc.batch.models import BatchConfig, BatchFileStatus, BatchMeta, FileRecord, StageName
from qc.batch.pipeline import process_file
from qc.batch.store import BatchStore


class BatchOrchestrator:
    """Claim files, run the recoverable pipeline and atomically finalize results."""

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
        self.store.mark_interrupted_running(batch_id)
        return self._process(batch_id, audio_runner, quality_service, resume=True)

    def _throttle_llm(self) -> None:
        rpm = max(1, int(self.config.llm_rpm or 60))
        min_interval = 60.0 / rpm
        with self._llm_lock:
            now = time.monotonic()
            wait = self._last_llm_at + min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_llm_at = time.monotonic()

    def _process(self, batch_id, audio_runner, quality_service, resume: bool) -> dict:
        candidates = (
            self.store.resume_candidates(batch_id)
            if resume
            else [
                file_row
                for file_row in self.store.list_files(batch_id)
                if file_row["status"] in {"PENDING", "INTERRUPTED"}
            ]
        )
        gpu_sem = threading.Semaphore(max(1, self.config.gpu_workers))
        max_workers = max(1, self.config.cpu_workers)
        max_attempts = max(1, int(self.config.max_attempts or 3))

        def handle(file_row):
            file_id = file_row["file_id"]
            expected = BatchFileStatus(file_row["status"])
            if not self.store.claim_file(file_id, expected):
                return

            file_record = self._to_file_record(file_row)

            class GpuGuardedRunner:
                def __init__(self, inner, semaphore):
                    self.inner = inner
                    self.semaphore = semaphore

                def transcode(self, record):
                    return self.inner.transcode(record)

                def run_asr(self, wav_path):
                    with self.semaphore:
                        return self.inner.run_asr(wav_path)

                def run_emotion(self, wav_path):
                    with self.semaphore:
                        return self.inner.run_emotion(wav_path)

                def producer_version(self, stage: StageName) -> str:
                    return self.inner.producer_version(stage)

            class ThrottledQuality:
                def __init__(self, inner, throttle):
                    self.inner = inner
                    self.throttle = throttle

                def analyze(self, request):
                    self.throttle()
                    return self.inner.analyze(request)

            result = process_file(
                file_record,
                GpuGuardedRunner(audio_runner, gpu_sem),
                ThrottledQuality(quality_service, self._throttle_llm),
                FileCheckpointSession(self.store, batch_id, file_id),
                max_attempts=max_attempts,
            )
            self.store.finalize_file(
                file_id,
                result.status,
                result.request_json,
                result.result_json,
                failed_reason=result.failed_reason,
            )

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
