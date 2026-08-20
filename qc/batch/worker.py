from __future__ import annotations

import json
import logging
import signal
from dataclasses import dataclass
import threading
import time
from pathlib import Path
from typing import Any, Callable

from qc.batch.models import BatchFileStatus
from qc.batch.checkpoints import FileCheckpointSession
from qc.batch.models import FileRecord
from qc.batch.pipeline import process_file
from qc.batch.retry_policy import RetryPolicy

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BatchMessage:
    message_id: str
    batch_id: str
    item_id: int
    idempotency_key: str | None = None

    @classmethod
    def from_mapping(cls, message_id: str, payload: dict[str, Any]) -> "BatchMessage":
        return cls(
            message_id=str(message_id),
            batch_id=str(payload["batch_id"]),
            item_id=int(payload["item_id"]),
            idempotency_key=payload.get("idempotency_key"),
        )


class BatchWorker:
    """One Consumer that commits business state before acknowledging Redis."""

    def __init__(
        self,
        store,
        redis_client,
        executor: Callable[[dict], Any],
        *,
        stream: str = "qc:batch-items:v1",
        group: str = "qc-workers",
        consumer: str = "worker-1",
        claim_idle_ms: int = 300_000,
    ):
        self.store = store
        self.redis = redis_client
        self.executor = executor
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.claim_idle_ms = claim_idle_ms
        self._stopping = False

    def handle_message(self, message: BatchMessage, *, reclaimed: bool = False) -> bool:
        row = self.store.get_file(message.item_id)
        expected = row.get("status")
        if expected == BatchFileStatus.RUNNING.value and not reclaimed:
            # A normal duplicate must remain pending/owned by the active
            # consumer. ACK here could lose the original work.
            LOGGER.info(
                "batch message still owned by active execution",
                extra={"batch_id": message.batch_id, "item_id": message.item_id},
            )
            return False
        if expected not in {BatchFileStatus.PENDING.value, BatchFileStatus.INTERRUPTED.value}:
            if expected != BatchFileStatus.RUNNING.value:
                self.redis.xack(self.stream, self.group, message.message_id)
            return False
        if not self.store.claim_file(message.item_id, expected):
            latest = self.store.get_file(message.item_id)
            if latest.get("status") in {
                BatchFileStatus.DONE.value,
                BatchFileStatus.HUMAN_REVIEW.value,
                BatchFileStatus.FAILED_FINAL.value,
            }:
                self.redis.xack(self.stream, self.group, message.message_id)
                return False
            return False

        try:
            result = self.executor(row)
            if result.status == BatchFileStatus.FAILED_FINAL:
                latest = self.store.get_file(message.item_id)
                failures = [
                    stage for stage in latest.get("stages", [])
                    if stage.get("status") == "FAILED"
                ]
                failure = failures[-1] if failures else {}
                dead_letter = {
                    "message_id": message.message_id,
                    "stage": str(failure.get("stage") or "UNKNOWN"),
                    "error_code": str(failure.get("error_code") or "FAILED_FINAL"),
                    "attempts": int(failure.get("attempts") or 0),
                    "last_error": str(
                        failure.get("error") or result.failed_reason or "failed"
                    ),
                    "reason": "retry exhausted or non-retryable failure",
                }
                if hasattr(self.store, "finalize_file_with_dead_letter"):
                    self.store.finalize_file_with_dead_letter(
                        message.item_id,
                        request_json=result.request_json,
                        result_json=result.result_json,
                        failed_reason=result.failed_reason,
                        **dead_letter,
                    )
                else:
                    self.store.finalize_file(
                        message.item_id,
                        result.status,
                        result.request_json,
                        result.result_json,
                        failed_reason=result.failed_reason,
                    )
                    if hasattr(self.store, "record_dead_letter"):
                        self.store.record_dead_letter(
                            batch_id=message.batch_id,
                            item_id=message.item_id,
                            **dead_letter,
                        )
            else:
                self.store.finalize_file(
                    message.item_id,
                    result.status,
                    result.request_json,
                    result.result_json,
                    failed_reason=result.failed_reason,
                )
        except Exception:
            LOGGER.exception(
                "batch worker execution failed",
                extra={"batch_id": message.batch_id, "item_id": message.item_id},
            )
            # Leave the message pending and make it reclaimable. A later
            # XAUTOCLAIM can conditionally claim the INTERRUPTED item.
            try:
                self.store.set_file_status(
                    message.item_id,
                    BatchFileStatus.INTERRUPTED,
                    failed_reason="WORKER_EXECUTION_INTERRUPTED",
                )
            except Exception:
                LOGGER.exception("failed to mark interrupted batch item")
            return False

        self.redis.xack(self.stream, self.group, message.message_id)
        return True

    def _parse_entries(self, entries) -> list[BatchMessage]:
        messages: list[BatchMessage] = []
        for _stream_name, stream_entries in entries or []:
            for message_id, payload in stream_entries:
                decoded = {
                    key.decode() if isinstance(key, bytes) else key:
                    value.decode() if isinstance(value, bytes) else value
                    for key, value in payload.items()
                }
                messages.append(BatchMessage.from_mapping(str(message_id), decoded))
        return messages

    def run_once(self, *, block_ms: int = 1000, count: int = 1) -> int:
        entries = self.redis.xreadgroup(
            self.group,
            self.consumer,
            {self.stream: ">"},
            count=count,
            block=block_ms,
        )
        processed = 0
        for message in self._parse_entries(entries):
            processed += int(self.handle_message(message))
        return processed

    def reclaim_once(self, *, count: int = 10) -> int:
        claimed = self.redis.xautoclaim(
            self.stream,
            self.group,
            self.consumer,
            min_idle_time=self.claim_idle_ms,
            start_id="0-0",
            count=count,
        )
        entries = claimed[1] if isinstance(claimed, (tuple, list)) and len(claimed) > 1 else []
        processed = 0
        for message in self._parse_entries([(self.stream, entries)]):
            row = self.store.get_file(message.item_id)
            if row.get("status") == BatchFileStatus.RUNNING.value:
                try:
                    self.store.set_file_status(
                        message.item_id,
                        BatchFileStatus.INTERRUPTED,
                        failed_reason="WORKER_LEASE_EXPIRED",
                    )
                except Exception:
                    LOGGER.exception(
                        "failed to mark reclaimed item interrupted",
                        extra={"batch_id": message.batch_id, "item_id": message.item_id},
                    )
                    continue
            processed += int(self.handle_message(message, reclaimed=True))
        return processed

    def stop(self, *_args):
        self._stopping = True

    def run_forever(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        while not self._stopping:
            self.reclaim_once()
            self.run_once()


class BatchItemExecutor:
    """Execute one already-claimed batch item with bounded shared resources."""

    def __init__(self, store, artifact_store, audio_runner, quality_service, config):
        self.store = store
        self.artifact_store = artifact_store
        self.audio_runner = audio_runner
        self.quality_service = quality_service
        self.config = config
        self._gpu = threading.Semaphore(max(1, config.gpu_workers))
        self._llm_lock = threading.Lock()
        self._last_llm_at = 0.0

    def _throttle_llm(self):
        minimum = 60.0 / max(1, int(self.config.llm_rpm or 60))
        with self._llm_lock:
            wait = self._last_llm_at + minimum - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_llm_at = time.monotonic()

    def __call__(self, row: dict):
        executor = self

        class GuardedAudio:
            def transcode(self, record):
                return executor.audio_runner.transcode(record)

            def run_asr(self, wav_path):
                with executor._gpu:
                    return executor.audio_runner.run_asr(wav_path)

            def run_emotion(self, wav_path):
                with executor._gpu:
                    return executor.audio_runner.run_emotion(wav_path)

            def producer_version(self, stage):
                return executor.audio_runner.producer_version(stage)

        class ThrottledQuality:
            def analyze(self, request):
                executor._throttle_llm()
                return executor.quality_service.analyze(request)

        record = FileRecord(
            source_uri=row["source_uri"],
            idempotency_key=row["idempotency_key"],
            callId=row.get("call_id"),
            metadata=row.get("metadata") or {},
        )
        return process_file(
            record,
            GuardedAudio(),
            ThrottledQuality(),
            FileCheckpointSession(
                self.store,
                row["batch_id"],
                row.get("item_id", row["file_id"]),
                artifact_store=self.artifact_store,
            ),
            max_attempts=self.config.max_attempts,
            retry_policy=RetryPolicy(
                max_attempts=self.config.max_attempts,
                initial_delay=self.config.backoff_initial,
                max_delay=self.config.backoff_max,
                jitter=self.config.retry_jitter,
            ),
            stage_timeout_seconds=self.config.stage_timeout_seconds,
            run_deadline_seconds=self.config.run_deadline_seconds,
        )


def decode_json_field(value: str | bytes) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("message payload must be a JSON object")
    return decoded


def build_redis_client():
    import os
    import redis

    return redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
        decode_responses=True,
    )


def ensure_consumer_group(client, stream: str, group: str) -> None:
    try:
        client.xgroup_create(stream, group, id="0-0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def main() -> int:
    import os

    from api_server import build_artifact_store, build_service
    from qc.batch.models import BatchConfig
    from qc.batch.postgres_store import PostgresBatchStore
    from qc.batch.real_audio_runner import RealAudioStageRunner
    from qc.database import database_url_from_env

    project_root = Path(__file__).resolve().parents[2]
    audio_root = Path(os.getenv("BATCH_AUDIO_ROOT", project_root / "audio"))
    work_root = project_root / ".runtime" / "tmp" / "batch-worker"
    store = PostgresBatchStore(database_url_from_env())
    artifacts = build_artifact_store()
    audio_runner = RealAudioStageRunner(audio_root, work_root)
    try:
        audio_runner.warmup()
        executor = BatchItemExecutor(
            store,
            artifacts,
            audio_runner,
            build_service(),
            BatchConfig(),
        )
        redis_client = build_redis_client()
        worker = BatchWorker(store, redis_client, executor)
        ensure_consumer_group(redis_client, worker.stream, worker.group)
        worker.run_forever()
    finally:
        audio_runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
