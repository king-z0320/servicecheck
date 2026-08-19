from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from qc.batch.models import (
    PIPELINE_STAGES,
    BatchFileStatus,
    BatchMeta,
    FileRecord,
    StageName,
    StageRecord,
    VALID_FILE_TRANSITIONS,
)
from qc.batch.store import StateConflictError
from qc.batch.service import BatchCapacityError, IdempotencyConflictError
from qc.database import create_database_engine, create_session_factory
from qc.orm_models import (
    BatchCreationRequestRow,
    BatchDeadLetterRow,
    BatchExportRow,
    BatchItemRow,
    BatchJobRow,
    OutboxEventRow,
    StageExecutionRow,
)
from qc.batch.outbox_publisher import OutboxEvent


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _json_value(value: str | dict | list | None):
    if value is None or isinstance(value, (dict, list)):
        return value
    if not value.strip():
        return None
    return json.loads(value)


class PostgresBatchStore:
    def __init__(self, database_url: str):
        self.engine = create_database_engine(database_url)
        self.session_factory = create_session_factory(self.engine)

    def create_batch(self, meta: BatchMeta) -> None:
        try:
            with self.session_factory.begin() as session:
                session.add(
                    BatchJobRow(
                        batch_id=meta.batch_id,
                        source=meta.source,
                        status="CREATED",
                        total=meta.total,
                        created_at=meta.created_at,
                        started_at=meta.created_at,
                    )
                )
        except IntegrityError as exc:
            raise ValueError(f"batch already exists: {meta.batch_id}") from exc

    def create_batch_if_absent(self, meta: BatchMeta) -> bool:
        with self.session_factory.begin() as session:
            result = session.execute(
                pg_insert(BatchJobRow)
                .values(
                    batch_id=meta.batch_id,
                    source=meta.source,
                    status="CREATED",
                    total=meta.total,
                    created_at=meta.created_at,
                    started_at=meta.created_at,
                )
                .on_conflict_do_nothing(index_elements=[BatchJobRow.batch_id])
                .returning(BatchJobRow.batch_id)
            )
            return result.scalar_one_or_none() is not None

    def create_batch_with_outbox(
        self,
        meta: BatchMeta,
        records: list[FileRecord],
        idempotency_key: str | None = None,
        *,
        request_hash: str | None = None,
        max_pending: int | None = None,
    ) -> dict:
        request_hash = request_hash or hashlib.sha256(
            meta.source.encode("utf-8")
        ).hexdigest()
        now = _utcnow()
        with self.session_factory.begin() as session:
            if idempotency_key:
                session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:lock_key))"),
                    {"lock_key": f"batch-idempotency:{idempotency_key}"},
                )
                existing = session.scalar(
                    select(BatchCreationRequestRow).where(
                        BatchCreationRequestRow.idempotency_key == idempotency_key
                    )
                )
                if existing is not None:
                    if existing.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            "Idempotency-Key was reused with a different request"
                        )
                    return self._batch_response_for_session(session, existing.batch_id)

            if max_pending is not None:
                session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:lock_key))"),
                    {"lock_key": "stage2-batch-pending-capacity"},
                )
                pending = int(
                    session.scalar(
                        select(func.count())
                        .select_from(BatchItemRow)
                        .where(
                            BatchItemRow.status.in_(
                                ["PENDING", "RUNNING", "INTERRUPTED"]
                            )
                        )
                    )
                    or 0
                )
                if pending + len(records) > max_pending:
                    raise BatchCapacityError(
                        "pending queue capacity would be exceeded: "
                        f"{pending}+{len(records)}>{max_pending}"
                    )

            batch = BatchJobRow(
                batch_id=meta.batch_id,
                source=meta.source,
                status="QUEUED",
                total=len(records),
                created_at=meta.created_at,
                started_at=None,
                source_snapshot={
                    "source_dir": meta.source,
                    "items": [record.model_dump(mode="json") for record in records],
                },
            )
            session.add(batch)
            session.flush()
            for record in records:
                item = BatchItemRow(
                    batch_id=meta.batch_id,
                    source_uri=record.source_uri,
                    source_sha256=record.metadata.get("sha256"),
                    source_size=record.metadata.get("size"),
                    call_id=record.callId,
                    idempotency_key=record.idempotency_key,
                    status="PENDING",
                    created_at=now,
                    updated_at=now,
                )
                session.add(item)
                session.flush()
                event_id = str(uuid4())
                payload = {
                    "schema_version": "batch-item-v1",
                    "event_id": event_id,
                    "event_type": "BATCH_ITEM_READY",
                    "batch_id": meta.batch_id,
                    "item_id": item.item_id,
                    "idempotency_key": record.idempotency_key,
                }
                session.add(
                    OutboxEventRow(
                        event_id=event_id,
                        event_type="BATCH_ITEM_READY",
                        aggregate_type="batch_item",
                        aggregate_id=str(item.item_id),
                        payload=payload,
                        status="PENDING",
                        attempts=0,
                        available_at=now,
                        created_at=now,
                        updated_at=now,
                    )
                )
            if idempotency_key:
                session.add(
                    BatchCreationRequestRow(
                        idempotency_key=idempotency_key,
                        batch_id=meta.batch_id,
                        request_hash=request_hash,
                        created_at=now,
                    )
                )
        return {
            "batch_id": meta.batch_id,
            "status": "QUEUED",
            "total": len(records),
        }

    @staticmethod
    def _items_for_session(session, batch_id: str):
        return session.scalars(
            select(BatchItemRow).where(BatchItemRow.batch_id == batch_id)
        ).all()

    @classmethod
    def _batch_response_for_session(cls, session, batch_id: str) -> dict:
        row = session.get(BatchJobRow, batch_id)
        if row is None:
            raise KeyError(batch_id)
        return {
            "batch_id": row.batch_id,
            "status": row.status,
            "total": len(cls._items_for_session(session, batch_id)),
        }

    def get_idempotent_batch(
        self,
        idempotency_key: str,
        request_hash: str,
    ) -> dict | None:
        with self.session_factory() as session:
            existing = session.scalar(
                select(BatchCreationRequestRow).where(
                    BatchCreationRequestRow.idempotency_key == idempotency_key
                )
            )
            if existing is None:
                return None
            if existing.request_hash != request_hash:
                raise IdempotencyConflictError(
                    "Idempotency-Key was reused with a different request"
                )
            return self._batch_response_for_session(session, existing.batch_id)

    def pending_item_count(self) -> int:
        with self.session_factory() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(BatchItemRow)
                    .where(
                        BatchItemRow.status.in_(
                            ["PENDING", "RUNNING", "INTERRUPTED"]
                        )
                    )
                )
                or 0
            )

    def pending_outbox_events(self, limit: int = 50) -> list[OutboxEvent]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(OutboxEventRow)
                .where(
                    OutboxEventRow.status == "PENDING",
                    OutboxEventRow.available_at <= _utcnow(),
                )
                .order_by(OutboxEventRow.created_at)
                .limit(limit)
            ).all()
            return [
                OutboxEvent(
                    event_id=row.event_id,
                    event_type=row.event_type,
                    batch_id=str(row.payload["batch_id"]),
                    item_id=int(row.payload["item_id"]),
                    idempotency_key=str(row.payload["idempotency_key"]),
                    created_at=row.created_at,
                    attempts=int(row.attempts or 0),
                )
                for row in rows
            ]

    def mark_outbox_published(self, event_id: str, redis_message_id: str) -> None:
        with self.session_factory.begin() as session:
            result = session.execute(
                update(OutboxEventRow)
                .where(OutboxEventRow.event_id == event_id, OutboxEventRow.status == "PENDING")
                .values(
                    status="PUBLISHED",
                    published_at=_utcnow(),
                    redis_message_id=redis_message_id,
                    updated_at=_utcnow(),
                )
            )
            if result.rowcount != 1:
                raise StateConflictError(f"outbox event {event_id} is not pending")

    def mark_outbox_failed(
        self,
        event_id: str,
        error: str,
        *,
        max_attempts: int = 5,
        retry_delay_seconds: float = 1.0,
    ) -> bool:
        with self.session_factory.begin() as session:
            row = session.scalar(
                select(OutboxEventRow)
                .where(OutboxEventRow.event_id == event_id)
                .with_for_update()
            )
            if row is None:
                raise KeyError(event_id)
            if row.status != "PENDING":
                raise StateConflictError(f"outbox event {event_id} is not pending")
            row.attempts = int(row.attempts or 0) + 1
            exhausted = row.attempts >= max(1, int(max_attempts))
            row.status = "FAILED" if exhausted else "PENDING"
            row.last_error = error[:500]
            row.available_at = _utcnow() + timedelta(
                seconds=max(0.0, float(retry_delay_seconds))
            )
            row.updated_at = _utcnow()
            return exhausted

    def record_dead_letter(
        self,
        *,
        batch_id: str,
        item_id: int,
        message_id: str | None,
        stage: str,
        error_code: str,
        attempts: int,
        last_error: str,
        reason: str,
    ) -> None:
        with self.session_factory.begin() as session:
            session.add(
                BatchDeadLetterRow(
                    batch_id=batch_id,
                    item_id=item_id,
                    message_id=message_id,
                    stage=stage,
                    error_code=error_code,
                    attempts=attempts,
                    last_error=last_error[:500],
                    reason=reason[:500],
                    created_at=_utcnow(),
                )
            )

    def add_file(self, batch_id: str, record: FileRecord) -> bool:
        now = _utcnow()
        with self.session_factory.begin() as session:
            result = session.execute(
                pg_insert(BatchItemRow)
                .values(
                    batch_id=batch_id,
                    source_uri=record.source_uri,
                    call_id=record.callId,
                    idempotency_key=record.idempotency_key,
                    status="PENDING",
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_nothing(
                    index_elements=[BatchItemRow.batch_id, BatchItemRow.idempotency_key]
                )
                .returning(BatchItemRow.item_id)
            )
            return result.scalar_one_or_none() is not None

    @staticmethod
    def _stage_payload(row: StageExecutionRow) -> dict:
        return {
            "id": row.stage_execution_id,
            "stage_execution_id": row.stage_execution_id,
            "file_id": row.item_id,
            "item_id": row.item_id,
            "stage": row.stage,
            "status": row.status,
            "started_at": _iso(row.started_at),
            "finished_at": _iso(row.finished_at),
            "duration_ms": row.duration_ms,
            "attempts": row.attempts,
            "artifact_uri": row.artifact_uri,
            "sha256": row.sha256,
            "producer_version": row.producer_version,
            "error_code": row.error_code,
            "retryable": row.retryable,
            "error": row.error_summary,
            "error_summary": row.error_summary,
        }

    @classmethod
    def _file_payload(cls, session, row: BatchItemRow) -> dict:
        stages = session.scalars(
            select(StageExecutionRow)
            .where(StageExecutionRow.item_id == row.item_id)
            .order_by(StageExecutionRow.stage_execution_id)
        ).all()
        return {
            "file_id": row.item_id,
            "item_id": row.item_id,
            "batch_id": row.batch_id,
            "source_uri": row.source_uri,
            "metadata": {
                key: value
                for key, value in {
                    "sha256": row.source_sha256,
                    "size": row.source_size,
                }.items()
                if value is not None
            },
            "call_id": row.call_id,
            "idempotency_key": row.idempotency_key,
            "status": row.status,
            "failed_reason": row.failed_reason,
            "request_json": (
                json.dumps(row.request_snapshot, ensure_ascii=False)
                if row.request_snapshot is not None
                else None
            ),
            "result_json": (
                json.dumps(row.result_snapshot, ensure_ascii=False)
                if row.result_snapshot is not None
                else None
            ),
            "stages": [cls._stage_payload(stage) for stage in stages],
        }

    def list_files(
        self,
        batch_id: str,
        statuses: list[str] | None = None,
    ) -> list[dict]:
        with self.session_factory() as session:
            statement = (
                select(BatchItemRow)
                .where(BatchItemRow.batch_id == batch_id)
                .order_by(BatchItemRow.item_id)
            )
            if statuses:
                statement = statement.where(BatchItemRow.status.in_(statuses))
            rows = session.scalars(statement).all()
            return [self._file_payload(session, row) for row in rows]

    def get_file(self, file_id: int) -> dict:
        with self.session_factory() as session:
            row = session.get(BatchItemRow, file_id)
            if row is None:
                raise KeyError(file_id)
            return self._file_payload(session, row)

    def record_stage(self, file_id: int, stage: StageRecord) -> None:
        with self.session_factory.begin() as session:
            session.add(
                StageExecutionRow(
                    item_id=file_id,
                    stage=stage.stage.value,
                    status=stage.status,
                    started_at=stage.started_at,
                    finished_at=stage.finished_at,
                    duration_ms=stage.duration_ms,
                    attempts=stage.attempts,
                    artifact_uri=stage.artifact_uri,
                    sha256=stage.sha256,
                    producer_version=stage.producer_version,
                    error_code=stage.error_code,
                    retryable=stage.retryable,
                    error_summary=stage.error,
                )
            )

    @staticmethod
    def _latest_stage(session, file_id: int, stage: StageName | str, *, lock=False):
        stage_value = stage.value if isinstance(stage, StageName) else str(stage)
        statement = (
            select(StageExecutionRow)
            .where(
                StageExecutionRow.item_id == file_id,
                StageExecutionRow.stage == stage_value,
            )
            .order_by(StageExecutionRow.stage_execution_id.desc())
            .limit(1)
        )
        if lock:
            statement = statement.with_for_update()
        return session.scalar(statement)

    def get_stage_checkpoint(self, file_id: int, stage: StageName | str) -> dict | None:
        with self.session_factory() as session:
            row = self._latest_stage(session, file_id, stage)
            return self._stage_payload(row) if row is not None else None

    def begin_stage(self, file_id: int, stage: StageName) -> int:
        with self.session_factory.begin() as session:
            if session.get(BatchItemRow, file_id) is None:
                raise KeyError(file_id)
            current = self._latest_stage(session, file_id, stage, lock=True)
            attempts = (current.attempts if current is not None else 0) + 1
            session.add(
                StageExecutionRow(
                    item_id=file_id,
                    stage=stage.value,
                    status="RUNNING",
                    started_at=_utcnow(),
                    duration_ms=0,
                    attempts=attempts,
                )
            )
            return attempts

    def complete_stage(
        self,
        file_id: int,
        stage: StageName,
        *,
        artifact_uri: str | Path,
        sha256: str,
        producer_version: str,
        duration_ms: float,
    ) -> None:
        with self.session_factory.begin() as session:
            current = self._latest_stage(session, file_id, stage, lock=True)
            if current is None or current.status != "RUNNING":
                raise StateConflictError(f"stage {stage.value} is not running")
            current.status = "DONE"
            current.finished_at = _utcnow()
            current.duration_ms = duration_ms
            current.artifact_uri = str(artifact_uri)
            current.sha256 = sha256
            current.producer_version = producer_version
            current.error_code = None
            current.retryable = None
            current.error_summary = None

    def fail_stage(
        self,
        file_id: int,
        stage: StageName,
        *,
        error_code: str,
        retryable: bool,
        error: str,
        duration_ms: float,
    ) -> None:
        with self.session_factory.begin() as session:
            current = self._latest_stage(session, file_id, stage, lock=True)
            if current is None or current.status != "RUNNING":
                raise StateConflictError(f"stage {stage.value} is not running")
            current.status = "FAILED"
            current.finished_at = _utcnow()
            current.duration_ms = duration_ms
            current.error_code = error_code
            current.retryable = retryable
            current.error_summary = error

    def invalidate_stages(self, file_id: int, stages: list[StageName]) -> None:
        with self.session_factory.begin() as session:
            for stage in stages:
                row = self._latest_stage(session, file_id, stage, lock=True)
                if row is not None:
                    row.status = "PENDING"
                    row.finished_at = None
                    row.artifact_uri = None
                    row.sha256 = None
                    row.producer_version = None
                    row.error_code = None
                    row.retryable = None
                    row.error_summary = None

    @staticmethod
    def _status_value(status: BatchFileStatus | str) -> BatchFileStatus:
        try:
            return status if isinstance(status, BatchFileStatus) else BatchFileStatus(str(status))
        except ValueError as exc:
            raise ValueError(f"unknown batch file status: {status!r}") from exc

    def claim_file(self, file_id: int, expected_status: BatchFileStatus | str) -> bool:
        expected = self._status_value(expected_status)
        if expected not in {BatchFileStatus.PENDING, BatchFileStatus.INTERRUPTED}:
            return False
        with self.session_factory.begin() as session:
            result = session.execute(
                update(BatchItemRow)
                .where(
                    BatchItemRow.item_id == file_id,
                    BatchItemRow.status == expected.value,
                )
                .values(status="RUNNING", failed_reason=None, updated_at=_utcnow())
            )
            claimed = result.rowcount == 1
            if claimed:
                batch_id = session.scalar(
                    select(BatchItemRow.batch_id).where(BatchItemRow.item_id == file_id)
                )
                session.execute(
                    update(BatchJobRow)
                    .where(BatchJobRow.batch_id == batch_id)
                    .values(status="RUNNING", started_at=func.coalesce(BatchJobRow.started_at, _utcnow()))
                )
            return claimed

    def set_file_status(
        self,
        file_id: int,
        status: BatchFileStatus | str,
        failed_reason: str | None = None,
    ) -> None:
        target = self._status_value(status)
        if target in {
            BatchFileStatus.DONE,
            BatchFileStatus.HUMAN_REVIEW,
            BatchFileStatus.FAILED_FINAL,
            BatchFileStatus.DEAD_LETTER,
        }:
            raise StateConflictError("terminal states must be written with finalize_file")
        with self.session_factory.begin() as session:
            row = session.get(BatchItemRow, file_id)
            if row is None:
                raise KeyError(file_id)
            current = BatchFileStatus(row.status)
            if target not in VALID_FILE_TRANSITIONS[current]:
                raise StateConflictError(
                    f"illegal file transition: {current.value} -> {target.value}"
                )
            result = session.execute(
                update(BatchItemRow)
                .where(
                    BatchItemRow.item_id == file_id,
                    BatchItemRow.status == current.value,
                )
                .values(
                    status=target.value,
                    failed_reason=failed_reason,
                    updated_at=_utcnow(),
                )
            )
            if result.rowcount != 1:
                raise StateConflictError(f"file {file_id} state changed concurrently")

    def finalize_file(
        self,
        file_id: int,
        status: BatchFileStatus | str,
        request_json: str,
        result_json: str,
        failed_reason: str | None = None,
    ) -> None:
        target = self._status_value(status)
        if target not in {
            BatchFileStatus.DONE,
            BatchFileStatus.HUMAN_REVIEW,
            BatchFileStatus.FAILED_FINAL,
        }:
            raise ValueError(f"not a writable final status: {target.value}")
        request_snapshot = _json_value(request_json)
        result_snapshot = _json_value(result_json)
        with self.session_factory.begin() as session:
            self._finalize_file_in_session(
                session,
                file_id,
                target,
                request_snapshot,
                result_snapshot,
                failed_reason,
            )

    def _finalize_file_in_session(
        self,
        session,
        file_id: int,
        target: BatchFileStatus,
        request_snapshot,
        result_snapshot,
        failed_reason: str | None,
    ) -> str:
        result = session.execute(
            update(BatchItemRow)
            .where(
                BatchItemRow.item_id == file_id,
                BatchItemRow.status == "RUNNING",
            )
            .values(
                status=target.value,
                request_snapshot=request_snapshot,
                result_snapshot=result_snapshot,
                failed_reason=failed_reason,
                updated_at=_utcnow(),
            )
        )
        if result.rowcount != 1:
            raise StateConflictError(f"file {file_id} is not RUNNING")
        batch_id = session.scalar(
            select(BatchItemRow.batch_id).where(BatchItemRow.item_id == file_id)
        )
        statuses = session.execute(
            select(BatchItemRow.status, func.count())
            .where(BatchItemRow.batch_id == batch_id)
            .group_by(BatchItemRow.status)
        ).all()
        counts = {status: int(count) for status, count in statuses}
        terminal = {"DONE", "HUMAN_REVIEW", "FAILED_FINAL"}
        if counts and all(status in terminal for status in counts):
            if counts.get("FAILED_FINAL"):
                batch_status = "FAILED"
            elif counts.get("HUMAN_REVIEW"):
                batch_status = "PARTIAL"
            else:
                batch_status = "COMPLETED"
            session.execute(
                update(BatchJobRow)
                .where(BatchJobRow.batch_id == batch_id)
                .values(status=batch_status, finished_at=_utcnow())
            )
        return str(batch_id)

    def finalize_file_with_dead_letter(
        self,
        file_id: int,
        *,
        request_json: str,
        result_json: str,
        failed_reason: str | None,
        message_id: str | None,
        stage: str,
        error_code: str,
        attempts: int,
        last_error: str,
        reason: str,
    ) -> None:
        with self.session_factory.begin() as session:
            batch_id = self._finalize_file_in_session(
                session,
                file_id,
                BatchFileStatus.FAILED_FINAL,
                _json_value(request_json),
                _json_value(result_json),
                failed_reason,
            )
            session.add(
                BatchDeadLetterRow(
                    batch_id=batch_id,
                    item_id=file_id,
                    message_id=message_id,
                    stage=stage,
                    error_code=error_code,
                    attempts=attempts,
                    last_error=last_error[:500],
                    reason=reason[:500],
                    created_at=_utcnow(),
                )
            )

    def batch_summary(self, batch_id: str) -> dict:
        with self.session_factory() as session:
            rows = session.execute(
                select(BatchItemRow.status, func.count())
                .where(BatchItemRow.batch_id == batch_id)
                .group_by(BatchItemRow.status)
            ).all()
            counts = {status: int(count) for status, count in rows}
            return {"batch_id": batch_id, "total": sum(counts.values()), "by_status": counts}

    def batch_started_at(self, batch_id: str) -> str | None:
        with self.session_factory() as session:
            row = session.get(BatchJobRow, batch_id)
            return _iso(row.started_at) if row else None

    def batch_durations(self, batch_id: str) -> dict[str, float]:
        with self.session_factory() as session:
            rows = session.execute(
                select(StageExecutionRow.stage, func.avg(StageExecutionRow.duration_ms))
                .join(BatchItemRow, BatchItemRow.item_id == StageExecutionRow.item_id)
                .where(
                    BatchItemRow.batch_id == batch_id,
                    StageExecutionRow.status == "DONE",
                )
                .group_by(StageExecutionRow.stage)
            ).all()
            return {stage: float(average) for stage, average in rows if average is not None}

    def resume_candidates(self, batch_id: str) -> list[dict]:
        return self.list_files(batch_id, ["PENDING", "INTERRUPTED"])

    def mark_interrupted_running(self, batch_id: str) -> int:
        with self.session_factory.begin() as session:
            result = session.execute(
                update(BatchItemRow)
                .where(
                    BatchItemRow.batch_id == batch_id,
                    BatchItemRow.status == "RUNNING",
                )
                .values(status="INTERRUPTED", updated_at=_utcnow())
            )
            return int(result.rowcount or 0)

    def last_completed_stage(self, file_id: int) -> StageName | None:
        with self.session_factory() as session:
            names = set(
                session.scalars(
                    select(StageExecutionRow.stage).where(
                        StageExecutionRow.item_id == file_id,
                        StageExecutionRow.status == "DONE",
                    )
                ).all()
            )
        for stage in reversed(PIPELINE_STAGES):
            if stage.value in names:
                return stage
        return None

    @staticmethod
    def _artifact_uri(value: str | Path) -> str:
        return str(value)

    def begin_export(
        self,
        batch_id: str,
        format: str,
        artifact_uri: str | Path,
        producer_version: str,
    ) -> None:
        uri = self._artifact_uri(artifact_uri)
        now = _utcnow()
        with self.session_factory.begin() as session:
            session.execute(
                pg_insert(BatchExportRow)
                .values(
                    batch_id=batch_id,
                    format=format,
                    artifact_uri=uri,
                    status="RUNNING",
                    producer_version=producer_version,
                    created_at=now,
                )
                .on_conflict_do_update(
                    constraint="uq_batch_exports_identity",
                    set_={
                        "status": "RUNNING",
                        "sha256": None,
                        "producer_version": producer_version,
                        "error_code": None,
                        "finished_at": None,
                    },
                )
            )

    def complete_export(
        self,
        batch_id: str,
        format: str,
        artifact_uri: str | Path,
        producer_version: str,
        sha256: str,
    ) -> None:
        uri = self._artifact_uri(artifact_uri)
        with self.session_factory.begin() as session:
            result = session.execute(
                update(BatchExportRow)
                .where(
                    BatchExportRow.batch_id == batch_id,
                    BatchExportRow.format == format,
                    BatchExportRow.artifact_uri == uri,
                    BatchExportRow.status == "RUNNING",
                )
                .values(
                    status="DONE",
                    sha256=sha256,
                    producer_version=producer_version,
                    error_code=None,
                    finished_at=_utcnow(),
                )
            )
            if result.rowcount != 1:
                raise StateConflictError("export record is not RUNNING")

    def fail_export(
        self,
        batch_id: str,
        format: str,
        artifact_uri: str | Path,
        producer_version: str,
        error_code: str,
    ) -> None:
        uri = self._artifact_uri(artifact_uri)
        with self.session_factory.begin() as session:
            session.execute(
                pg_insert(BatchExportRow)
                .values(
                    batch_id=batch_id,
                    format=format,
                    artifact_uri=uri,
                    status="FAILED",
                    producer_version=producer_version,
                    error_code=error_code,
                    created_at=_utcnow(),
                    finished_at=_utcnow(),
                )
                .on_conflict_do_update(
                    constraint="uq_batch_exports_identity",
                    set_={
                        "status": "FAILED",
                        "sha256": None,
                        "producer_version": producer_version,
                        "error_code": error_code,
                        "finished_at": _utcnow(),
                    },
                )
            )

    def get_export_record(
        self,
        batch_id: str,
        format: str,
        artifact_uri: str | Path,
    ) -> dict | None:
        uri = self._artifact_uri(artifact_uri)
        with self.session_factory() as session:
            row = session.scalar(
                select(BatchExportRow).where(
                    BatchExportRow.batch_id == batch_id,
                    BatchExportRow.format == format,
                    BatchExportRow.artifact_uri == uri,
                )
            )
            if row is None:
                return None
            return {
                "export_id": row.export_id,
                "batch_id": row.batch_id,
                "format": row.format,
                "artifact_uri": row.artifact_uri,
                "status": row.status,
                "sha256": row.sha256,
                "producer_version": row.producer_version,
                "error_code": row.error_code,
            }

    def close(self) -> None:
        self.engine.dispose()
