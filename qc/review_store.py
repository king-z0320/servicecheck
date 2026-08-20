from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import Select, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from qc.database import create_database_engine, create_session_factory
from qc.orm_models import (
    AgentTraceEventRow,
    CallRow,
    QCReportRow,
    QCRunRow,
    ReviewRevisionRow,
    ReviewTaskRow,
)
from qc.review_models import HumanOutcome, ReviewTaskStatus
from qc.review_service import (
    ReviewIdempotencyConflict,
    ReviewNotFound,
    ReviewStateConflict,
    ReviewVersionConflict,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _route_reasons_payload(reasons: list[Any]) -> list[dict[str, Any]]:
    payload = []
    for item in reasons:
        if hasattr(item, "model_dump"):
            payload.append(item.model_dump(mode="json"))
        else:
            payload.append(dict(item))
    return payload


def ensure_review_task_in_session(
    session,
    run_id: str,
    route_reasons: list[Any],
) -> str:
    now = _utcnow()
    task_id = f"REVIEW-{uuid4().hex[:12].upper()}"
    result = session.execute(
        pg_insert(ReviewTaskRow)
        .values(
            review_task_id=task_id,
            run_id=run_id,
            batch_item_id=None,
            status=ReviewTaskStatus.PENDING.value,
            route_reasons=_route_reasons_payload(route_reasons),
            version=1,
            effective_revision_id=None,
            unresolved_reason=None,
            created_at=now,
            updated_at=now,
        )
        .on_conflict_do_nothing(index_elements=[ReviewTaskRow.run_id])
        .returning(ReviewTaskRow.review_task_id)
    )
    created = result.scalar_one_or_none()
    if created is not None:
        return created
    existing = session.scalar(
        select(ReviewTaskRow.review_task_id).where(ReviewTaskRow.run_id == run_id)
    )
    if existing is None:
        raise RuntimeError(f"review task missing after insert conflict: {run_id}")
    return existing


def attach_batch_item_in_session(session, run_id: str, batch_item_id: int) -> None:
    session.execute(
        update(ReviewTaskRow)
        .where(ReviewTaskRow.run_id == run_id)
        .values(batch_item_id=batch_item_id, updated_at=_utcnow())
    )


class PostgresReviewStore:
    def __init__(self, database_url: str):
        self.engine = create_database_engine(database_url)
        self.session_factory = create_session_factory(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def list_tasks(
        self,
        status: str | None,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        with self.session_factory() as session:
            count_statement = select(func.count()).select_from(ReviewTaskRow)
            if status:
                count_statement = count_statement.where(ReviewTaskRow.status == status)
            total = session.scalar(count_statement)
            statement: Select = (
                select(ReviewTaskRow, QCRunRow, QCReportRow)
                .join(QCRunRow, QCRunRow.run_id == ReviewTaskRow.run_id)
                .outerjoin(QCReportRow, QCReportRow.run_id == ReviewTaskRow.run_id)
            )
            if status:
                statement = statement.where(ReviewTaskRow.status == status)
            rows = session.execute(
                statement.order_by(
                    ReviewTaskRow.created_at.desc(),
                    ReviewTaskRow.review_task_id.desc(),
                )
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            items = [
                self._task_summary(task, run, report)
                for task, run, report in rows
            ]
        return {
            "items": items,
            "page": page,
            "pageSize": page_size,
            "total": int(total or 0),
        }

    def get_task_detail(self, review_task_id: str) -> dict[str, Any] | None:
        with self.session_factory() as session:
            task = session.get(ReviewTaskRow, review_task_id)
            if task is None:
                return None
            return self._task_detail(session, task)

    def summaries_for_run(self, run_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            return self.summaries_for_run_in_session(session, run_id)

    def summaries_for_runs(self, run_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not run_ids:
            return {}
        with self.session_factory() as session:
            tasks = session.scalars(
                select(ReviewTaskRow).where(ReviewTaskRow.run_id.in_(run_ids))
            ).all()
            return {
                task.run_id: self._bundle(session, task)
                for task in tasks
            }

    def summaries_for_run_in_session(self, session, run_id: str) -> dict[str, Any]:
        task = session.scalar(select(ReviewTaskRow).where(ReviewTaskRow.run_id == run_id))
        if task is None:
            return {
                "reviewTask": None,
                "effectiveRevision": None,
                "reviewHistory": [],
            }
        return self._bundle(session, task)

    def submit(
        self,
        *,
        review_task_id: str,
        expected_version: int,
        outcome: str,
        reason_code: str,
        note: str,
        idempotency_key: str,
        request_hash: str,
        reviewer_id: str,
        context_source: str,
    ) -> dict[str, Any]:
        try:
            with self.session_factory.begin() as session:
                existing = session.scalar(
                    select(ReviewRevisionRow).where(
                        ReviewRevisionRow.task_id == review_task_id,
                        ReviewRevisionRow.idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    if existing.request_hash != request_hash:
                        raise ReviewIdempotencyConflict()
                    task = session.get(ReviewTaskRow, review_task_id)
                    if task is None:
                        raise ReviewNotFound(review_task_id)
                    return self._task_detail(session, task)

                task = session.execute(
                    select(ReviewTaskRow)
                    .where(ReviewTaskRow.review_task_id == review_task_id)
                    .with_for_update()
                ).scalar_one_or_none()
                if task is None:
                    raise ReviewNotFound(review_task_id)

                existing = session.scalar(
                    select(ReviewRevisionRow).where(
                        ReviewRevisionRow.task_id == review_task_id,
                        ReviewRevisionRow.idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    if existing.request_hash != request_hash:
                        raise ReviewIdempotencyConflict()
                    return self._task_detail(session, task)

                if task.status != ReviewTaskStatus.PENDING.value:
                    raise ReviewStateConflict(task.status)
                if task.version != expected_version:
                    raise ReviewVersionConflict(task.version)

                now = _utcnow()
                revision_id = f"REV-{uuid4().hex[:12].upper()}"
                session.add(
                    ReviewRevisionRow(
                        revision_id=revision_id,
                        task_id=task.review_task_id,
                        run_id=task.run_id,
                        outcome=outcome,
                        reason_code=reason_code,
                        note=note,
                        reviewer_id=reviewer_id,
                        context_source=context_source,
                        decision_source="HUMAN",
                        idempotency_key=idempotency_key,
                        request_hash=request_hash,
                        created_at=now,
                    )
                )
                session.flush()

                resolved = outcome != HumanOutcome.UNRESOLVED.value
                new_status = (
                    ReviewTaskStatus.RESOLVED.value
                    if resolved
                    else ReviewTaskStatus.UNRESOLVED.value
                )
                effective_id = revision_id if resolved else None
                unresolved_reason = None if resolved else (note or reason_code)
                result = session.execute(
                    update(ReviewTaskRow)
                    .where(
                        ReviewTaskRow.review_task_id == review_task_id,
                        ReviewTaskRow.status == ReviewTaskStatus.PENDING.value,
                        ReviewTaskRow.version == expected_version,
                    )
                    .values(
                        status=new_status,
                        version=ReviewTaskRow.version + 1,
                        effective_revision_id=effective_id,
                        unresolved_reason=unresolved_reason,
                        updated_at=now,
                    )
                )
                if result.rowcount != 1:
                    raise ReviewVersionConflict(task.version)
                session.refresh(task)
                return self._task_detail(session, task)
        except IntegrityError as exc:
            with self.session_factory() as session:
                existing = session.scalar(
                    select(ReviewRevisionRow).where(
                        ReviewRevisionRow.task_id == review_task_id,
                        ReviewRevisionRow.idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    if existing.request_hash != request_hash:
                        raise ReviewIdempotencyConflict() from exc
                    task = session.get(ReviewTaskRow, review_task_id)
                    if task is None:
                        raise ReviewNotFound(review_task_id) from exc
                    return self._task_detail(session, task)
                task = session.get(ReviewTaskRow, review_task_id)
                if task is not None and task.status != ReviewTaskStatus.PENDING.value:
                    raise ReviewStateConflict(task.status) from exc
                if task is not None:
                    raise ReviewVersionConflict(task.version) from exc
            raise

    def _task_summary(
        self,
        task: ReviewTaskRow,
        run: QCRunRow | None,
        report: QCReportRow | None,
    ) -> dict[str, Any]:
        return {
            "reviewTaskId": task.review_task_id,
            "runId": task.run_id,
            "batchItemId": task.batch_item_id,
            "callId": run.call_id if run is not None else None,
            "status": task.status,
            "version": task.version,
            "routeReasons": list(task.route_reasons or []),
            "score": report.score if report is not None else None,
            "createdAt": _iso(task.created_at),
            "updatedAt": _iso(task.updated_at),
            "effectiveRevisionId": task.effective_revision_id,
        }

    def _revision_view(self, row: ReviewRevisionRow) -> dict[str, Any]:
        return {
            "revisionId": row.revision_id,
            "taskId": row.task_id,
            "runId": row.run_id,
            "outcome": row.outcome,
            "reasonCode": row.reason_code,
            "note": row.note,
            "reviewerId": row.reviewer_id,
            "contextSource": row.context_source,
            "decisionSource": row.decision_source,
            "createdAt": _iso(row.created_at),
        }

    def _bundle(self, session, task: ReviewTaskRow) -> dict[str, Any]:
        run = session.get(QCRunRow, task.run_id)
        report = session.scalar(select(QCReportRow).where(QCReportRow.run_id == task.run_id))
        revisions = session.scalars(
            select(ReviewRevisionRow)
            .where(ReviewRevisionRow.task_id == task.review_task_id)
            .order_by(ReviewRevisionRow.created_at, ReviewRevisionRow.revision_id)
        ).all()
        history = [self._revision_view(item) for item in revisions]
        effective = None
        if task.effective_revision_id:
            effective = next(
                (item for item in history if item["revisionId"] == task.effective_revision_id),
                None,
            )
        return {
            "reviewTask": self._task_summary(task, run, report),
            "effectiveRevision": effective,
            "reviewHistory": history,
        }

    def _task_detail(self, session, task: ReviewTaskRow) -> dict[str, Any]:
        bundle = self._bundle(session, task)
        run = session.get(QCRunRow, task.run_id)
        report = session.scalar(select(QCReportRow).where(QCReportRow.run_id == task.run_id))
        call = session.get(CallRow, run.call_id) if run is not None else None
        events = []
        if run is not None:
            events = [
                item.event_json
                for item in session.scalars(
                    select(AgentTraceEventRow)
                    .where(AgentTraceEventRow.run_id == run.run_id)
                    .order_by(AgentTraceEventRow.event_id)
                ).all()
            ]
        original = report.report_json if report is not None else None
        return {
            **bundle,
            "originalReport": original,
            "transcript": list(call.transcript_json) if call is not None else [],
            "knowledgeHits": (original or {}).get("knowledgeHits") or [],
            "auditSnapshot": (original or {}).get("auditSnapshot"),
            "trace": events,
            "errors": list(run.errors_json or []) if run is not None else [],
            "audioAvailable": bool(call.audio_artifact_uri) if call is not None else False,
            "callId": run.call_id if run is not None else None,
            "caseId": run.case_id if run is not None else None,
        }
