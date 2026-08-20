from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from qc.database import Base


class CaseRow(Base):
    __tablename__ = "cases"
    __table_args__ = (
        CheckConstraint(
            "source_kind IN ('REAL_AUDIO', 'DEMO', 'IMPORTED')",
            name="ck_cases_source_kind",
        ),
    )

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    assigned_agent_display_name: Mapped[str | None] = mapped_column(String(128))
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class CallRow(Base):
    __tablename__ = "calls"
    __table_args__ = (
        CheckConstraint("duration_ms >= 0", name="ck_calls_duration_nonnegative"),
        CheckConstraint(
            "jsonb_typeof(transcript_json) = 'array'",
            name="ck_calls_transcript_array",
        ),
        Index("ix_calls_case_id", "case_id"),
    )

    call_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), nullable=False)
    call_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    audio_artifact_uri: Mapped[str | None] = mapped_column(Text)
    audio_sha256: Mapped[str | None] = mapped_column(String(64))
    audio_mime_type: Mapped[str | None] = mapped_column(String(128))
    transcript_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    transcript_version: Mapped[str] = mapped_column(String(128), nullable=False)
    asr_model: Mapped[str | None] = mapped_column(String(128))
    asr_model_version: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QCRunRow(Base):
    __tablename__ = "qc_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('RUNNING', 'COMPLETED', 'PARTIAL', 'FAILED')",
            name="ck_qc_runs_status",
        ),
        Index("ix_qc_runs_call_started", "call_id", text("started_at DESC")),
        Index("ix_qc_runs_status_started", "status", text("started_at DESC")),
    )

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), nullable=False)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    request_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    errors_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    loop_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    loop_reason: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(128))
    prompt_version: Mapped[str | None] = mapped_column(String(128))
    rule_version: Mapped[str | None] = mapped_column(String(128))
    knowledge_version: Mapped[str | None] = mapped_column(String(128))
    runtime_version: Mapped[str] = mapped_column(String(128), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QCReportRow(Base):
    __tablename__ = "qc_reports"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_qc_reports_run_id"),
        CheckConstraint("score BETWEEN 0 AND 100", name="ck_qc_reports_score"),
        CheckConstraint(
            "disposition IN ('AUTO_PASS', 'AUTO_VIOLATION', 'HUMAN_REVIEW_REQUIRED')",
            name="ck_qc_reports_disposition",
        ),
    )

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("qc_runs.run_id"), nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    disposition: Mapped[str] = mapped_column(String(32), nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AgentTraceEventRow(Base):
    __tablename__ = "agent_trace_events"
    __table_args__ = (
        CheckConstraint("iteration >= 0", name="ck_agent_trace_iteration"),
        CheckConstraint(
            "phase IN ('PLAN', 'ACT', 'OBSERVE', 'EVALUATE', 'REPLAN', 'FINALIZE')",
            name="ck_agent_trace_phase",
        ),
        Index("ix_agent_trace_run_event", "run_id", "event_id"),
    )

    event_id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("qc_runs.run_id"), nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    phase: Mapped[str] = mapped_column(String(32), nullable=False)
    event_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class BatchJobRow(Base):
    __tablename__ = "batch_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('CREATED', 'QUEUED', 'RUNNING', 'PARTIAL', 'COMPLETED', 'FAILED')",
            name="ck_batch_jobs_status",
        ),
        CheckConstraint("total >= 0", name="ck_batch_jobs_total"),
    )

    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    total: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_snapshot: Mapped[dict | None] = mapped_column(JSONB)


class BatchCreationRequestRow(Base):
    __tablename__ = "batch_creation_requests"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_batch_creation_idempotency"),)

    request_id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    batch_id: Mapped[str] = mapped_column(ForeignKey("batch_jobs.batch_id"), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OutboxEventRow(Base):
    __tablename__ = "outbox_events"
    __table_args__ = (
        CheckConstraint("status IN ('PENDING', 'PUBLISHED', 'FAILED')", name="ck_outbox_status"),
        Index("ix_outbox_pending", "status", "available_at", "created_at"),
    )

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    redis_message_id: Mapped[str | None] = mapped_column(String(128))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class BatchDeadLetterRow(Base):
    __tablename__ = "batch_dead_letters"
    __table_args__ = (Index("ix_batch_dead_letters_item", "item_id", "created_at"),)

    dead_letter_id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("batch_jobs.batch_id"), nullable=False)
    item_id: Mapped[int] = mapped_column(ForeignKey("batch_items.item_id"), nullable=False)
    message_id: Mapped[str | None] = mapped_column(String(128))
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    error_code: Mapped[str] = mapped_column(String(128), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    last_error: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class BatchItemRow(Base):
    __tablename__ = "batch_items"
    __table_args__ = (
        UniqueConstraint(
            "batch_id",
            "idempotency_key",
            name="uq_batch_items_batch_idempotency",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'INTERRUPTED', 'DONE', "
            "'FAILED_FINAL', 'DEAD_LETTER', 'HUMAN_REVIEW')",
            name="ck_batch_items_status",
        ),
        Index("ix_batch_items_batch_id", "batch_id"),
    )

    item_id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("batch_jobs.batch_id"), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    source_sha256: Mapped[str | None] = mapped_column(String(64))
    source_size: Mapped[int | None] = mapped_column(BigInteger)
    call_id: Mapped[str | None] = mapped_column(ForeignKey("calls.call_id"))
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    failed_reason: Mapped[str | None] = mapped_column(Text)
    request_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    result_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    qc_run_id: Mapped[str | None] = mapped_column(ForeignKey("qc_runs.run_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StageExecutionRow(Base):
    __tablename__ = "stage_executions"
    __table_args__ = (
        CheckConstraint(
            "stage IN ('TRANSCODE', 'ASR', 'EMOTION', 'EVENT_EXTRACT', "
            "'RAG', 'AUDIT', 'QC', 'LOOP')",
            name="ck_stage_executions_stage",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'DONE', 'FAILED')",
            name="ck_stage_executions_status",
        ),
        CheckConstraint("duration_ms >= 0", name="ck_stage_duration"),
        CheckConstraint("attempts >= 0", name="ck_stage_attempts"),
        Index(
            "ix_stage_item_stage_latest",
            "item_id",
            "stage",
            text("stage_execution_id DESC"),
        ),
    )

    stage_execution_id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    item_id: Mapped[int] = mapped_column(ForeignKey("batch_items.item_id"), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    producer_version: Mapped[str | None] = mapped_column(String(128))
    error_code: Mapped[str | None] = mapped_column(String(128))
    retryable: Mapped[bool | None] = mapped_column(Boolean)
    error_summary: Mapped[str | None] = mapped_column(Text)


class BatchExportRow(Base):
    __tablename__ = "batch_exports"
    __table_args__ = (
        UniqueConstraint(
            "batch_id",
            "format",
            "artifact_uri",
            name="uq_batch_exports_identity",
        ),
        CheckConstraint(
            "status IN ('RUNNING', 'DONE', 'FAILED')",
            name="ck_batch_exports_status",
        ),
        Index("ix_batch_exports_batch_id", "batch_id"),
    )

    export_id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("batch_jobs.batch_id"), nullable=False)
    format: Mapped[str] = mapped_column(String(16), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64))
    producer_version: Mapped[str] = mapped_column(String(128), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ReviewTaskRow(Base):
    __tablename__ = "review_tasks"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_review_tasks_run_id"),
        CheckConstraint(
            "status IN ('PENDING', 'RESOLVED', 'UNRESOLVED')",
            name="ck_review_tasks_status",
        ),
        CheckConstraint("version >= 1", name="ck_review_tasks_version"),
        CheckConstraint(
            "(status <> 'PENDING') OR (effective_revision_id IS NULL)",
            name="ck_review_tasks_pending_pointer",
        ),
        CheckConstraint(
            "(status <> 'RESOLVED') OR (effective_revision_id IS NOT NULL)",
            name="ck_review_tasks_resolved_pointer",
        ),
        CheckConstraint(
            "(status <> 'UNRESOLVED') OR "
            "(effective_revision_id IS NULL AND unresolved_reason IS NOT NULL)",
            name="ck_review_tasks_unresolved",
        ),
        Index("ix_review_tasks_status_updated", "status", text("updated_at DESC")),
        Index("ix_review_tasks_created", text("created_at DESC")),
        Index("ix_review_tasks_batch_item", "batch_item_id"),
    )

    review_task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("qc_runs.run_id"), nullable=False)
    batch_item_id: Mapped[int | None] = mapped_column(ForeignKey("batch_items.item_id"))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    route_reasons: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    effective_revision_id: Mapped[str | None] = mapped_column(String(64))
    unresolved_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ReviewRevisionRow(Base):
    __tablename__ = "review_revisions"
    __table_args__ = (
        UniqueConstraint("task_id", name="uq_review_revisions_task_id"),
        UniqueConstraint(
            "task_id",
            "idempotency_key",
            name="uq_review_revisions_task_idempotency",
        ),
        CheckConstraint(
            "outcome IN ('CONFIRMED_PASS', 'CONFIRMED_VIOLATION', 'UNRESOLVED')",
            name="ck_review_revisions_outcome",
        ),
        CheckConstraint(
            "decision_source IN ('HUMAN')",
            name="ck_review_revisions_decision_source",
        ),
        CheckConstraint(
            "context_source IN ('CONFIGURED_DEMO')",
            name="ck_review_revisions_context_source",
        ),
        Index("ix_review_revisions_run_id", "run_id"),
    )

    revision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        ForeignKey("review_tasks.review_task_id"),
        nullable=False,
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("qc_runs.run_id"), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reviewer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    context_source: Mapped[str] = mapped_column(String(32), nullable=False)
    decision_source: Mapped[str] = mapped_column(String(16), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
