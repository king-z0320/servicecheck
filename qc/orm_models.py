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

try:  # Optional at import time so offline tests do not need pgvector wheels.
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover - exercised only before dependency install
    class Vector:  # type: ignore[no-redef]
        def __new__(cls, dimensions: int):
            return JSONB()


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


class EvalRunRow(Base):
    __tablename__ = "eval_runs"
    __table_args__ = (
        CheckConstraint("split IN ('dev', 'regression', 'challenge')", name="ck_eval_runs_split"),
        CheckConstraint("status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')", name="ck_eval_runs_status"),
        Index("ix_eval_runs_split_created", "split", text("created_at DESC")),
    )

    eval_run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    baseline_eval_run_id: Mapped[str | None] = mapped_column(String(64))
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    case_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvalCaseResultRow(Base):
    __tablename__ = "eval_case_results"
    __table_args__ = (
        UniqueConstraint("eval_run_id", "case_id", name="uq_eval_case_results_run_case"),
        CheckConstraint("status IN ('passed', 'failed', 'needs_review', 'not_run', 'unavailable')", name="ck_eval_case_results_status"),
        Index("ix_eval_case_results_run_status", "eval_run_id", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    eval_run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.eval_run_id"), nullable=False)
    case_id: Mapped[str] = mapped_column(String(128), nullable=False)
    case_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    deterministic_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    rag_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    judge_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    failure_reasons: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    trace_id: Mapped[str | None] = mapped_column(String(64))
    run_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class LLMUsageRecordRow(Base):
    __tablename__ = "llm_usage_records"
    __table_args__ = (
        UniqueConstraint("invocation_id", name="uq_llm_usage_records_invocation"),
        CheckConstraint("attempt >= 1", name="ck_llm_usage_attempt"),
        Index("ix_llm_usage_run", "run_id", "created_at"),
        Index("ix_llm_usage_eval_run", "eval_run_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    invocation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(64))
    eval_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    token_source: Mapped[str] = mapped_column(String(32), nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(BigInteger)
    output_tokens: Mapped[int | None] = mapped_column(BigInteger)
    estimated_cost: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    price_config_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


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


class KnowledgeBuildRow(Base):
    __tablename__ = "knowledge_builds"
    __table_args__ = (
        CheckConstraint("status IN ('BUILDING', 'READY', 'PUBLISHED', 'FAILED')", name="ck_knowledge_build_status"),
        Index("ix_knowledge_build_status_created", "status", text("created_at DESC")),
    )

    knowledge_version: Mapped[str] = mapped_column(String(128), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    index_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeDocumentRow(Base):
    __tablename__ = "knowledge_documents"
    __table_args__ = (Index("ix_knowledge_document_version_status", "knowledge_version", "document_status"),)

    document_key: Mapped[str] = mapped_column(String(256), primary_key=True)
    knowledge_version: Mapped[str] = mapped_column(ForeignKey("knowledge_builds.knowledge_version"), nullable=False)
    document_id: Mapped[str] = mapped_column(String(128), nullable=False)
    document_version: Mapped[str] = mapped_column(String(128), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    document_status: Mapped[str] = mapped_column(String(16), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class KnowledgeChunkRow(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("knowledge_version", "chunk_id", name="uq_knowledge_chunk_version_id"),
        Index("ix_knowledge_chunk_filter", "knowledge_version", "document_status", "event_type", "effective_from"),
    )

    chunk_key: Mapped[str] = mapped_column(String(256), primary_key=True)
    knowledge_version: Mapped[str] = mapped_column(ForeignKey("knowledge_builds.knowledge_version"), nullable=False)
    document_id: Mapped[str] = mapped_column(String(128), nullable=False)
    document_version: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(256), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_range: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    document_status: Mapped[str] = mapped_column(String(16), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_relation: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    effective_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    embedding_json: Mapped[list[float] | None] = mapped_column(JSONB)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(512), nullable=True)


class KnowledgeCurrentPointerRow(Base):
    __tablename__ = "knowledge_current_pointer"
    pointer_id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    knowledge_version: Mapped[str] = mapped_column(ForeignKey("knowledge_builds.knowledge_version"), nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
