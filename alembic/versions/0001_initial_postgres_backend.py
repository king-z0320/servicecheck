"""Create the Phase 1 PostgreSQL backend tables.

Revision ID: 0001
Revises:
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("case_id", sa.String(64), primary_key=True),
        sa.Column("customer_display_name", sa.String(128), nullable=False),
        sa.Column("assigned_agent_display_name", sa.String(128)),
        sa.Column("source_kind", sa.String(32), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "source_kind IN ('REAL_AUDIO', 'DEMO', 'IMPORTED')",
            name="ck_cases_source_kind",
        ),
    )
    op.create_table(
        "calls",
        sa.Column("call_id", sa.String(64), primary_key=True),
        sa.Column("case_id", sa.String(64), sa.ForeignKey("cases.case_id"), nullable=False),
        sa.Column("call_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.BigInteger(), nullable=False),
        sa.Column("audio_artifact_uri", sa.Text()),
        sa.Column("audio_sha256", sa.String(64)),
        sa.Column("audio_mime_type", sa.String(128)),
        sa.Column(
            "transcript_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("transcript_version", sa.String(128), nullable=False),
        sa.Column("asr_model", sa.String(128)),
        sa.Column("asr_model_version", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("duration_ms >= 0", name="ck_calls_duration_nonnegative"),
        sa.CheckConstraint(
            "jsonb_typeof(transcript_json) = 'array'",
            name="ck_calls_transcript_array",
        ),
    )
    op.create_index("ix_calls_case_id", "calls", ["case_id"])
    op.create_table(
        "qc_runs",
        sa.Column("run_id", sa.String(64), primary_key=True),
        sa.Column("case_id", sa.String(64), sa.ForeignKey("cases.case_id"), nullable=False),
        sa.Column("call_id", sa.String(64), sa.ForeignKey("calls.call_id"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("request_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column(
            "errors_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("loop_used", sa.Boolean(), nullable=False),
        sa.Column("loop_reason", sa.Text()),
        sa.Column("model", sa.String(128)),
        sa.Column("prompt_version", sa.String(128)),
        sa.Column("rule_version", sa.String(128)),
        sa.Column("knowledge_version", sa.String(128)),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('RUNNING', 'COMPLETED', 'PARTIAL', 'FAILED')",
            name="ck_qc_runs_status",
        ),
    )
    op.create_index(
        "ix_qc_runs_call_started",
        "qc_runs",
        ["call_id", sa.text("started_at DESC")],
    )
    op.create_index(
        "ix_qc_runs_status_started",
        "qc_runs",
        ["status", sa.text("started_at DESC")],
    )
    op.create_table(
        "qc_reports",
        sa.Column("report_id", sa.String(64), primary_key=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("qc_runs.run_id"), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("disposition", sa.String(32), nullable=False),
        sa.Column("report_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", name="uq_qc_reports_run_id"),
        sa.CheckConstraint("score BETWEEN 0 AND 100", name="ck_qc_reports_score"),
        sa.CheckConstraint(
            "disposition IN ('AUTO_PASS', 'AUTO_VIOLATION', 'HUMAN_REVIEW_REQUIRED')",
            name="ck_qc_reports_disposition",
        ),
    )
    op.create_table(
        "agent_trace_events",
        sa.Column("event_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("qc_runs.run_id"), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("phase", sa.String(32), nullable=False),
        sa.Column("event_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("iteration >= 0", name="ck_agent_trace_iteration"),
        sa.CheckConstraint(
            "phase IN ('PLAN', 'ACT', 'OBSERVE', 'EVALUATE', 'REPLAN', 'FINALIZE')",
            name="ck_agent_trace_phase",
        ),
    )
    op.create_index(
        "ix_agent_trace_run_event",
        "agent_trace_events",
        ["run_id", "event_id"],
    )
    op.create_table(
        "batch_jobs",
        sa.Column("batch_id", sa.String(64), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('CREATED', 'RUNNING', 'PARTIAL', 'COMPLETED', 'FAILED')",
            name="ck_batch_jobs_status",
        ),
        sa.CheckConstraint("total >= 0", name="ck_batch_jobs_total"),
    )
    op.create_table(
        "batch_items",
        sa.Column("item_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("batch_id", sa.String(64), sa.ForeignKey("batch_jobs.batch_id"), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("call_id", sa.String(64), sa.ForeignKey("calls.call_id")),
        sa.Column("idempotency_key", sa.String(256), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("failed_reason", sa.Text()),
        sa.Column("request_snapshot", postgresql.JSONB()),
        sa.Column("result_snapshot", postgresql.JSONB()),
        sa.Column("qc_run_id", sa.String(64), sa.ForeignKey("qc_runs.run_id")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "batch_id",
            "idempotency_key",
            name="uq_batch_items_batch_idempotency",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'INTERRUPTED', 'DONE', "
            "'FAILED_FINAL', 'DEAD_LETTER', 'HUMAN_REVIEW')",
            name="ck_batch_items_status",
        ),
    )
    op.create_index("ix_batch_items_batch_id", "batch_items", ["batch_id"])
    op.create_table(
        "stage_executions",
        sa.Column(
            "stage_execution_id",
            sa.BigInteger(),
            sa.Identity(),
            primary_key=True,
        ),
        sa.Column("item_id", sa.BigInteger(), sa.ForeignKey("batch_items.item_id"), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("artifact_uri", sa.Text()),
        sa.Column("sha256", sa.String(64)),
        sa.Column("producer_version", sa.String(128)),
        sa.Column("error_code", sa.String(128)),
        sa.Column("retryable", sa.Boolean()),
        sa.Column("error_summary", sa.Text()),
        sa.CheckConstraint(
            "stage IN ('TRANSCODE', 'ASR', 'EMOTION', 'EVENT_EXTRACT', "
            "'RAG', 'AUDIT', 'QC', 'LOOP')",
            name="ck_stage_executions_stage",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'DONE', 'FAILED')",
            name="ck_stage_executions_status",
        ),
        sa.CheckConstraint("duration_ms >= 0", name="ck_stage_duration"),
        sa.CheckConstraint("attempts >= 0", name="ck_stage_attempts"),
    )
    op.create_index(
        "ix_stage_item_stage_latest",
        "stage_executions",
        ["item_id", "stage", sa.text("stage_execution_id DESC")],
    )
    op.create_table(
        "batch_exports",
        sa.Column("export_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("batch_id", sa.String(64), sa.ForeignKey("batch_jobs.batch_id"), nullable=False),
        sa.Column("format", sa.String(16), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("sha256", sa.String(64)),
        sa.Column("producer_version", sa.String(128), nullable=False),
        sa.Column("error_code", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint(
            "batch_id",
            "format",
            "artifact_uri",
            name="uq_batch_exports_identity",
        ),
        sa.CheckConstraint(
            "status IN ('RUNNING', 'DONE', 'FAILED')",
            name="ck_batch_exports_status",
        ),
    )
    op.create_index("ix_batch_exports_batch_id", "batch_exports", ["batch_id"])


def downgrade() -> None:
    op.drop_index("ix_batch_exports_batch_id", table_name="batch_exports")
    op.drop_table("batch_exports")
    op.drop_index("ix_stage_item_stage_latest", table_name="stage_executions")
    op.drop_table("stage_executions")
    op.drop_index("ix_batch_items_batch_id", table_name="batch_items")
    op.drop_table("batch_items")
    op.drop_table("batch_jobs")
    op.drop_index("ix_agent_trace_run_event", table_name="agent_trace_events")
    op.drop_table("agent_trace_events")
    op.drop_table("qc_reports")
    op.drop_index("ix_qc_runs_status_started", table_name="qc_runs")
    op.drop_index("ix_qc_runs_call_started", table_name="qc_runs")
    op.drop_table("qc_runs")
    op.drop_index("ix_calls_case_id", table_name="calls")
    op.drop_table("calls")
    op.drop_table("cases")
