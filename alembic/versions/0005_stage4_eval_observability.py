"""Add stage 4 evaluation and LLM usage tables.

Revision ID: 0005
Revises: 0004
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column("eval_run_id", sa.String(64), primary_key=True),
        sa.Column("split", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("dataset_hash", sa.String(64), nullable=False),
        sa.Column("manifest", postgresql.JSONB(), nullable=False),
        sa.Column("baseline_eval_run_id", sa.String(64)),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("case_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_code", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("split IN ('dev', 'regression', 'challenge')", name="ck_eval_runs_split"),
        sa.CheckConstraint("status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')", name="ck_eval_runs_status"),
    )
    op.create_index("ix_eval_runs_split_created", "eval_runs", ["split", sa.text("created_at DESC")])
    op.create_table(
        "eval_case_results",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("eval_run_id", sa.String(64), sa.ForeignKey("eval_runs.eval_run_id"), nullable=False),
        sa.Column("case_id", sa.String(128), nullable=False),
        sa.Column("case_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("deterministic_metrics", postgresql.JSONB(), nullable=False),
        sa.Column("rag_metrics", postgresql.JSONB(), nullable=False),
        sa.Column("judge_result", postgresql.JSONB()),
        sa.Column("failure_reasons", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("trace_id", sa.String(64)),
        sa.Column("run_id", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("eval_run_id", "case_id", name="uq_eval_case_results_run_case"),
        sa.CheckConstraint("status IN ('passed', 'failed', 'needs_review', 'not_run', 'unavailable')", name="ck_eval_case_results_status"),
    )
    op.create_index("ix_eval_case_results_run_status", "eval_case_results", ["eval_run_id", "status"])
    op.create_table(
        "llm_usage_records",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("invocation_id", sa.String(64), nullable=False),
        sa.Column("run_id", sa.String(64)),
        sa.Column("eval_run_id", sa.String(64)),
        sa.Column("operation", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("token_source", sa.String(32), nullable=False),
        sa.Column("input_tokens", sa.BigInteger()),
        sa.Column("output_tokens", sa.BigInteger()),
        sa.Column("estimated_cost", sa.Float()),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("price_config_version", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("invocation_id", name="uq_llm_usage_records_invocation"),
        sa.CheckConstraint("attempt >= 1", name="ck_llm_usage_attempt"),
    )
    op.create_index("ix_llm_usage_run", "llm_usage_records", ["run_id", "created_at"])
    op.create_index("ix_llm_usage_eval_run", "llm_usage_records", ["eval_run_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_llm_usage_eval_run", table_name="llm_usage_records")
    op.drop_index("ix_llm_usage_run", table_name="llm_usage_records")
    op.drop_table("llm_usage_records")
    op.drop_index("ix_eval_case_results_run_status", table_name="eval_case_results")
    op.drop_table("eval_case_results")
    op.drop_index("ix_eval_runs_split_created", table_name="eval_runs")
    op.drop_table("eval_runs")

