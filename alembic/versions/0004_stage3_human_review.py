"""Add stage 3 human review tables.

Revision ID: 0004
Revises: 0003
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "review_tasks",
        sa.Column("review_task_id", sa.String(64), primary_key=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("qc_runs.run_id"), nullable=False),
        sa.Column("batch_item_id", sa.BigInteger(), sa.ForeignKey("batch_items.item_id")),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column(
            "route_reasons",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("effective_revision_id", sa.String(64)),
        sa.Column("unresolved_reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", name="uq_review_tasks_run_id"),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RESOLVED', 'UNRESOLVED')",
            name="ck_review_tasks_status",
        ),
        sa.CheckConstraint("version >= 1", name="ck_review_tasks_version"),
        sa.CheckConstraint(
            "(status <> 'PENDING') OR (effective_revision_id IS NULL)",
            name="ck_review_tasks_pending_pointer",
        ),
        sa.CheckConstraint(
            "(status <> 'RESOLVED') OR (effective_revision_id IS NOT NULL)",
            name="ck_review_tasks_resolved_pointer",
        ),
        sa.CheckConstraint(
            "(status <> 'UNRESOLVED') OR "
            "(effective_revision_id IS NULL AND unresolved_reason IS NOT NULL)",
            name="ck_review_tasks_unresolved",
        ),
    )
    op.create_index(
        "ix_review_tasks_status_updated",
        "review_tasks",
        ["status", "updated_at"],
    )
    op.create_index("ix_review_tasks_created", "review_tasks", ["created_at"])
    op.create_index("ix_review_tasks_batch_item", "review_tasks", ["batch_item_id"])
    op.create_table(
        "review_revisions",
        sa.Column("revision_id", sa.String(64), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(64),
            sa.ForeignKey("review_tasks.review_task_id"),
            nullable=False,
        ),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("qc_runs.run_id"), nullable=False),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("reason_code", sa.String(64), nullable=False),
        sa.Column("note", sa.Text(), nullable=False, server_default=""),
        sa.Column("reviewer_id", sa.String(128), nullable=False),
        sa.Column("context_source", sa.String(32), nullable=False),
        sa.Column("decision_source", sa.String(16), nullable=False),
        sa.Column("idempotency_key", sa.String(256), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("task_id", name="uq_review_revisions_task_id"),
        sa.UniqueConstraint(
            "task_id",
            "idempotency_key",
            name="uq_review_revisions_task_idempotency",
        ),
        sa.CheckConstraint(
            "outcome IN ('CONFIRMED_PASS', 'CONFIRMED_VIOLATION', 'UNRESOLVED')",
            name="ck_review_revisions_outcome",
        ),
        sa.CheckConstraint(
            "decision_source IN ('HUMAN')",
            name="ck_review_revisions_decision_source",
        ),
        sa.CheckConstraint(
            "context_source IN ('CONFIGURED_DEMO')",
            name="ck_review_revisions_context_source",
        ),
    )
    op.create_index("ix_review_revisions_run_id", "review_revisions", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_review_revisions_run_id", table_name="review_revisions")
    op.drop_table("review_revisions")
    op.drop_index("ix_review_tasks_batch_item", table_name="review_tasks")
    op.drop_index("ix_review_tasks_created", table_name="review_tasks")
    op.drop_index("ix_review_tasks_status_updated", table_name="review_tasks")
    op.drop_table("review_tasks")
