"""Add stage 2 batch async/outbox tables.

Revision ID: 0003
Revises: 0002
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_batch_jobs_status", "batch_jobs", type_="check")
    op.create_check_constraint(
        "ck_batch_jobs_status",
        "batch_jobs",
        "status IN ('CREATED', 'QUEUED', 'RUNNING', 'PARTIAL', 'COMPLETED', 'FAILED')",
    )
    op.add_column("batch_jobs", sa.Column("source_snapshot", postgresql.JSONB()))
    op.add_column("batch_items", sa.Column("source_sha256", sa.String(64)))
    op.add_column("batch_items", sa.Column("source_size", sa.BigInteger()))

    op.create_table(
        "batch_creation_requests",
        sa.Column("request_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("idempotency_key", sa.String(256), nullable=False),
        sa.Column("batch_id", sa.String(64), sa.ForeignKey("batch_jobs.batch_id"), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("idempotency_key", name="uq_batch_creation_idempotency"),
    )
    op.create_table(
        "outbox_events",
        sa.Column("event_id", sa.String(64), primary_key=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("aggregate_type", sa.String(64), nullable=False),
        sa.Column("aggregate_id", sa.String(128), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="PENDING"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("redis_message_id", sa.String(128)),
        sa.Column("last_error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('PENDING', 'PUBLISHED', 'FAILED')", name="ck_outbox_status"),
    )
    op.create_index(
        "ix_outbox_pending",
        "outbox_events",
        ["status", "available_at", "created_at"],
    )
    op.create_table(
        "batch_dead_letters",
        sa.Column("dead_letter_id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("batch_id", sa.String(64), sa.ForeignKey("batch_jobs.batch_id"), nullable=False),
        sa.Column("item_id", sa.BigInteger(), sa.ForeignKey("batch_items.item_id"), nullable=False),
        sa.Column("message_id", sa.String(128)),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("error_code", sa.String(128), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_batch_dead_letters_item",
        "batch_dead_letters",
        ["item_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_batch_dead_letters_item", table_name="batch_dead_letters")
    op.drop_table("batch_dead_letters")
    op.drop_index("ix_outbox_pending", table_name="outbox_events")
    op.drop_table("outbox_events")
    op.drop_table("batch_creation_requests")
    # Tolerate an unreleased development snapshot of 0003 that predated
    # these two columns; released revisions must never be edited this way.
    op.execute("ALTER TABLE batch_items DROP COLUMN IF EXISTS source_size")
    op.execute("ALTER TABLE batch_items DROP COLUMN IF EXISTS source_sha256")
    op.drop_column("batch_jobs", "source_snapshot")
    op.drop_constraint("ck_batch_jobs_status", "batch_jobs", type_="check")
    op.create_check_constraint(
        "ck_batch_jobs_status",
        "batch_jobs",
        "status IN ('CREATED', 'RUNNING', 'PARTIAL', 'COMPLETED', 'FAILED')",
    )
