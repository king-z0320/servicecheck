"""Add and backfill the runtime version.

Revision ID: 0002
Revises: 0001
"""

from alembic import op
import sqlalchemy as sa


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "qc_runs",
        sa.Column("runtime_version", sa.String(128), nullable=True),
    )
    op.execute(
        "UPDATE qc_runs SET runtime_version = 'legacy-unknown' "
        "WHERE runtime_version IS NULL OR runtime_version = ''"
    )
    op.alter_column("qc_runs", "runtime_version", nullable=False)


def downgrade() -> None:
    op.drop_column("qc_runs", "runtime_version")
