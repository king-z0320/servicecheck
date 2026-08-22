"""Immutable knowledge builds, pointer, chunks and pgvector storage.

The first statement intentionally fails with PostgreSQL's native extension
error when pgvector is not installed; an unavailable vector extension must not
silently downgrade production retrieval.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "knowledge_builds",
        sa.Column("knowledge_version", sa.String(128), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("manifest", postgresql.JSONB(), nullable=False),
        sa.Column("source_hash", sa.String(64), nullable=False),
        sa.Column("manifest_hash", sa.String(64), nullable=False),
        sa.Column("index_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('BUILDING','READY','PUBLISHED','FAILED')", name="ck_knowledge_build_status"),
    )
    op.create_table(
        "knowledge_documents",
        sa.Column("document_key", sa.String(256), primary_key=True),
        sa.Column("knowledge_version", sa.String(128), sa.ForeignKey("knowledge_builds.knowledge_version"), nullable=False),
        sa.Column("document_id", sa.String(128), nullable=False),
        sa.Column("document_version", sa.String(128), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("document_status", sa.String(16), nullable=False),
        sa.Column("source_hash", sa.String(64), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False),
    )
    op.create_table(
        "knowledge_chunks",
        sa.Column("chunk_key", sa.String(256), primary_key=True),
        sa.Column("knowledge_version", sa.String(128), sa.ForeignKey("knowledge_builds.knowledge_version"), nullable=False),
        sa.Column("document_id", sa.String(128), nullable=False),
        sa.Column("document_version", sa.String(128), nullable=False),
        sa.Column("chunk_id", sa.String(256), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("source_range", postgresql.JSONB(), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("document_status", sa.String(16), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("rule_relation", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("effective_from", sa.DateTime(timezone=True)),
        sa.Column("effective_to", sa.DateTime(timezone=True)),
        # The JSON copy supports offline inspection; embedding is the pgvector
        # column used by the production Dense query.
        sa.Column("embedding_json", postgresql.JSONB()),
    )
    # bge-small-zh-v1.5 emits 512 dimensions; a future model dimension must
    # create a new migration/build family rather than mutate this table.
    op.execute("ALTER TABLE knowledge_chunks ADD COLUMN embedding vector(512)")
    op.create_table(
        "knowledge_current_pointer",
        sa.Column("pointer_id", sa.Integer(), primary_key=True),
        sa.Column("knowledge_version", sa.String(128), sa.ForeignKey("knowledge_builds.knowledge_version"), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_knowledge_chunk_filter", "knowledge_chunks", ["knowledge_version", "document_status", "event_type", "effective_from"])
    op.create_index("ix_knowledge_chunk_embedding_hnsw", "knowledge_chunks", ["embedding"], postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"})


def downgrade() -> None:
    op.drop_index("ix_knowledge_chunk_embedding_hnsw", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunk_filter", table_name="knowledge_chunks")
    op.drop_table("knowledge_current_pointer")
    op.drop_table("knowledge_chunks")
    op.drop_table("knowledge_documents")
    op.drop_table("knowledge_builds")
