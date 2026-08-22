from __future__ import annotations

"""Small pgvector Dense adapter used by the stage-5 production integration."""

from datetime import datetime

from sqlalchemy import select

from qc.orm_models import KnowledgeChunkRow


class PgVectorDenseStore:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def search(self, query_vector: list[float], *, knowledge_version: str, event_type: str, at_time: datetime, top_k: int = 20, rule_relation: str | None = None):
        """Apply all hard filters in SQL before cosine-distance ordering."""
        return [row for row, _ in self.search_scored(query_vector, knowledge_version=knowledge_version, event_type=event_type, at_time=at_time, top_k=top_k, rule_relation=rule_relation)]

    def search_scored(self, query_vector: list[float], *, knowledge_version: str, event_type: str, at_time: datetime, top_k: int = 20, rule_relation: str | None = None):
        distance = KnowledgeChunkRow.embedding.cosine_distance(query_vector)
        statement = select(KnowledgeChunkRow, distance.label("distance")).where(
            KnowledgeChunkRow.knowledge_version == knowledge_version,
            KnowledgeChunkRow.document_status == "published",
            KnowledgeChunkRow.event_type == event_type,
            KnowledgeChunkRow.effective_from <= at_time,
            (KnowledgeChunkRow.effective_to.is_(None) | (KnowledgeChunkRow.effective_to > at_time)),
        )
        if rule_relation:
            statement = statement.where(KnowledgeChunkRow.rule_relation.contains([rule_relation]))
        statement = statement.order_by(distance).limit(top_k)
        with self.session_factory() as session:
            return list(session.execute(statement).all())
