from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from qc.database import create_database_engine, create_session_factory
from qc.orm_models import KnowledgeBuildRow, KnowledgeCurrentPointerRow, KnowledgeDocumentRow, KnowledgeChunkRow


class PostgresKnowledgeStore:
    """Transactional READY/pointer store for production deployments."""

    def __init__(self, database_url: str):
        self.engine = create_database_engine(database_url)
        self.session_factory = create_session_factory(self.engine)

    def save_ready(self, manifest: dict[str, Any], documents=None, vectors=None) -> None:
        version = manifest["knowledgeVersion"]
        with self.session_factory.begin() as session:
            session.add(KnowledgeBuildRow(knowledge_version=version, status="READY", manifest=manifest, source_hash=manifest["sourceHash"], manifest_hash=manifest["manifestHash"], index_hash=manifest["indexHash"], created_at=datetime.now(timezone.utc)))
            if documents is not None:
                seen_documents = set()
                for index, document in enumerate(documents):
                    document_key = f"{version}:{document['documentId']}"
                    if document_key not in seen_documents:
                        seen_documents.add(document_key)
                        session.add(KnowledgeDocumentRow(document_key=document_key, knowledge_version=version, document_id=document["documentId"], document_version=document["version"], category=document["category"], document_status=document.get("documentStatus", "published"), source_hash=document["contentHash"], metadata_json={key: value for key, value in document.items() if key not in {"content"}}))
                    key = f"{version}:{document['chunkId']}"
                    parse = lambda value: datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
                    session.add(KnowledgeChunkRow(chunk_key=key, knowledge_version=version, document_id=document["documentId"], document_version=document["version"], chunk_id=document["chunkId"], title=document["title"], content=document["content"], content_hash=document["contentHash"], source_range=document.get("sourceRange") or {}, category=document["category"], document_status=document.get("documentStatus", "published"), event_type=document["eventType"], rule_relation=document.get("relatedRuleIds", []), effective_from=parse(document.get("effectiveFrom")), effective_to=parse(document.get("effectiveTo")), embedding_json=(vectors[index].tolist() if vectors is not None else None), embedding=(vectors[index].tolist() if vectors is not None else None)))

    def publish(self, version: str, actor: str = "system") -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        with self.session_factory.begin() as session:
            build = session.get(KnowledgeBuildRow, version)
            if build is None or build.status not in {"READY", "PUBLISHED"}:
                raise ValueError("only READY/PUBLISHED builds can be selected")
            session.query(KnowledgeBuildRow).filter(KnowledgeBuildRow.status == "PUBLISHED").update({"status": "READY"}, synchronize_session=False)
            build.status = "PUBLISHED"
            build.published_at = now
            pointer = session.get(KnowledgeCurrentPointerRow, 1)
            if pointer is None:
                pointer = KnowledgeCurrentPointerRow(pointer_id=1, knowledge_version=version, actor=actor, updated_at=now)
                session.add(pointer)
            else:
                pointer.knowledge_version, pointer.actor, pointer.updated_at = version, actor, now
        return {"knowledgeVersion": version, "actor": actor, "updatedAt": now.isoformat()}

    def current(self) -> str | None:
        with self.session_factory() as session:
            pointer = session.get(KnowledgeCurrentPointerRow, 1)
            return pointer.knowledge_version if pointer else None

    def rollback(self, version: str, actor: str = "system") -> dict[str, Any]:
        return self.publish(version, actor)
