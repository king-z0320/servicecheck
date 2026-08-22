from datetime import datetime, timezone

from qc.knowledge_build import KnowledgeBuildService
from qc.models import EventType
from qc.rag import KnowledgeIndex, StructuredChunker, rrf_merge


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        return [[float("还款" in text), float("威胁" in text), float("第三方" in text)] for text in texts]


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        return [float("核验" in text) for text in texts]


def test_rrf_uses_rank_not_heterogeneous_raw_score():
    rows = rrf_merge([("a", 0.1), ("b", 0.99)], [("a", 3.0)], rrf_k=60)
    assert rows[0]["chunkId"] == "a"
    assert rows[0]["rrfScore"] == 1 / 61 + 1 / 61


def test_structured_chunk_has_stable_identity_and_hash():
    document = {"documentId": "R006", "content": "x" * 1700}
    first = StructuredChunker(max_chars=800).chunk(document)
    second = StructuredChunker(max_chars=800).chunk(document)
    assert [item["chunkId"] for item in first] == [item["chunkId"] for item in second]
    assert [item["contentHash"] for item in first] == [item["contentHash"] for item in second]
    assert len(first) == 3


def test_reranker_is_called_and_snapshot_fields_are_present():
    reranker = FakeReranker()
    index = KnowledgeIndex("knowledge", embedder=FakeEmbedder(), reranker=reranker)
    index.build()
    hits = index.search("还款争议核验", EventType.REPAYMENT_DISPUTE, datetime(2026, 7, 27, tzinfo=timezone.utc))
    assert reranker.calls == 1
    assert hits[0].chunkId and hits[0].contentHash and hits[0].knowledgeVersion
    assert hits[0].retrievalMethod == "hybrid_rrf_rerank"


def test_pointer_publish_and_rollback_do_not_rebuild(tmp_path):
    service = KnowledgeBuildService("knowledge", state_dir=tmp_path, embedder=FakeEmbedder(), reranker=FakeReranker())
    first = service.build()
    second = service.build()
    service.publish(first.knowledge_version)
    assert service.current()["knowledgeVersion"] == first.knowledge_version
    service.rollback(second.knowledge_version)
    assert service.current()["knowledgeVersion"] == second.knowledge_version
    assert service._read(first.knowledge_version)["status"] == "READY"


def test_hard_filter_rejects_unpublished_and_wrong_rule_relation():
    index = KnowledgeIndex("knowledge", embedder=FakeEmbedder(), reranker=FakeReranker())
    index.build()
    index.documents[0]["documentStatus"] = "draft"
    assert all(hit.metadata["documentStatus"] == "published" for hit in index.search("还款", EventType.REPAYMENT_DISPUTE, datetime(2026, 7, 27, tzinfo=timezone.utc)))
    assert index.search("还款", EventType.REPAYMENT_DISPUTE, datetime(2026, 7, 27, tzinfo=timezone.utc), rule_relation="R-NOT-EXIST") == []

