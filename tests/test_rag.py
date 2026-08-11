from datetime import datetime, timezone

from qc.models import EventType
from qc.rag import KnowledgeIndex


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            vectors.append([
                float("还款" in text or "还清" in text or "REPAYMENT" in text),
                float("第三方" in text or "THIRD" in text or "家属" in text),
                float("威胁" in text or "抓人" in text or "恐吓" in text or "THREAT" in text),
            ])
        return vectors


def test_search_filters_by_event_type_and_returns_source_metadata():
    index = KnowledgeIndex("knowledge", embedder=FakeEmbedder())
    index.build()
    hits = index.search(
        "客户说已经还清，客服应如何处理",
        EventType.REPAYMENT_DISPUTE,
        datetime(2026, 7, 27, tzinfo=timezone.utc),
        top_k=3,
    )
    assert hits[0].documentId == "POLICY-REPAYMENT-003"
    assert all(
        hit.metadata["eventType"] == "REPAYMENT_DISPUTE"
        for hit in hits
    )
    assert index.index_version
    assert "indexVersion" in hits[0].metadata


def test_search_does_not_return_unrelated_third_party_policy():
    index = KnowledgeIndex("knowledge", embedder=FakeEmbedder())
    index.build()
    hits = index.search(
        "客户说已经还清",
        EventType.REPAYMENT_DISPUTE,
        datetime(2026, 7, 27, tzinfo=timezone.utc),
    )
    assert "POLICY-THIRD-PARTY-001" not in {
        hit.documentId for hit in hits
    }


def test_hybrid_search_can_surface_rule_documents():
    index = KnowledgeIndex("knowledge", embedder=FakeEmbedder())
    index.build()
    # 规则 R006 已进入索引
    assert any(d["documentId"] == "R006" for d in index.documents)
    hits = index.search(
        "还款争议 直接否定 核验工单",
        EventType.REPAYMENT_DISPUTE,
        datetime(2026, 7, 27, tzinfo=timezone.utc),
        top_k=5,
    )
    ids = {h.documentId for h in hits}
    assert "POLICY-REPAYMENT-003" in ids or "R006" in ids


def test_threat_policy_retrievable():
    index = KnowledgeIndex("knowledge", embedder=FakeEmbedder())
    index.build()
    hits = index.search(
        "法院抓人 恐吓",
        EventType.THREAT_OR_COERCION,
        datetime(2026, 7, 27, tzinfo=timezone.utc),
        top_k=3,
    )
    assert hits
    assert all(h.metadata["eventType"] == "THREAT_OR_COERCION" for h in hits)
