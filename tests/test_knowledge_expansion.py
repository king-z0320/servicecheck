import json
from datetime import datetime, timezone
from pathlib import Path

from qc.models import EventType
from qc.rag import KnowledgeIndex


ROOT = Path(__file__).resolve().parents[1]
EXPANDED_EVENTS = {
    EventType.DEBT_DENIAL,
    EventType.AMOUNT_DISPUTE,
    EventType.FINANCIAL_HARDSHIP,
    EventType.COMPLAINT_INTENT,
    EventType.STOP_CONTACT_REQUEST,
    EventType.EMOTIONAL_ESCALATION,
}


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("债务" in lowered or "金额" in lowered),
                    float("困难" in lowered or "还不起" in lowered),
                    float("投诉" in lowered or "举报" in lowered),
                    float("停止" in lowered or "拉黑" in lowered),
                    float("情绪" in lowered or "激动" in lowered),
                ]
            )
        return vectors


def test_each_missing_event_has_policy_rule_good_bad_and_boundary_assets():
    rules = json.loads(
        (ROOT / "knowledge" / "rules" / "quality_rules.json").read_text(
            encoding="utf-8"
        )
    )
    documents = []
    for path in (
        ROOT / "knowledge" / "cases" / "good_cases.json",
        ROOT / "knowledge" / "cases" / "bad_cases.json",
        ROOT / "knowledge" / "cases" / "boundary_cases.json",
    ):
        documents.extend(json.loads(path.read_text(encoding="utf-8")))

    policy_metadata = []
    for path in (ROOT / "knowledge" / "policies").glob("*.md"):
        first_line = path.read_text(encoding="utf-8").split("\n", 1)[0]
        policy_metadata.append(json.loads(first_line))

    for event_type in EXPANDED_EVENTS:
        event_value = event_type.value
        matching_rules = [
            rule for rule in rules if event_value in (rule.get("eventTypes") or [])
        ]
        assert len(matching_rules) == 1, event_value
        rule = matching_rules[0]
        assert rule["reviewStatus"] == "PROJECT_DEMO"
        assert rule["automationStatus"] == "AUTO_ELIGIBLE"
        assert rule["penalty"] > 0
        assert rule["sourceDocumentId"] in {
            item["documentId"] for item in policy_metadata
        }
        assert any(
            item["eventType"] == event_value
            and rule["ruleId"] in (item.get("relatedRuleIds") or [])
            for item in policy_metadata
        )
        for category in {"GOOD_CASE", "BAD_CASE", "BOUNDARY_CASE"}:
            assert any(
                item["eventType"] == event_value
                and item["category"] == category
                and rule["ruleId"] in (item.get("relatedRuleIds") or [])
                for item in documents
            ), (event_value, category)


def test_expanded_events_are_retrievable_with_rule_relation():
    index = KnowledgeIndex(ROOT / "knowledge", embedder=FakeEmbedder())
    index.build()
    at_time = datetime(2026, 8, 20, tzinfo=timezone.utc)

    queries = {
        EventType.DEBT_DENIAL: "客户说不是本人借的 要核验债务",
        EventType.AMOUNT_DISPUTE: "客户说金额和逾期费用不对 要核对账单",
        EventType.FINANCIAL_HARDSHIP: "客户说失业没有能力还款 申请困难协商",
        EventType.COMPLAINT_INTENT: "客户说要投诉举报 要登记投诉渠道",
        EventType.STOP_CONTACT_REQUEST: "客户要求停止联系 已经被频繁打扰",
        EventType.EMOTIONAL_ESCALATION: "客户情绪激动抱怨骚扰 需要降级沟通",
    }
    for event_type, query in queries.items():
        hits = index.search(query, event_type, at_time, top_k=5)
        assert hits, event_type
        assert any(hit.metadata["eventType"] == event_type.value for hit in hits)
        assert any(hit.metadata.get("relatedRuleIds") for hit in hits), event_type
