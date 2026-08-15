import json
from datetime import datetime, timezone

import pytest

from qc.models import EventType, QualityEvent, Violation
from qc.rules import RuleRepository, calculate_score, deduplicate_violations


AT = datetime(2026, 7, 27, tzinfo=timezone.utc)


def event(event_id, turn_id="T0001"):
    return QualityEvent(
        eventId=event_id,
        type=EventType.REPAYMENT_DISPUTE,
        statement="客户称已还款",
        turnIds=[turn_id],
        confidence=0.9,
        ambiguous=False,
    )


def violation(event_id, turn_id="T0001", penalty=999):
    return Violation(
        eventId=event_id,
        ruleId="R006",
        ruleName="还款争议处置",
        penalty=penalty,
        evidenceTurnIds=[turn_id],
        knowledgeDocumentIds=["POLICY-REPAYMENT-003"],
        explanation="未经核验直接否定客户",
        suggestion="登记并发起核查",
    )


def test_loads_only_active_rules_for_event():
    repo = RuleRepository("knowledge/rules/quality_rules.json")
    rules = repo.load_active(EventType.REPAYMENT_DISPUTE, AT)
    assert {rule.ruleId for rule in rules} == {"R006"}
    assert repo.get_active("R006", EventType.REPAYMENT_DISPUTE, AT).ruleId == "R006"


def test_rule_effective_interval_is_start_inclusive_end_exclusive(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps(
            [
                {
                    "ruleId": "R-TIME",
                    "name": "时效规则",
                    "eventTypes": ["REPAYMENT_DISPUTE"],
                    "description": "test",
                    "penalty": 10,
                    "version": "1",
                    "effectiveFrom": "2025-01-01T00:00:00Z",
                    "effectiveTo": "2025-02-01T00:00:00Z",
                    "sourceDocumentId": "POLICY-TIME",
                }
            ]
        ),
        encoding="utf-8",
    )
    repo = RuleRepository(path)

    assert repo.get_active(
        "R-TIME",
        EventType.REPAYMENT_DISPUTE,
        datetime(2025, 1, 1, tzinfo=timezone.utc),
    ) is not None
    assert repo.get_active(
        "R-TIME",
        EventType.REPAYMENT_DISPUTE,
        datetime(2025, 2, 1, tzinfo=timezone.utc),
    ) is None


def test_score_uses_active_repository_penalty_not_candidate_penalty():
    repo = RuleRepository("knowledge/rules/quality_rules.json")
    item = violation("EVT-" + "A" * 32)
    assert calculate_score([item], repo, AT, [event(item.eventId)]) == 80


def test_duplicate_same_event_rule_is_deducted_once():
    repo = RuleRepository("knowledge/rules/quality_rules.json")
    event_id = "EVT-" + "A" * 32
    items = [violation(event_id), violation(event_id)]

    assert len(deduplicate_violations(items)) == 1
    assert calculate_score(items, repo, AT, [event(event_id)]) == 80


def test_independent_same_rule_events_are_each_deducted():
    repo = RuleRepository("knowledge/rules/quality_rules.json")
    event_ids = ["EVT-" + "A" * 32, "EVT-" + "B" * 32]
    items = [violation(event_ids[0], "T0001"), violation(event_ids[1], "T0002")]
    events = [event(event_ids[0], "T0001"), event(event_ids[1], "T0002")]

    assert calculate_score(items, repo, AT, events) == 60


def test_inactive_rule_does_not_reduce_score():
    repo = RuleRepository("knowledge/rules/quality_rules.json")
    event_id = "EVT-" + "A" * 32
    before_effective = datetime(2024, 12, 31, tzinfo=timezone.utc)

    assert calculate_score(
        [violation(event_id)],
        repo,
        before_effective,
        [event(event_id)],
    ) == 100


def test_unknown_rule_cannot_be_scored():
    repo = RuleRepository("knowledge/rules/quality_rules.json")
    item = violation("EVT-" + "A" * 32)
    item.ruleId = "R999"
    with pytest.raises(KeyError, match="R999"):
        calculate_score([item], repo, AT, [event(item.eventId)])
