from datetime import datetime, timezone

import pytest

from qc.models import EventType, Violation
from qc.rules import RuleRepository, calculate_score


def test_loads_only_active_rules_for_event():
    repo = RuleRepository("knowledge/rules/quality_rules.json")
    rules = repo.load_active(
        EventType.REPAYMENT_DISPUTE,
        datetime(2026, 7, 27, tzinfo=timezone.utc),
    )
    assert {rule.ruleId for rule in rules} == {"R006"}


def test_score_uses_repository_penalty_not_model_penalty():
    repo = RuleRepository("knowledge/rules/quality_rules.json")
    violation = Violation(
        ruleId="R006",
        ruleName="还款争议处置",
        penalty=999,
        evidenceTurnIds=["T0001"],
        knowledgeDocumentIds=["POLICY-REPAYMENT-003"],
        explanation="未经核验直接否定客户",
        suggestion="登记并发起核查",
    )
    assert calculate_score([violation], repo) == 80


def test_multiple_violations_with_same_penalty_are_each_counted():
    repo = RuleRepository("knowledge/rules/quality_rules.json")
    violations = [
        Violation(
            ruleId="R006",
            ruleName="还款争议处置",
            penalty=20,
            evidenceTurnIds=[f"T{index:04d}"],
            knowledgeDocumentIds=["POLICY-REPAYMENT-003"],
            explanation="两次独立违规事件",
            suggestion="分别登记核验",
        )
        for index in (1, 2)
    ]
    assert calculate_score(violations, repo) == 60


def test_unknown_rule_cannot_be_scored():
    repo = RuleRepository("knowledge/rules/quality_rules.json")
    violation = Violation(
        ruleId="R999",
        ruleName="不存在",
        penalty=1,
        evidenceTurnIds=["T0001"],
        knowledgeDocumentIds=["DOC"],
        explanation="x",
        suggestion="y",
    )
    with pytest.raises(KeyError, match="R999"):
        calculate_score([violation], repo)
