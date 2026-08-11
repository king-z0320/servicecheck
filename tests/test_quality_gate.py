from qc.models import QualityReport, TranscriptTurn, Violation
from qc.quality_gate import QualityGate
from qc.rules import RuleRepository


def make_violation(turn_ids=None, docs=None, penalty=20):
    return Violation(
        ruleId="R006",
        ruleName="还款争议处置",
        penalty=penalty,
        evidenceTurnIds=turn_ids or ["T0001"],
        knowledgeDocumentIds=docs or ["POLICY-REPAYMENT-003"],
        explanation="未经核验直接否定",
        suggestion="登记核验",
    )


def test_gate_rejects_missing_transcript_reference():
    report = QualityReport(
        callId="CALL",
        score=80,
        violations=[make_violation(["T9999"])],
    )
    result = QualityGate(
        RuleRepository("knowledge/rules/quality_rules.json")
    ).check(
        report,
        [
            TranscriptTurn(
                turnId="T0001",
                speaker="客户",
                text="已还完",
                start=0,
                end=1,
            )
        ],
    )
    assert "MISSING_TRANSCRIPT_EVIDENCE" in {
        issue.code for issue in result.issues
    }


def test_gate_returns_unknown_rule_issue_without_crashing():
    violation = Violation(
        ruleId="R999",
        ruleName="不存在",
        penalty=10,
        evidenceTurnIds=["T0001"],
        knowledgeDocumentIds=["UNKNOWN-DOC"],
        explanation="无效规则",
        suggestion="人工复核",
    )
    report = QualityReport(
        callId="CALL",
        score=90,
        violations=[violation],
    )
    result = QualityGate(
        RuleRepository("knowledge/rules/quality_rules.json")
    ).check(
        report,
        [
            TranscriptTurn(
                turnId="T0001",
                speaker="客户",
                text="测试",
                start=0,
                end=1,
            )
        ],
    )
    assert "UNKNOWN_RULE" in {issue.code for issue in result.issues}


def test_gate_rejects_policy_id_that_is_not_in_retrieved_knowledge():
    report = QualityReport(
        callId="CALL",
        score=80,
        violations=[make_violation(docs=["POLICY-FABRICATED-999"])],
    )
    result = QualityGate(
        RuleRepository("knowledge/rules/quality_rules.json")
    ).check(
        report,
        [
            TranscriptTurn(
                turnId="T0001",
                speaker="客户",
                text="已还完",
                start=0,
                end=1,
            )
        ],
    )
    assert "UNKNOWN_POLICY_EVIDENCE" in {
        issue.code for issue in result.issues
    }


def test_gate_recomputes_score_and_rejects_model_penalty():
    report = QualityReport(
        callId="CALL",
        score=1,
        violations=[make_violation(penalty=999)],
    )
    result = QualityGate(
        RuleRepository("knowledge/rules/quality_rules.json")
    ).check(
        report,
        [
            TranscriptTurn(
                turnId="T0001",
                speaker="客户",
                text="已还完",
                start=0,
                end=1,
            )
        ],
    )
    assert "INVALID_PENALTY" in {
        issue.code for issue in result.issues
    }
    assert "INVALID_SCORE" in {
        issue.code for issue in result.issues
    }
