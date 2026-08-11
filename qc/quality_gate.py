from pydantic import BaseModel, Field

from qc.models import QualityReport, TranscriptTurn
from qc.rules import RuleRepository, calculate_score


class GateIssue(BaseModel):
    code: str
    message: str
    reportPath: str


class GateResult(BaseModel):
    passed: bool
    issues: list[GateIssue] = Field(default_factory=list)


class QualityGate:
    def __init__(self, rule_repository: RuleRepository):
        self.rules = rule_repository

    def check(
        self,
        report: QualityReport,
        transcript: list[TranscriptTurn],
    ) -> GateResult:
        issues = []
        valid_turns = {turn.turnId for turn in transcript}
        valid_knowledge_ids = {
            hit.documentId for hit in report.knowledgeHits
        }
        valid_knowledge_ids.update(self.rules.source_document_ids())
        for index, violation in enumerate(report.violations):
            try:
                rule = self.rules.get(violation.ruleId)
            except KeyError:
                issues.append(
                    GateIssue(
                        code="UNKNOWN_RULE",
                        message=f"规则 {violation.ruleId} 不存在",
                        reportPath=f"violations[{index}].ruleId",
                    )
                )
                continue
            if any(
                turn_id not in valid_turns
                for turn_id in violation.evidenceTurnIds
            ):
                issues.append(
                    GateIssue(
                        code="MISSING_TRANSCRIPT_EVIDENCE",
                        message="违规项引用了不存在的转录证据",
                        reportPath=f"violations[{index}].evidenceTurnIds",
                    )
                )
            if not violation.knowledgeDocumentIds:
                issues.append(
                    GateIssue(
                        code="MISSING_POLICY_EVIDENCE",
                        message="违规项没有规则或制度依据",
                        reportPath=(
                            f"violations[{index}].knowledgeDocumentIds"
                        ),
                    )
                )
            elif any(
                document_id not in valid_knowledge_ids
                for document_id in violation.knowledgeDocumentIds
            ):
                issues.append(
                    GateIssue(
                        code="UNKNOWN_POLICY_EVIDENCE",
                        message="违规项引用了未检索到或规则库中不存在的制度依据",
                        reportPath=(
                            f"violations[{index}].knowledgeDocumentIds"
                        ),
                    )
                )
            if violation.penalty != rule.penalty:
                issues.append(
                    GateIssue(
                        code="INVALID_PENALTY",
                        message=f"应使用规则库扣分 {rule.penalty}",
                        reportPath=f"violations[{index}].penalty",
                    )
                )
        known_violations = []
        for violation in report.violations:
            try:
                self.rules.get(violation.ruleId)
            except KeyError:
                continue
            known_violations.append(violation)
        expected_score = calculate_score(
            known_violations,
            self.rules,
        )
        if len(known_violations) == len(report.violations) and report.score != expected_score:
            issues.append(
                GateIssue(
                    code="INVALID_SCORE",
                    message=f"总分应为 {expected_score}",
                    reportPath="score",
                )
            )
        if report.businessFact.status.value != "NOT_CHECKED":
            issues.append(
                GateIssue(
                    code="BUSINESS_FACT_OUT_OF_SCOPE",
                    message="通话质检不得判断客户是否结清",
                    reportPath="businessFact.status",
                )
            )
        return GateResult(
            passed=not issues,
            issues=issues,
        )
