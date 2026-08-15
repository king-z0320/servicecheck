from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field

from qc.models import QualityReport, ReviewDisposition, TranscriptTurn
from qc.rag_support import document_is_active, document_relates_to_rule, supporting_hits
from qc.rules import RuleRepository, calculate_score


EVENT_ID_PATTERN = re.compile(r"^EVT-[0-9A-F]{32}$")


class GateIssue(BaseModel):
    code: str
    message: str
    reportPath: str


class GateResult(BaseModel):
    passed: bool
    issues: list[GateIssue] = Field(default_factory=list)


class QualityGate:
    def __init__(
        self,
        rule_repository: RuleRepository,
        min_support_score: float | None = None,
    ):
        if min_support_score is None:
            from qc.config import calibrated_support_score

            min_support_score = calibrated_support_score()
        if not 0 <= min_support_score <= 1:
            raise ValueError("min_support_score must be between zero and one")
        self.rules = rule_repository
        self.min_support_score = min_support_score

    @staticmethod
    def _issue(code: str, message: str, path: str) -> GateIssue:
        return GateIssue(code=code, message=message, reportPath=path)

    def check(
        self,
        report: QualityReport,
        transcript: list[TranscriptTurn],
        call_started_at: datetime,
        min_support_score: float | None = None,
    ) -> GateResult:
        minimum = self.min_support_score if min_support_score is None else min_support_score
        issues: list[GateIssue] = []
        valid_turns = {turn.turnId for turn in transcript}
        event_by_id = {}
        duplicate_event_ids = set()
        for index, event in enumerate(report.events):
            if event.eventId in event_by_id:
                duplicate_event_ids.add(event.eventId)
                issues.append(
                    self._issue(
                        "DUPLICATE_EVENT_ID",
                        "事件编号重复",
                        f"events[{index}].eventId",
                    )
                )
            else:
                event_by_id[event.eventId] = event
            if not EVENT_ID_PATTERN.fullmatch(event.eventId):
                issues.append(
                    self._issue(
                        "INVALID_EVENT_ID_FORMAT",
                        "事件编号不是后端规范格式",
                        f"events[{index}].eventId",
                    )
                )
            if any(turn_id not in valid_turns for turn_id in event.turnIds):
                issues.append(
                    self._issue(
                        "MISSING_TRANSCRIPT_EVIDENCE",
                        "事件引用了不存在的转写证据",
                        f"events[{index}].turnIds",
                    )
                )

        hits_by_id: dict[str, list] = {}
        for item in report.knowledgeHits:
            hits_by_id.setdefault(item.documentId, []).append(item)

        seen_violation_keys = set()
        scorable = []
        for index, violation in enumerate(report.violations):
            path = f"violations[{index}]"
            key = (
                (violation.eventId, violation.ruleId)
                if violation.eventId
                else (
                    violation.ruleId,
                    tuple(sorted(set(violation.evidenceTurnIds))),
                )
            )
            if key in seen_violation_keys:
                issues.append(
                    self._issue(
                        "DUPLICATE_VIOLATION",
                        "同一事件和规则出现了重复违规",
                        path,
                    )
                )
            seen_violation_keys.add(key)

            event = event_by_id.get(violation.eventId or "")
            if event is None:
                issues.append(
                    self._issue(
                        "MISSING_EVENT_REFERENCE",
                        "违规项没有引用本次报告中的事件",
                        f"{path}.eventId",
                    )
                )
            if any(turn_id not in valid_turns for turn_id in violation.evidenceTurnIds):
                issues.append(
                    self._issue(
                        "MISSING_TRANSCRIPT_EVIDENCE",
                        "违规项引用了不存在的转写证据",
                        f"{path}.evidenceTurnIds",
                    )
                )

            try:
                rule = self.rules.get(violation.ruleId)
            except KeyError:
                issues.append(
                    self._issue(
                        "UNKNOWN_RULE",
                        "违规项引用了不存在的规则",
                        f"{path}.ruleId",
                    )
                )
                continue

            active_rule = (
                self.rules.get_active(
                    violation.ruleId,
                    event.type,
                    call_started_at,
                )
                if event is not None
                else None
            )
            if event is not None and active_rule is None:
                issues.append(
                    self._issue(
                        "INACTIVE_RULE",
                        "规则在通话发生时间或事件类型下不生效",
                        f"{path}.ruleId",
                    )
                )
            if violation.penalty != rule.penalty:
                issues.append(
                    self._issue(
                        "INVALID_PENALTY",
                        f"应使用规则库扣分 {rule.penalty}",
                        f"{path}.penalty",
                    )
                )

            if not violation.knowledgeDocumentIds:
                issues.append(
                    self._issue(
                        "MISSING_POLICY_EVIDENCE",
                        "违规项没有本次检索到的规则支持",
                        f"{path}.knowledgeDocumentIds",
                    )
                )

            candidate_hits = []
            for document_id in violation.knowledgeDocumentIds:
                matching = hits_by_id.get(document_id, [])
                if not matching:
                    issues.append(
                        self._issue(
                            "UNKNOWN_POLICY_EVIDENCE",
                            "违规项引用的知识文档未在本次检索中出现",
                            f"{path}.knowledgeDocumentIds",
                        )
                    )
                    continue
                candidate_hits.extend(matching)
                if event is None:
                    continue
                if all(hit.score < minimum for hit in matching):
                    issues.append(
                        self._issue(
                            "RAG_BELOW_THRESHOLD",
                            "知识命中分数低于支持阈值",
                            f"{path}.knowledgeDocumentIds",
                        )
                    )
                if all(hit.metadata.get("eventType") != event.type.value for hit in matching):
                    issues.append(
                        self._issue(
                            "RAG_EVENT_TYPE_MISMATCH",
                            "知识命中的事件类型与违规事件不一致",
                            f"{path}.knowledgeDocumentIds",
                        )
                    )
                if all(not document_is_active(hit, call_started_at) for hit in matching):
                    issues.append(
                        self._issue(
                            "RAG_DOCUMENT_INACTIVE",
                            "知识文档在通话发生时间无效",
                            f"{path}.knowledgeDocumentIds",
                        )
                    )

            supported = (
                supporting_hits(
                    violation=violation,
                    event=event,
                    rule=rule,
                    hits=candidate_hits,
                    at_time=call_started_at,
                    min_score=minimum,
                )
                if event is not None and active_rule is not None
                else []
            )
            if not supported:
                issues.append(
                    self._issue(
                        "NO_ACTIVE_RULE_SUPPORT",
                        "没有本次实际检索且有效的规则支持",
                        f"{path}.knowledgeDocumentIds",
                    )
                )
            if event is not None:
                scorable.append(violation)

        if len(scorable) == len(report.violations):
            expected_score = calculate_score(
                scorable,
                self.rules,
                call_started_at,
                report.events,
            )
            if report.score != expected_score:
                issues.append(
                    self._issue(
                        "INVALID_SCORE",
                        f"总分应为 {expected_score}",
                        "score",
                    )
                )

        pending = report.summary.get("pendingReviewIssues", [])
        for item in pending:
            code = item.get("code", "NO_ACTIVE_RULE_SUPPORT") if isinstance(item, dict) else str(item)
            issues.append(self._issue(code, "存在尚未自动解决的裁决问题", "summary.pendingReviewIssues"))

        disposition = report.disposition
        if disposition == ReviewDisposition.HUMAN_REVIEW_REQUIRED:
            issues.append(
                self._issue(
                    "DISPOSITION_CONFLICT",
                    "报告仍要求人工复核，不能作为自动终态放行",
                    "disposition",
                )
            )
        if report.violations and disposition == ReviewDisposition.AUTO_PASS:
            issues.append(self._issue("DISPOSITION_CONFLICT", "存在违规时不能自动通过", "disposition"))
        if not report.violations and disposition == ReviewDisposition.AUTO_VIOLATION:
            issues.append(self._issue("DISPOSITION_CONFLICT", "没有违规时不能自动判罚", "disposition"))
        if any(event.ambiguous for event in report.events) and disposition != ReviewDisposition.HUMAN_REVIEW_REQUIRED:
            issues.append(self._issue("DISPOSITION_CONFLICT", "歧义事件必须人工复核", "disposition"))
        if report.auditSnapshot and report.auditSnapshot.errors:
            issues.append(self._issue("AUDIT_ERROR_REQUIRES_REVIEW", "审计信息不完整，必须人工复核", "auditSnapshot.errors"))
            if disposition != ReviewDisposition.HUMAN_REVIEW_REQUIRED:
                issues.append(self._issue("DISPOSITION_CONFLICT", "审计失败时不能自动处置", "disposition"))
        if report.businessFact.status.value != "NOT_CHECKED":
            issues.append(self._issue("BUSINESS_FACT_OUT_OF_SCOPE", "通话质检不得判断客户是否结清", "businessFact.status"))

        return GateResult(passed=not issues, issues=issues)
