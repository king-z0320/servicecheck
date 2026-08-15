from time import monotonic

from qc.models import (
    AnalysisRequest,
    EventType,
    QualityReport,
    ReviewDisposition,
    TranscriptTurn,
    Violation,
)
from qc.rules import calculate_score
from qc.rag_support import supporting_hits
from qc.rules import deduplicate_violations

# 明确路径用的浅层话术特征（可解释；生产可换成分类器）
DENIAL_TERMS = ("不可能", "你没还", "必须处理", "你在撒谎", "别装")
THREAT_TERMS = (
    "法院抓人",
    "马上坐牢",
    "给你送进去",
    "公安找你",
    "把你抓起来",
    "后果很严重马上",
    "上门砸",
)
PRIVACY_LEAK_TERMS = (
    "欠了",
    "欠款",
    "逾期",
    "借款",
    "平台贷",
    "多少钱没还",
    "债务",
)


class DirectAnalyzer:
    def __init__(
        self,
        extractor,
        knowledge_index,
        rule_repository,
        audit_client,
        min_support_score: float | None = None,
    ):
        self.extractor = extractor
        self.knowledge_index = knowledge_index
        self.rule_repository = rule_repository
        self.audit_client = audit_client
        if min_support_score is None:
            from qc.config import calibrated_support_score

            min_support_score = calibrated_support_score()
        if not 0 <= min_support_score <= 1:
            raise ValueError("min_support_score must be between zero and one")
        self.min_support_score = min_support_score
        self.last_stage_times = {
            "eventExtraction": 0.0,
            "knowledgeRetrieval": 0.0,
            "actionAudit": 0.0,
        }

    def analyze(self, request: AnalysisRequest) -> QualityReport:
        extraction_started = monotonic()
        events = self.extractor.extract(request)
        self.last_stage_times["eventExtraction"] = (
            monotonic() - extraction_started
        )

        audit_started = monotonic()
        audit = self.audit_client.fetch_snapshot(request.callId)
        self.last_stage_times["actionAudit"] = monotonic() - audit_started
        knowledge_seconds = 0.0
        hits = []
        violations = []
        pending_review_issues = []
        turn_by_id = {
            turn.turnId: turn
            for turn in request.transcript
        }

        for event in events:
            knowledge_started = monotonic()
            event_hits = self.knowledge_index.search(
                event.statement,
                event.type,
                request.callStartedAt,
                top_k=5,
            )
            knowledge_seconds += monotonic() - knowledge_started
            hits.extend(event_hits)
            if event.ambiguous:
                continue

            if event.type == EventType.REPAYMENT_DISPUTE:
                violation = self._check_repayment_dispute(
                    event, request.transcript, turn_by_id, audit
                )
            elif event.type == EventType.THREAT_OR_COERCION:
                violation = self._check_threat(event, request.transcript, turn_by_id)
            elif event.type == EventType.THIRD_PARTY_CONTACT:
                violation = self._check_third_party(
                    event, request.transcript, turn_by_id
                )
            else:
                violation = None

            if violation is None:
                continue
            rule = self.rule_repository.get_active(
                violation.ruleId,
                event.type,
                request.callStartedAt,
            )
            if rule is None:
                pending_review_issues.append(
                    {
                        "code": "NO_ACTIVE_RULE_SUPPORT",
                        "eventId": event.eventId,
                        "ruleId": violation.ruleId,
                    }
                )
                continue
            supported = supporting_hits(
                violation=violation,
                event=event,
                rule=rule,
                hits=event_hits,
                at_time=request.callStartedAt,
                min_score=self.min_support_score,
            )
            if not supported:
                pending_review_issues.append(
                    {
                        "code": "RAG_WEAK_SUPPORT",
                        "eventId": event.eventId,
                        "ruleId": violation.ruleId,
                    }
                )
                continue
            violation.knowledgeDocumentIds = [
                item.documentId for item in supported
            ]
            violations.append(violation)

        self.last_stage_times["knowledgeRetrieval"] = knowledge_seconds
        unique_hits = {}
        for item in hits:
            current = unique_hits.get(item.documentId)
            if current is None or item.score > current.score:
                unique_hits[item.documentId] = item
        event_order = {
            event.eventId: index for index, event in enumerate(events)
        }
        violations = deduplicate_violations(violations, event_order)
        report = QualityReport(
            callId=request.callId,
            events=events,
            violations=violations,
            knowledgeHits=list(unique_hits.values()),
            auditSnapshot=audit,
            summary={"pendingReviewIssues": pending_review_issues},
        )
        report.score = calculate_score(
            violations,
            self.rule_repository,
            request.callStartedAt,
            events,
        )
        if (
            any(event.ambiguous for event in events)
            or audit.errors
            or pending_review_issues
        ):
            report.disposition = ReviewDisposition.HUMAN_REVIEW_REQUIRED
        elif violations:
            report.disposition = ReviewDisposition.AUTO_VIOLATION
        return report

    def _agent_turns_after(
        self,
        transcript: list[TranscriptTurn],
        turn_by_id: dict,
        event_turn_ids: list[str],
    ) -> list[TranscriptTurn]:
        event_end = max(
            turn_by_id[tid].end for tid in event_turn_ids if tid in turn_by_id
        )
        return [
            turn
            for turn in transcript
            if turn.speaker == "坐席" and turn.start >= event_end
        ]

    def _check_repayment_dispute(self, event, transcript, turn_by_id, audit):
        response_turns = self._agent_turns_after(
            transcript, turn_by_id, event.turnIds
        )
        denial_turns = [
            turn
            for turn in response_turns
            if any(term in turn.text for term in DENIAL_TERMS)
        ]
        direct_denial = bool(denial_turns)
        post_call_failure = audit.disputeTicketCreated is False
        if not (direct_denial or post_call_failure):
            return None
        rule = self.rule_repository.get("R006")
        evidence_turn_ids = list(event.turnIds)
        evidence_turn_ids.extend(
            turn.turnId
            for turn in denial_turns
            if turn.turnId not in evidence_turn_ids
        )
        return Violation(
            eventId=event.eventId,
            ruleId=rule.ruleId,
            ruleName=rule.name,
            penalty=rule.penalty,
            evidenceTurnIds=evidence_turn_ids,
            knowledgeDocumentIds=[],
            explanation=(
                "客户提出还款争议后，坐席处置或通话后登记不符合规范。"
            ),
            suggestion=(
                "确认还款时间、金额和渠道，准确记录客户主张并创建核验工单。"
            ),
        )

    def _check_threat(self, event, transcript, turn_by_id):
        candidate_ids = set(event.turnIds)
        for turn in self._agent_turns_after(transcript, turn_by_id, event.turnIds):
            candidate_ids.add(turn.turnId)
        hit_turns = [
            turn
            for turn in transcript
            if turn.turnId in candidate_ids
            and turn.speaker == "坐席"
            and any(term in turn.text for term in THREAT_TERMS)
        ]
        if not hit_turns and event.statement and any(
            t in event.statement for t in THREAT_TERMS
        ):
            hit_turns = [
                turn_by_id[tid]
                for tid in event.turnIds
                if tid in turn_by_id and turn_by_id[tid].speaker == "坐席"
            ]
        if not hit_turns:
            return None
        rule = self.rule_repository.get("R002")
        return Violation(
            eventId=event.eventId,
            ruleId=rule.ruleId,
            ruleName=rule.name,
            penalty=rule.penalty,
            evidenceTurnIds=[t.turnId for t in hit_turns],
            knowledgeDocumentIds=[],
            explanation="坐席使用威胁、恐吓或夸大法律/征信后果的话术。",
            suggestion="仅客观说明合同约定与可能流程，禁止确定性恐吓表述。",
        )

    def _check_third_party(self, event, transcript, turn_by_id):
        response_turns = self._agent_turns_after(
            transcript, turn_by_id, event.turnIds
        )
        for tid in event.turnIds:
            turn = turn_by_id.get(tid)
            if turn and turn.speaker == "坐席" and turn not in response_turns:
                response_turns.insert(0, turn)
        leak_turns = [
            turn
            for turn in response_turns
            if any(term in turn.text for term in PRIVACY_LEAK_TERMS)
        ]
        if not leak_turns:
            return None
        rule = self.rule_repository.get("R005")
        return Violation(
            eventId=event.eventId,
            ruleId=rule.ruleId,
            ruleName=rule.name,
            penalty=rule.penalty,
            evidenceTurnIds=[t.turnId for t in leak_turns],
            knowledgeDocumentIds=[],
            explanation="未确认债务人身份前，向第三方披露了债务相关信息。",
            suggestion="仅可请求转告回电，不得透露欠款金额、逾期或平台信息。",
        )


def requires_loop(
    report: QualityReport,
    gate_result=None,
) -> tuple[bool, str | None]:
    if any(event.ambiguous for event in report.events):
        return True, "AMBIGUOUS_EVENT"
    if report.auditSnapshot and report.auditSnapshot.errors:
        return True, "PARTIAL_AUDIT_FAILURE"
    if report.summary.get("pendingReviewIssues"):
        return True, "RAG_WEAK_SUPPORT"
    if gate_result is not None and not gate_result.passed:
        return True, "QUALITY_GATE_FAILED"
    return False, None
