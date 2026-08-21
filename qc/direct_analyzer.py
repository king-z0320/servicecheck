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
DEBT_DENIAL_TERMS = (
    "别装了",
    "就是你借的",
    "你在撒谎",
    "恶意逃债",
    "诈骗",
)
AMOUNT_DISPUTE_TERMS = (
    "系统不可能算错",
    "金额不可能错",
    "必须按这个金额",
    "你看错了",
    "不可能算错",
)
FINANCIAL_HARDSHIP_TERMS = (
    "没钱也要还",
    "去借钱还",
    "借新还旧",
    "自己想办法",
    "不接受协商",
    "一定减免",
    "肯定减免",
)
COMPLAINT_INTENT_TERMS = (
    "随便你投诉",
    "投诉也没用",
    "不怕你投诉",
    "别投诉",
    "投诉会影响",
    "你去投诉试试",
)
STOP_CONTACT_REQUEST_TERMS = (
    "拉黑也没用",
    "还会继续打",
    "会一直打",
    "换个号码",
    "联系你家人",
    "不可能不打",
)
EMOTIONAL_ESCALATION_TERMS = (
    "你吵什么",
    "烦的是你",
    "激动什么",
    "闭嘴",
    "有病",
    "发什么疯",
    "爱咋咋地",
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
            elif event.type == EventType.DEBT_DENIAL:
                violation = self._check_agent_language(
                    event,
                    request.transcript,
                    turn_by_id,
                    rule_id="R003",
                    bad_terms=DEBT_DENIAL_TERMS,
                    explanation="客户提出债务或身份异议后，坐席未经核验直接指认客户欺诈或恶意逃债。",
                    suggestion="记录客户异议并进入身份/债务核验流程，不得先行认定欺诈或恶意逃债。",
                )
            elif event.type == EventType.AMOUNT_DISPUTE:
                violation = self._check_agent_language(
                    event,
                    request.transcript,
                    turn_by_id,
                    rule_id="R004",
                    bad_terms=AMOUNT_DISPUTE_TERMS,
                    explanation="客户提出金额或费用争议后，坐席未经核验将争议金额作为确定事实并施压。",
                    suggestion="复述争议字段、查询权威账务并登记异议，不得强行要求按未核实金额处理。",
                )
            elif event.type == EventType.FINANCIAL_HARDSHIP:
                violation = self._check_agent_language(
                    event,
                    request.transcript,
                    turn_by_id,
                    rule_id="R008",
                    bad_terms=FINANCIAL_HARDSHIP_TERMS,
                    explanation="客户表达还款困难后，坐席以强迫借款、拒绝协商或未授权承诺等方式施压。",
                    suggestion="记录困难与协商诉求，告知正式申请渠道，按审批结果沟通，不得强迫借新还旧或擅自承诺减免。",
                )
            elif event.type == EventType.COMPLAINT_INTENT:
                violation = self._check_agent_language(
                    event,
                    request.transcript,
                    turn_by_id,
                    rule_id="R009",
                    bad_terms=COMPLAINT_INTENT_TERMS,
                    explanation="客户表达投诉或举报意图后，坐席阻拦、讥讽或以投诉为由继续施压。",
                    suggestion="确认并记录投诉事项，提供正式投诉渠道，不得阻拦、讥讽或威胁客户放弃投诉。",
                )
            elif event.type == EventType.STOP_CONTACT_REQUEST:
                violation = self._check_agent_language(
                    event,
                    request.transcript,
                    turn_by_id,
                    rule_id="R007",
                    bad_terms=STOP_CONTACT_REQUEST_TERMS,
                    explanation="客户明确提出停止联系后，坐席表示将绕过请求继续通过其他渠道或号码施压。",
                    suggestion="记录停联请求并按停联、降频或升级流程处理，不得改用第三方或其他号码继续施压。",
                )
            elif event.type == EventType.EMOTIONAL_ESCALATION:
                violation = self._check_agent_language(
                    event,
                    request.transcript,
                    turn_by_id,
                    rule_id="R010",
                    bad_terms=EMOTIONAL_ESCALATION_TERMS,
                    explanation="客户情绪升级时，坐席以嘲讽、争吵或辱骂回应并进一步激化对抗。",
                    suggestion="降低对抗、确认客户诉求并按升级流程转交，禁止嘲讽、争吵或辱骂。",
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

    def _check_agent_language(
        self,
        event,
        transcript,
        turn_by_id,
        *,
        rule_id,
        bad_terms,
        explanation,
        suggestion,
    ):
        """Check only the agent's response to an extracted customer event.

        The event itself is not a violation: a customer may legitimately deny a
        debt, dispute an amount, ask to stop contact, complain, report hardship,
        or become upset.  A deterministic violation requires a matching agent
        utterance in the event context or after it.
        """
        response_turns = self._agent_turns_after(
            transcript, turn_by_id, event.turnIds
        )
        for turn_id in event.turnIds:
            turn = turn_by_id.get(turn_id)
            if turn and turn.speaker == "坐席" and turn not in response_turns:
                response_turns.insert(0, turn)
        hit_turns = [
            turn
            for turn in response_turns
            if any(term in turn.text for term in bad_terms)
        ]
        if not hit_turns:
            return None
        rule = self.rule_repository.get(rule_id)
        evidence_turn_ids = list(event.turnIds)
        evidence_turn_ids.extend(
            turn.turnId
            for turn in hit_turns
            if turn.turnId not in evidence_turn_ids
        )
        return Violation(
            eventId=event.eventId,
            ruleId=rule.ruleId,
            ruleName=rule.name,
            penalty=rule.penalty,
            evidenceTurnIds=evidence_turn_ids,
            knowledgeDocumentIds=[],
            explanation=explanation,
            suggestion=suggestion,
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
