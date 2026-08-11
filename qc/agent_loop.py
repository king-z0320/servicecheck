from __future__ import annotations

from datetime import datetime, timezone
from time import monotonic

from pydantic import BaseModel, Field

from qc.context_builder import (
    build_evaluator_user,
    build_evidence_window,
    build_planner_user,
    select_focus_event,
    summarize_observation,
)
from qc.models import (
    AgentTraceEvent,
    QualityReport,
    ReviewDisposition,
    TranscriptTurn,
    Violation,
)
from qc.rules import calculate_score


class LoopContext(BaseModel):
    report: QualityReport
    transcript: list[TranscriptTurn]
    reason: str
    callStartedAt: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    gaps: list[str] = Field(default_factory=list)
    observations: list[dict] = Field(default_factory=list)
    # 工作记忆：焦点与有界证据窗，避免每轮把全文塞进 LLM
    focusEventId: str | None = None
    focusTurnIds: list[str] = Field(default_factory=list)
    evidenceWindow: list[TranscriptTurn] = Field(default_factory=list)
    decisionLog: list[dict] = Field(default_factory=list)
    evidenceRadius: int = 2


class LoopResult(BaseModel):
    status: str
    report: QualityReport
    iterations: int
    toolCalls: int
    trace: list[AgentTraceEvent]


class BoundedAgentLoop:
    def __init__(
        self,
        planner,
        executor,
        evaluator,
        quality_gate,
        max_iterations=3,
        max_tool_calls=8,
        timeout_seconds=90,
    ):
        self.planner = planner
        self.executor = executor
        self.evaluator = evaluator
        self.quality_gate = quality_gate
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.timeout_seconds = timeout_seconds

    def run(self, context: LoopContext) -> LoopResult:
        self._ensure_working_memory(context)
        trace = []
        tool_calls = 0
        started = monotonic()
        iterations = 0
        no_tool_actions = {"REVISE_REPORT", "FINALIZE"}

        for iteration in range(1, self.max_iterations + 1):
            if (
                monotonic() - started >= self.timeout_seconds
                or tool_calls >= self.max_tool_calls
            ):
                break
            iterations = iteration
            decision = self.planner.decide(context)
            trace.append(
                AgentTraceEvent(
                    iteration=iteration,
                    phase="PLAN",
                    message=decision["type"],
                    details=decision,
                )
            )
            observation = self.executor.execute(decision, context)
            if decision["type"] not in no_tool_actions:
                tool_calls += 1
            # 完整观察进轨迹；工作记忆只留摘要，防止上下文膨胀
            context.observations.append(summarize_observation(observation))
            context.decisionLog.append(
                {
                    "iteration": iteration,
                    "action": decision.get("type"),
                    "reason": decision.get("reason", ""),
                    "summary": summarize_observation(observation),
                }
            )
            trace.append(
                AgentTraceEvent(
                    iteration=iteration,
                    phase="ACT",
                    message=decision["type"],
                    details=decision,
                )
            )
            trace.append(
                AgentTraceEvent(
                    iteration=iteration,
                    phase="OBSERVE",
                    message="行动结果已记录",
                    details=observation,
                )
            )
            gate_result = self.quality_gate.check(
                context.report,
                context.transcript,
            )
            evaluation = self.evaluator.evaluate(
                context,
                gate_result,
            )
            trace.append(
                AgentTraceEvent(
                    iteration=iteration,
                    phase="EVALUATE",
                    message=evaluation["verdict"],
                    details=evaluation,
                )
            )
            if gate_result.passed and evaluation["verdict"] == "PASS":
                trace.append(
                    AgentTraceEvent(
                        iteration=iteration,
                        phase="FINALIZE",
                        message="质量门禁与Evaluator均通过",
                    )
                )
                return LoopResult(
                    status="COMPLETED",
                    report=context.report,
                    iterations=iteration,
                    toolCalls=tool_calls,
                    trace=trace,
                )
            if evaluation["verdict"] == "HUMAN_REVIEW_REQUIRED":
                break
            context.gaps = list(evaluation.get("issues", []))
            trace.append(
                AgentTraceEvent(
                    iteration=iteration,
                    phase="REPLAN",
                    message="根据评估差距重新规划",
                    details={"gaps": context.gaps},
                )
            )

        return LoopResult(
            status="HUMAN_REVIEW_REQUIRED",
            report=context.report,
            iterations=iterations,
            toolCalls=tool_calls,
            trace=trace,
        )

    @staticmethod
    def _ensure_working_memory(context: LoopContext) -> None:
        focus = select_focus_event(context.report)
        if focus and not context.focusEventId:
            context.focusEventId = focus.eventId
        if focus and not context.focusTurnIds:
            context.focusTurnIds = list(focus.turnIds)
        if not context.evidenceWindow:
            context.evidenceWindow = build_evidence_window(
                context.transcript,
                context.focusTurnIds,
                radius=context.evidenceRadius,
            )


class LLMPlanner:
    def __init__(self, gateway):
        self.gateway = gateway

    def decide(self, context: LoopContext) -> dict:
        return self.gateway.complete_json(
            system=(
                "你是复杂催收质检案件的规划器。每轮只能根据输入中的案件卡、"
                "焦点事件、转录证据窗、gaps、知识与审计摘要、最近观察和决策"
                "记录选择一个动作。不得假设未提供的完整通话或业务事实，不得"
                "查询或判断客户余额、是否真实还款或是否已经结清。"
                "必须按照下列动作契约选择type："
                "1. EXPAND_CONTEXT：仅当gaps表明缺少焦点事件前后转录，而且多看"
                "相邻对话可能解决问题时选择。该动作每次只扩大一圈、最大半径为"
                "6，不查询知识、不查询审计，也不修改报告。若最近观察显示重复"
                "扩窗没有增加新的windowTurnIds，不要再次选择。"
                "2. SEARCH_KNOWLEDGE：仅当缺少规则、制度或案例依据，或者规则适用"
                "条件和版本不明确时选择。该动作最多检索5条知识并合并到报告，"
                "不会直接新增、删除或修改违规。由于当前执行器会把reason直接"
                "作为检索query，选择该动作时reason必须写成具体、可直接搜索的"
                "质检问题，不要只写‘需要补充知识’之类的泛化理由。"
                "3. QUERY_ACTION_AUDIT：仅当审计快照缺失、auditSummary中存在"
                "errors，或gaps要求核实CRM小结、争议工单、跟进任务、坐席操作时"
                "选择。该动作会重新获取完整审计快照，不得用于查询客户余额。"
                "4. REVISE_REPORT：仅当gaps指出报告存在当前结构化修订能够处理的"
                "问题时选择，例如无效转录证据、错误总分、业务事实越界或错误"
                "处置状态。该动作可以删除证据无效的违规、按规则库重算总分并"
                "调整处置状态；不能新增违规、修改规则编号、修正单条违规扣分、"
                "补写知识文档编号或自由改写解释。"
                "5. FINALIZE：仅当gaps为空，现有证据、知识和审计信息已经足以"
                "支持当前报告，并且继续执行其他动作不会获得必要的新信息时选择。"
                "该动作本身不修改报告，之后仍须通过确定性质量门禁和独立评估。"
                "选择时优先解决gaps中最具体且当前工具可以处理的问题；结合最近"
                "观察和决策记录，避免重复没有产生新信息的动作。reason必须简洁"
                "说明本轮动作针对的具体问题；选择SEARCH_KNOWLEDGE时遵守上述"
                "query要求。不要输出隐藏推理，只返回符合JSON Schema的结果。"
            ),
            user=build_planner_user(context),
            schema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "EXPAND_CONTEXT",
                            "SEARCH_KNOWLEDGE",
                            "QUERY_ACTION_AUDIT",
                            "REVISE_REPORT",
                            "FINALIZE",
                        ],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["type", "reason"],
                "additionalProperties": False,
            },
        )


class LLMEvaluator:
    def __init__(self, gateway):
        self.gateway = gateway

    def evaluate(self, context: LoopContext, gate_result) -> dict:
        return self.gateway.complete_json(
            system=(
                "你是催收通话质检的独立评估器。你的任务不是规划动作，也不能直接"
                "修改报告；你只判断当前报告是否已被现有证据充分支持，并指出仍未"
                "解决的具体问题。只依据输入中的案件卡、转录证据窗、知识命中、"
                "审计快照、gaps和确定性质量门禁，不得假设未提供的完整通话或业务"
                "事实，不得确认客户是否已经真实还款或结清。"
                "必须按以下标准选择verdict："
                "1. PASS：仅当质量门禁passed=true、现有报告与证据一致，且没有"
                "必须继续补充或修订的关键问题时返回；此时issues必须为空。"
                "2. NEEDS_MORE_CONTEXT：仅当缺少焦点事件前后转录，而扩大证据窗"
                "可能解决问题时返回；issues必须说明缺少什么上下文，并建议"
                "EXPAND_CONTEXT。"
                "3. RULE_AMBIGUITY：仅当转录证据基本充分，但规则或制度的适用"
                "条件、版本、知识依据仍不明确时返回；issues必须指出具体歧义，"
                "必要时建议SEARCH_KNOWLEDGE。"
                "4. REPORT_REVISION_REQUIRED：仅当证据已经充分，但报告存在当前"
                "结构化修订动作能够处理的问题时返回，例如不存在的转录证据、"
                "错误总分、业务事实越界或错误处置状态；issues必须指出具体报告"
                "字段，并建议REVISE_REPORT。不要假设该动作能够新增违规、修改"
                "规则编号、修正单条违规扣分或自由改写解释。"
                "5. HUMAN_REVIEW_REQUIRED：当现有工具无法修复问题、关键审计数据"
                "仍然失败、规则冲突无法自动消除、关键证据无法取得，或继续循环"
                "也不可能得到可靠结论时返回。"
                "如果质量门禁passed=false，不得返回PASS。非PASS结果的issues"
                "不得为空。每条issue都要具体说明问题、缺少的证据或错误字段，"
                "以及建议的下一步处理方向。不要输出隐藏推理，只返回符合JSON "
                "Schema的结果。"
            ),
            user=build_evaluator_user(context, gate_result),
            schema={
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "PASS",
                            "NEEDS_MORE_CONTEXT",
                            "RULE_AMBIGUITY",
                            "REPORT_REVISION_REQUIRED",
                            "HUMAN_REVIEW_REQUIRED",
                        ],
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["verdict", "issues"],
                "additionalProperties": False,
            },
        )


class QualityLoopExecutor:
    def __init__(self, knowledge_index, audit_client, rule_repository=None):
        self.knowledge_index = knowledge_index
        self.audit_client = audit_client
        self.rule_repository = rule_repository

    def execute(self, decision: dict, context: LoopContext) -> dict:
        action = decision["type"]
        if action == "EXPAND_CONTEXT":
            return self._expand_context(context, decision)
        if action == "SEARCH_KNOWLEDGE":
            return self._search_knowledge(context, decision)
        if action == "QUERY_ACTION_AUDIT":
            snapshot = self.audit_client.fetch_snapshot(context.report.callId)
            context.report.auditSnapshot = snapshot
            return {
                "action": action,
                "audit": snapshot.model_dump(mode="json"),
            }
        if action == "REVISE_REPORT":
            return self._revise_report(context, decision)
        if action == "FINALIZE":
            return {"action": action}
        raise ValueError(f"unsupported loop action: {action}")

    def _focus_event(self, context: LoopContext):
        if context.focusEventId:
            for event in context.report.events:
                if event.eventId == context.focusEventId:
                    return event
        return select_focus_event(context.report)

    def _expand_context(self, context: LoopContext, decision: dict) -> dict:
        """扩大焦点证据窗（有界），不把全文塞回 observations。"""
        radius = int(decision.get("radius") or context.evidenceRadius or 2)
        radius = min(max(radius + 1, 1), 6)  # 每调用扩大一圈，封顶
        context.evidenceRadius = radius
        focus = self._focus_event(context)
        turn_ids = list(context.focusTurnIds)
        if focus and not turn_ids:
            turn_ids = list(focus.turnIds)
            context.focusTurnIds = turn_ids
        window = build_evidence_window(context.transcript, turn_ids, radius=radius)
        context.evidenceWindow = window
        return {
            "action": "EXPAND_CONTEXT",
            "radius": radius,
            "windowTurnIds": [t.turnId for t in window],
            "turns": [t.model_dump(mode="json") for t in window],
        }

    def _search_knowledge(self, context: LoopContext, decision: dict) -> dict:
        focus = self._focus_event(context)
        if focus is None:
            return {"action": "SEARCH_KNOWLEDGE", "hits": []}
        query = decision.get("reason") or focus.statement
        hits = self.knowledge_index.search(
            query,
            focus.type,
            context.callStartedAt,
            top_k=5,
        )
        # 合并而非只保留本轮，按 documentId 去重
        existing = {h.documentId: h for h in context.report.knowledgeHits}
        for hit in hits:
            existing[hit.documentId] = hit
        context.report.knowledgeHits = list(existing.values())
        return {
            "action": "SEARCH_KNOWLEDGE",
            "focusEventId": focus.eventId,
            "hits": [hit.model_dump(mode="json") for hit in hits],
        }

    def _revise_report(self, context: LoopContext, decision: dict) -> dict:
        """有界修订：可清除错误违规、用规则库重算分、调整 disposition。

        不调用 LLM 自由改写全文；复杂语义修订由 Evaluator 驱动 gaps，
        本动作落实「可执行的结构化修订」。
        """
        report = context.report
        before = len(report.violations)
        # 丢弃证据 turn 不在全文中的违规
        valid_turns = {t.turnId for t in context.transcript}
        report.violations = [
            v
            for v in report.violations
            if v.evidenceTurnIds
            and all(tid in valid_turns for tid in v.evidenceTurnIds)
        ]
        # 若 gaps 要求修订且仍无违规、但审计明确失败，保持人工
        report.score = calculate_score(report.violations, self.rule_repository)

        if report.businessFact.status.value != "NOT_CHECKED":
            from qc.models import BusinessFact, ClaimFactStatus

            report.businessFact = BusinessFact(status=ClaimFactStatus.NOT_CHECKED)

        if report.violations:
            report.disposition = ReviewDisposition.AUTO_VIOLATION
        elif report.auditSnapshot and report.auditSnapshot.errors:
            report.disposition = ReviewDisposition.HUMAN_REVIEW_REQUIRED
        elif any(e.ambiguous for e in report.events):
            report.disposition = ReviewDisposition.HUMAN_REVIEW_REQUIRED
        else:
            report.disposition = ReviewDisposition.AUTO_PASS

        return {
            "action": "REVISE_REPORT",
            "revised": True,
            "violationCount": len(report.violations),
            "removedInvalidEvidence": before - len(report.violations),
            "disposition": report.disposition.value,
            "score": report.score,
            "reason": decision.get("reason", ""),
        }
