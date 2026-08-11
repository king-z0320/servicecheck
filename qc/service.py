import json
from time import monotonic
from uuid import uuid4

from qc.agent_loop import LoopContext
from qc.direct_analyzer import requires_loop
from qc.models import AgentTraceEvent, AnalysisRequest, AnalysisResult


class QualityAnalysisService:
    def __init__(self, direct_analyzer, agent_loop, run_store):
        self.direct_analyzer = direct_analyzer #direct_analyzer 是一个 DirectAnalyzer 对象，用于直接分析质检请求，生成初步的质检报告。
        self.agent_loop = agent_loop #agent_loop 是一个 AgentLoop 对象，用于处理复杂的质检情况，通常在直接分析无法得出明确结论时启动。
        self.run_store = run_store #run_store 是一个持久化存储对象，用于保存质检运行的结果和事件追踪信息。

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        started = monotonic()
        stage_times = {
            "eventExtraction": 0.0,
            "knowledgeRetrieval": 0.0,
            "actionAudit": 0.0,
            "directAnalysis": 0.0,
            "agentLoop": 0.0,
            "persistence": 0.0,
        }
        run_id = f"RUN-{uuid4().hex[:12].upper()}"

        persistence_started = monotonic()
        self.run_store.create_run(run_id, request)
        stage_times["persistence"] += monotonic() - persistence_started

        direct_started = monotonic()
        report = self.direct_analyzer.analyze(request)
        stage_times["directAnalysis"] = monotonic() - direct_started
        stage_times.update(
            getattr(self.direct_analyzer, "last_stage_times", {})
        )
        use_loop, reason = requires_loop(report)
        trace = []
        status = "COMPLETED"
        loop_iterations = 0

        if use_loop:
            loop_started = monotonic()
            loop_result = self.agent_loop.run(
                LoopContext(
                    report=report,
                    transcript=request.transcript,
                    callStartedAt=request.callStartedAt,
                    reason=reason or "COMPLEX_CASE",
                )
            )
            stage_times["agentLoop"] = monotonic() - loop_started
            loop_iterations = loop_result.iterations
            report = loop_result.report
            trace = loop_result.trace
            status = (
                "COMPLETED"
                if loop_result.status == "COMPLETED"
                else "PARTIAL"
            )
            persistence_started = monotonic()
            for event in trace:
                self.run_store.append_event(run_id, event)
            stage_times["persistence"] += monotonic() - persistence_started
        else:
            event = AgentTraceEvent(
                iteration=0,
                phase="FINALIZE",
                message="Agent Loop未启动：规则明确、证据完整，使用直接质检路径。",
            )
            trace.append(event)
            persistence_started = monotonic()
            self.run_store.append_event(run_id, event)
            stage_times["persistence"] += monotonic() - persistence_started

        persistence_started = monotonic()
        self.run_store.save_result(run_id, status, report)
        stage_times["persistence"] += monotonic() - persistence_started

        result = AnalysisResult(
            runId=run_id,
            status=status,
            loopUsed=use_loop,
            loopReason=reason,
            report=report,
            trace=trace,
        )
        print(
            json.dumps(
                {
                    "event": "quality_analysis_timing",
                    "runId": run_id,
                    "callId": request.callId,
                    "loopUsed": use_loop,
                    "llmRequestCount": 1 + loop_iterations * 2,
                    "loopIterations": loop_iterations,
                    "totalMs": round((monotonic() - started) * 1000, 1),
                    "stageMs": {
                        name: round(seconds * 1000, 1)
                        for name, seconds in stage_times.items()
                    },
                },
                ensure_ascii=False,
            )
        )
        return result

    def get_run(self, run_id: str):
        return self.run_store.get_run(run_id)
#服务层没有异常兜底，失败后 run 可能永久停在 RUNNING。
# 直接路径不会经过 QualityGate。