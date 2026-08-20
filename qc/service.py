from __future__ import annotations

from uuid import uuid4

from qc.agent_loop import LoopContext
from qc.direct_analyzer import requires_loop
from qc.errors import AnalysisError, ErrorStage, PipelineFailure
from qc.models import (
    AgentTraceEvent,
    AnalysisRequest,
    AnalysisResult,
    ReviewDisposition,
)
from qc.review_service import compute_route_reasons


class QualityAnalysisService:
    def __init__(
        self,
        direct_analyzer,
        agent_loop,
        quality_gate,
        run_store,
        *,
        min_support_score: float | None = None,
    ):
        self.direct_analyzer = direct_analyzer
        self.agent_loop = agent_loop
        self.quality_gate = quality_gate
        self.run_store = run_store
        if min_support_score is None:
            from qc.config import calibrated_support_score

            min_support_score = calibrated_support_score()
        self.min_support_score = min_support_score

    @staticmethod
    def _deduplicate_errors(errors: list[AnalysisError]) -> list[AnalysisError]:
        seen = set()
        result = []
        for error in errors:
            key = (error.code, error.stage, error.message)
            if key not in seen:
                seen.add(key)
                result.append(error)
        return result

    @staticmethod
    def _gate_errors(gate_result) -> list[AnalysisError]:
        return [
            AnalysisError(
                code=issue.code,
                stage=ErrorStage.QUALITY_GATE,
                message=issue.message,
                retryable=False,
                attempts=1,
            )
            for issue in gate_result.issues
        ]

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        run_id = f"RUN-{uuid4().hex[:12].upper()}"
        try:
            self.run_store.create_run(run_id, request)
        except PipelineFailure as exc:
            return AnalysisResult(
                runId=run_id,
                status="FAILED",
                loopUsed=False,
                report=None,
                errors=[exc.error],
            )

        report = None
        trace = []
        loop_used = False
        loop_reason = None
        errors: list[AnalysisError] = []
        status = "FAILED"

        try:
            report = self.direct_analyzer.analyze(request)
            initial_gate = self.quality_gate.check(
                report,
                request.transcript,
                request.callStartedAt,
                self.min_support_score,
            )
            loop_used, loop_reason = requires_loop(report, initial_gate)

            if loop_used:
                loop_result = self.agent_loop.run(
                    LoopContext(
                        report=report,
                        transcript=request.transcript,
                        callStartedAt=request.callStartedAt,
                        reason=loop_reason or "COMPLEX_CASE",
                    )
                )
                report = loop_result.report
                trace = loop_result.trace
                errors.extend(loop_result.errors)
            else:
                trace = [
                    AgentTraceEvent(
                        iteration=0,
                        phase="FINALIZE",
                        message="直接分析完成，进入最终确定性质量门禁。",
                    )
                ]

            final_gate = self.quality_gate.check(
                report,
                request.transcript,
                request.callStartedAt,
                self.min_support_score,
            )
            errors.extend(self._gate_errors(final_gate))
            if report.auditSnapshot:
                errors.extend(report.auditSnapshot.errors)
            for item in report.summary.get("pendingReviewIssues", []):
                if isinstance(item, dict):
                    errors.append(
                        AnalysisError(
                            code=item.get("code", "QUALITY_GATE_FAILED"),
                            stage=ErrorStage.RULE_EVALUATION,
                            message="存在尚未自动解决的裁决问题",
                            retryable=False,
                            attempts=1,
                        )
                    )
            errors = self._deduplicate_errors(errors)

            if final_gate.passed and not errors:
                status = "COMPLETED"
            else:
                status = "PARTIAL"
                report.disposition = ReviewDisposition.HUMAN_REVIEW_REQUIRED

            for event in trace:
                self.run_store.append_event(run_id, event)
        except PipelineFailure as exc:
            status = "FAILED"
            report = None
            errors = [exc.error]
        except Exception as exc:
            status = "FAILED"
            report = None
            errors = [
                AnalysisError(
                    code="INTERNAL_ERROR",
                    stage=ErrorStage.API,
                    message="质检分析发生内部错误",
                    retryable=False,
                    attempts=1,
                )
            ]

        try:
            reasons = compute_route_reasons(status, report, errors)
            self.run_store.finish_run(
                run_id,
                status,
                report,
                errors,
                route_reasons=reasons,
            )
        except TypeError:
            self.run_store.finish_run(run_id, status, report, errors)
        except PipelineFailure as exc:
            status = "FAILED"
            report = None
            errors = [exc.error]

        review_task = None
        getter = getattr(self.run_store, "get_review_summary", None)
        if getter is not None and status == "PARTIAL":
            review_task = getter(run_id)

        return AnalysisResult(
            runId=run_id,
            status=status,
            loopUsed=loop_used,
            loopReason=loop_reason,
            report=report,
            trace=trace,
            errors=errors,
            reviewTask=review_task,
        )

    def get_run(self, run_id: str):
        return self.run_store.get_run(run_id)
