from datetime import datetime, timezone

import pytest

from qc.agent_loop import LoopResult
from qc.errors import AnalysisError, ErrorStage, PipelineFailure
from qc.models import (
    AnalysisRequest,
    AuditSnapshot,
    QualityReport,
    ReviewDisposition,
    TranscriptTurn,
)
from qc.quality_gate import GateIssue, GateResult
from qc.run_store import RunStore
from qc.service import QualityAnalysisService


def request():
    return AnalysisRequest(
        caseId="CASE",
        callId="CALL",
        callStartedAt=datetime(2026, 7, 27, tzinfo=timezone.utc),
        transcript=[
            TranscriptTurn(
                turnId="T0001",
                speaker="客户",
                text="正常沟通",
                start=0,
                end=1,
            )
        ],
    )


class StaticDirect:
    def __init__(self, report=None, failure=None):
        self.report = report or QualityReport(callId="CALL")
        self.failure = failure

    def analyze(self, request):
        if self.failure:
            raise self.failure
        return self.report


class CountingGate:
    def __init__(self, passed=True):
        self.passed = passed
        self.calls = 0

    def check(self, report, transcript, call_started_at, min_support_score=None):
        self.calls += 1
        if self.passed:
            return GateResult(passed=True)
        return GateResult(
            passed=False,
            issues=[
                GateIssue(
                    code="QUALITY_GATE_FAILED",
                    message="最终门禁失败",
                    reportPath="report",
                )
            ],
        )


class NeverLoop:
    def run(self, context):
        raise AssertionError("loop should not run")


class PartialLoop:
    def run(self, context):
        return LoopResult(
            status="HUMAN_REVIEW_REQUIRED",
            report=context.report,
            iterations=1,
            toolCalls=1,
            trace=[],
            errors=[
                AnalysisError(
                    code="LOOP_BUDGET_EXHAUSTED",
                    stage=ErrorStage.AGENT_LOOP,
                    message="循环预算耗尽",
                    retryable=False,
                    attempts=1,
                )
            ],
        )


def service(tmp_path, direct, gate, loop=None):
    return QualityAnalysisService(
        direct,
        loop or NeverLoop(),
        gate,
        RunStore(tmp_path / "runs.db"),
        min_support_score=0.7,
    )


def test_completed_path_runs_initial_and_final_gate_and_persists(tmp_path):
    gate = CountingGate(passed=True)
    system = service(tmp_path, StaticDirect(), gate)
    result = system.analyze(request())

    assert result.status == "COMPLETED"
    assert result.report.disposition == ReviewDisposition.AUTO_PASS
    assert gate.calls == 2
    assert system.get_run(result.runId)["status"] == "COMPLETED"


def test_final_gate_failure_becomes_partial_human_review(tmp_path):
    gate = CountingGate(passed=False)
    system = service(tmp_path, StaticDirect(), gate, PartialLoop())
    result = system.analyze(request())

    assert result.status == "PARTIAL"
    assert result.report.disposition == ReviewDisposition.HUMAN_REVIEW_REQUIRED
    assert {item.code for item in result.errors} >= {"QUALITY_GATE_FAILED"}
    assert system.get_run(result.runId)["status"] == "PARTIAL"


def test_loop_budget_exhaustion_becomes_partial(tmp_path):
    report = QualityReport(
        callId="CALL",
        summary={"pendingReviewIssues": [{"code": "RAG_WEAK_SUPPORT"}]},
        disposition=ReviewDisposition.HUMAN_REVIEW_REQUIRED,
    )
    system = service(
        tmp_path,
        StaticDirect(report),
        CountingGate(passed=True),
        PartialLoop(),
    )
    result = system.analyze(request())

    assert result.status == "PARTIAL"
    assert "LOOP_BUDGET_EXHAUSTED" in {item.code for item in result.errors}


def test_audit_error_cannot_be_completed_or_auto_passed(tmp_path):
    audit_error = AnalysisError(
        code="AUDIT_TIMEOUT",
        stage=ErrorStage.AUDIT,
        message="审计超时",
        retryable=True,
        attempts=2,
    )
    report = QualityReport(
        callId="CALL",
        auditSnapshot=AuditSnapshot(callId="CALL", errors=[audit_error]),
        disposition=ReviewDisposition.HUMAN_REVIEW_REQUIRED,
    )
    system = service(
        tmp_path,
        StaticDirect(report),
        CountingGate(passed=True),
        PartialLoop(),
    )
    result = system.analyze(request())

    assert result.status == "PARTIAL"
    assert "AUDIT_TIMEOUT" in {item.code for item in result.errors}


@pytest.mark.parametrize("code", ["LLM_TIMEOUT", "LLM_AUTH_FAILED"])
def test_expected_dependency_failure_is_failed_and_persisted(tmp_path, code):
    failure = PipelineFailure(
        AnalysisError(
            code=code,
            stage=ErrorStage.EVENT_EXTRACTION,
            message="大模型失败",
            retryable=code == "LLM_TIMEOUT",
            attempts=2 if code == "LLM_TIMEOUT" else 1,
        )
    )
    system = service(tmp_path, StaticDirect(failure=failure), CountingGate())
    result = system.analyze(request())

    assert result.status == "FAILED"
    assert result.report is None
    stored = system.get_run(result.runId)
    assert stored["status"] == "FAILED"
    assert stored["errors"][0]["code"] == code


def test_unknown_exception_is_failed_and_does_not_leave_running(tmp_path):
    system = service(
        tmp_path,
        StaticDirect(failure=RuntimeError("private traceback detail")),
        CountingGate(),
    )
    result = system.analyze(request())

    assert result.status == "FAILED"
    assert result.errors[0].code == "INTERNAL_ERROR"
    assert "private traceback detail" not in result.errors[0].message
    assert system.get_run(result.runId)["status"] == "FAILED"


def test_create_run_failure_returns_run_id_even_when_not_queryable(tmp_path):
    class BrokenStore:
        def create_run(self, run_id, request):
            raise PipelineFailure(
                AnalysisError(
                    code="SQLITE_LOCKED",
                    stage=ErrorStage.PERSISTENCE,
                    message="结果存储暂时不可用",
                    retryable=True,
                    attempts=3,
                )
            )

        def get_run(self, run_id):
            raise KeyError(run_id)

    system = QualityAnalysisService(
        StaticDirect(),
        NeverLoop(),
        CountingGate(),
        BrokenStore(),
        min_support_score=0.7,
    )
    result = system.analyze(request())

    assert result.runId.startswith("RUN-")
    assert result.status == "FAILED"
    with pytest.raises(KeyError):
        system.get_run(result.runId)
