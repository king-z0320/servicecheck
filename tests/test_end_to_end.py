import json

from qc.models import AnalysisRequest, TranscriptTurn


def test_noncompliant_repayment_dispute_uses_direct_path(system_factory):
    system = system_factory(ambiguous=False)
    result = system.analyze(
        AnalysisRequest(
            caseId="CASE-002",
            callId="CALL-NONCOMPLIANT-002",
            transcript=[
                TranscriptTurn(
                    turnId="T0001",
                    speaker="客户",
                    text="我已经还完了",
                    start=0,
                    end=1,
                ),
                TranscriptTurn(
                    turnId="T0002",
                    speaker="坐席",
                    text="不可能，你今天必须处理",
                    start=1,
                    end=2,
                ),
            ],
        )
    )
    assert result.loopUsed is False
    assert result.report.violations[0].ruleId == "R006"
    assert result.report.businessFact.status.value == "NOT_CHECKED"
    assert result.report.knowledgeHits[0].documentId == "POLICY-REPAYMENT-003"
    assert result.report.auditSnapshot.disputeTicketCreated is False


def test_ambiguous_language_uses_bounded_loop_and_evaluator_replans(
    system_factory,
):
    system = system_factory(ambiguous=True)
    result = system.analyze(
        AnalysisRequest(
            caseId="CASE-003",
            callId="CALL-NONCOMPLIANT-002",
            transcript=[
                TranscriptTurn(
                    turnId="T0001",
                    speaker="客户",
                    text="我好像处理过了",
                    start=0,
                    end=1,
                )
            ],
        )
    )
    assert result.loopUsed is True
    assert result.loopReason == "AMBIGUOUS_EVENT"
    assert len([item for item in result.trace if item.phase == "PLAN"]) == 2
    assert len([item for item in result.trace if item.phase == "REPLAN"]) == 1
    assert result.status == "COMPLETED"


def test_complex_timing_counts_event_extraction_planner_and_evaluator(
    system_factory,
    capsys,
):
    system = system_factory(ambiguous=True)
    system.analyze(
        AnalysisRequest(
            caseId="CASE-COMPLEX-TIMING",
            callId="CALL-NONCOMPLIANT-002",
            transcript=[
                TranscriptTurn(
                    turnId="T0001",
                    speaker="客户",
                    text="我好像处理过了",
                    start=0,
                    end=1,
                )
            ],
        )
    )
    timing = next(
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
        and json.loads(line).get("event") == "quality_analysis_timing"
    )
    assert timing["loopIterations"] == 2
    assert timing["llmRequestCount"] == 5


def test_completed_run_is_readable_from_a_new_store_instance(system_factory):
    system = system_factory(ambiguous=False)
    result = system.analyze(
        AnalysisRequest(
            caseId="CASE-004",
            callId="CALL-NONCOMPLIANT-002",
            transcript=[
                TranscriptTurn(
                    turnId="T0001",
                    speaker="客户",
                    text="我已经还完了",
                    start=0,
                    end=1,
                )
            ],
        )
    )
    reloaded = system_factory.reload_run(result.runId)
    assert reloaded["status"] == "COMPLETED"
    assert reloaded["result"]["businessFact"]["status"] == "NOT_CHECKED"


def test_service_emits_structured_single_call_timing(system_factory, capsys):
    system = system_factory(ambiguous=False)
    result = system.analyze(
        AnalysisRequest(
            caseId="CASE-TIMING",
            callId="CALL-NONCOMPLIANT-002",
            transcript=[
                TranscriptTurn(
                    turnId="T0001",
                    speaker="客户",
                    text="我已经还完了",
                    start=0,
                    end=1,
                )
            ],
        )
    )
    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    timing = next(
        item for item in records if item.get("event") == "quality_analysis_timing"
    )
    assert timing["runId"] == result.runId
    assert timing["callId"] == "CALL-NONCOMPLIANT-002"
    assert timing["loopUsed"] is False
    assert timing["llmRequestCount"] == 1
    assert timing["loopIterations"] == 0
    assert timing["totalMs"] >= 0
    assert set(timing["stageMs"]) == {
        "eventExtraction",
        "knowledgeRetrieval",
        "actionAudit",
        "directAnalysis",
        "agentLoop",
        "persistence",
    }
