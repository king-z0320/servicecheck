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
    assert result.status == "COMPLETED"
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
    assert len([item for item in result.trace if item.phase == "PLAN"]) == 3
    assert "LOOP_BUDGET_EXHAUSTED" in {item.code for item in result.errors}
    assert result.status == "PARTIAL"
    assert result.report.disposition.value == "HUMAN_REVIEW_REQUIRED"


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


def test_service_does_not_emit_observability_logs_in_this_phase(
    system_factory,
    capsys,
):
    system = system_factory(ambiguous=False)
    system.analyze(
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
    assert capsys.readouterr().out == ""
