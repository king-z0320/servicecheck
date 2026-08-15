import pytest
from pydantic import ValidationError

from process_audio import ensure_turn_ids
from qc.models import (
    AnalysisRequest,
    AnalysisResult,
    ClaimFactStatus,
    QualityReport,
    TranscriptTurn,
)


def test_transcript_turn_requires_stable_turn_id():
    with pytest.raises(ValidationError):
        TranscriptTurn(speaker="客户", text="我已经还完了", start=1.0, end=2.0)


def test_analysis_request_accepts_call_context():
    request = AnalysisRequest(
        caseId="CASE-001",
        callId="CALL-001",
        transcript=[
            TranscriptTurn(
                turnId="T0001",
                speaker="客户",
                text="我已经还完了",
                start=1.0,
                end=2.0,
            )
        ],
    )
    assert request.callId == "CALL-001"


def test_report_business_fact_defaults_to_not_checked():
    report = QualityReport(callId="CALL-001")
    assert report.businessFact.status == ClaimFactStatus.NOT_CHECKED
    assert report.score == 100


def test_ensure_turn_ids_is_deterministic():
    transcript = [{"speaker": "客户", "text": "测试", "start": 0, "end": 1}]
    assert ensure_turn_ids(transcript)[0]["turnId"] == "T0001"


def test_analysis_result_accepts_failed_without_report():
    result = AnalysisResult(
        runId="RUN-FAILED",
        status="FAILED",
        loopUsed=False,
        report=None,
    )

    assert result.status == "FAILED"
    assert result.report is None


def test_analysis_result_rejects_blocked_status():
    with pytest.raises(ValidationError):
        AnalysisResult(
            runId="RUN-BLOCKED",
            status="BLOCKED",
            loopUsed=False,
            report=None,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("turnId", "   "),
        ("speaker", "   "),
        ("text", "   "),
    ],
)
def test_transcript_turn_rejects_blank_reference_fields(field, value):
    data = {
        "turnId": "T0001",
        "speaker": "客户",
        "text": "测试",
        "start": 0,
        "end": 1,
    }
    data[field] = value

    with pytest.raises(ValidationError):
        TranscriptTurn(**data)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_transcript_turn_rejects_invalid_start(value):
    with pytest.raises(ValidationError):
        TranscriptTurn(
            turnId="T0001",
            speaker="客户",
            text="测试",
            start=value,
            end=1,
        )


def test_analysis_request_rejects_duplicate_turn_ids():
    with pytest.raises(ValidationError):
        AnalysisRequest(
            caseId="CASE-001",
            callId="CALL-001",
            transcript=[
                TranscriptTurn(
                    turnId="T0001",
                    speaker="客户",
                    text="第一句",
                    start=0,
                    end=1,
                ),
                TranscriptTurn(
                    turnId="T0001",
                    speaker="坐席",
                    text="第二句",
                    start=1,
                    end=2,
                ),
            ],
        )


def test_analysis_request_rejects_out_of_order_starts():
    with pytest.raises(ValidationError):
        AnalysisRequest(
            caseId="CASE-001",
            callId="CALL-001",
            transcript=[
                TranscriptTurn(
                    turnId="T0001",
                    speaker="客户",
                    text="第一句",
                    start=2,
                    end=3,
                ),
                TranscriptTurn(
                    turnId="T0002",
                    speaker="坐席",
                    text="第二句",
                    start=1,
                    end=2.5,
                ),
            ],
        )


def test_analysis_request_allows_overlapping_turns():
    request = AnalysisRequest(
        caseId=" CASE-001 ",
        callId=" CALL-001 ",
        transcript=[
            TranscriptTurn(
                turnId=" T0001 ",
                speaker=" 客户 ",
                text=" 第一句 ",
                start=0,
                end=2,
            ),
            TranscriptTurn(
                turnId="T0002",
                speaker="坐席",
                text="第二句",
                start=1,
                end=3,
            ),
        ],
    )

    assert request.caseId == "CASE-001"
    assert request.callId == "CALL-001"
    assert request.transcript[0].turnId == "T0001"
#这个测试文件的作用是：测试 qc.models 模块中的各种模型类是否能正确工作。
