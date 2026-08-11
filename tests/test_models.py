import pytest
from pydantic import ValidationError

from process_audio import ensure_turn_ids
from qc.models import AnalysisRequest, ClaimFactStatus, QualityReport, TranscriptTurn


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
#这个测试文件的作用是：测试 qc.models 模块中的各种模型类是否能正确工作。