from datetime import datetime, timezone

import pytest

from qc.models import AnalysisRequest, TranscriptTurn
from qc.rag import KnowledgeIndex
from tests.live_support import (
    ROOT,
    build_live_service,
    live_settings,
    running_mock_audit_server,
)


CASES = [
    {
        "id": "live-repayment-violation",
        "callId": "CALL-NONCOMPLIANT-002",
        "turns": [
            ("客户", "我这笔钱昨天已经还清了。"),
            ("坐席", "不可能，你没还，今天必须处理。"),
        ],
        "eventType": "REPAYMENT_DISPUTE",
        "ruleId": "R006",
    },
    {
        "id": "live-repayment-compliant",
        "callId": "CALL-COMPLIANT-001",
        "turns": [
            ("客户", "我昨天已经还款了。"),
            ("坐席", "请提供时间、金额和渠道，我为您登记核验。"),
        ],
        "eventType": "REPAYMENT_DISPUTE",
        "ruleId": None,
    },
    {
        "id": "live-threat",
        "callId": "CALL-COMPLIANT-001",
        "turns": [
            ("客户", "我目前确实有困难。"),
            ("坐席", "你再不还款，法院马上把你抓起来坐牢。"),
        ],
        "eventType": "THREAT_OR_COERCION",
        "ruleId": "R002",
    },
    {
        "id": "live-third-party",
        "callId": "CALL-COMPLIANT-001",
        "turns": [
            ("客户", "我是他姐姐，他现在不在。"),
            ("坐席", "他在我们平台欠款五千元，已经逾期很多天。"),
        ],
        "eventType": "THIRD_PARTY_CONTACT",
        "ruleId": "R005",
    },
]


@pytest.mark.live_llm
@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_real_deepseek_text_to_persisted_report(case, tmp_path):
    live_settings()  # fail clearly before building expensive local components
    knowledge = KnowledgeIndex(ROOT / "knowledge")
    knowledge.build()
    with running_mock_audit_server() as audit_url:
        system = build_live_service(
            tmp_path / "live-runs.db",
            audit_url,
            knowledge_index=knowledge,
        )
        turns = [
            TranscriptTurn(
                turnId=f"T{index:04d}",
                speaker=speaker,
                text=text,
                start=float(index - 1),
                end=float(index),
            )
            for index, (speaker, text) in enumerate(case["turns"], 1)
        ]
        result = system.analyze(
            AnalysisRequest(
                caseId=case["id"],
                callId=case["callId"],
                callStartedAt=datetime(2026, 7, 27, tzinfo=timezone.utc),
                transcript=turns,
            )
        )

    assert result.status == "COMPLETED", [error.code for error in result.errors]
    assert result.report is not None
    assert case["eventType"] in {event.type.value for event in result.report.events}
    assert all(event.eventId.startswith("EVT-") for event in result.report.events)
    rule_ids = {item.ruleId for item in result.report.violations}
    if case["ruleId"] is None:
        assert rule_ids == set()
    else:
        assert case["ruleId"] in rule_ids
    assert system.get_run(result.runId)["status"] == "COMPLETED"
