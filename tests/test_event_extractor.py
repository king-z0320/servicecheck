from datetime import datetime, timezone

import pytest

from qc.errors import PipelineFailure
from qc.event_extractor import EVENT_SCHEMA, EventExtractor
from qc.models import AnalysisRequest, EventType, TranscriptTurn


def make_request(call_id="CALL-001", turns=None):
    return AnalysisRequest(
        caseId="CASE-001",
        callId=call_id,
        callStartedAt=datetime(2025, 10, 15, tzinfo=timezone.utc),
        transcript=turns
        or [
            TranscriptTurn(
                turnId="T0001",
                speaker="客户",
                text="我已经还完了",
                start=1,
                end=2,
            )
        ],
    )


class ValidatingGateway:
    def __init__(self, data):
        self.data = data

    def complete_json(self, *, system, user, schema, validate):
        return validate(self.data)


def candidate(**overrides):
    data = {
        "type": "REPAYMENT_DISPUTE",
        "statement": "客户声称已还款",
        "turnIds": ["T0001"],
        "confidence": 0.98,
        "ambiguous": False,
    }
    data.update(overrides)
    return data


def test_event_schema_removes_model_controlled_event_id():
    properties = EVENT_SCHEMA["properties"]["events"]["items"]["properties"]
    assert "eventId" not in properties


def test_extracts_candidate_and_generates_backend_event_id():
    events = EventExtractor(
        ValidatingGateway({"events": [candidate()]})
    ).extract(make_request())

    assert events[0].type == EventType.REPAYMENT_DISPUTE
    assert events[0].turnIds == ["T0001"]
    assert events[0].eventId.startswith("EVT-")
    assert len(events[0].eventId) == 36


def test_event_id_is_stable_across_statement_rewording():
    request = make_request()
    first = EventExtractor(
        ValidatingGateway({"events": [candidate(statement="版本一")]})
    ).extract(request)
    second = EventExtractor(
        ValidatingGateway({"events": [candidate(statement="版本二")]})
    ).extract(request)

    assert first[0].eventId == second[0].eventId


def test_event_id_changes_with_call_or_evidence_turn():
    first = EventExtractor(
        ValidatingGateway({"events": [candidate()]})
    ).extract(make_request(call_id="CALL-001"))
    other_call = EventExtractor(
        ValidatingGateway({"events": [candidate()]})
    ).extract(make_request(call_id="CALL-002"))
    turns = [
        TranscriptTurn(
            turnId="T0002",
            speaker="客户",
            text="我已经还完了",
            start=1,
            end=2,
        )
    ]
    other_turn = EventExtractor(
        ValidatingGateway(
            {"events": [candidate(turnIds=["T0002"])]}
        )
    ).extract(make_request(turns=turns))

    assert first[0].eventId != other_call[0].eventId
    assert first[0].eventId != other_turn[0].eventId


def test_rejects_hallucinated_turn_ids_instead_of_returning_empty_events():
    extractor = EventExtractor(
        ValidatingGateway(
            {"events": [candidate(turnIds=["T9999"])]}
        )
    )

    with pytest.raises(PipelineFailure) as captured:
        extractor.extract(make_request())

    assert captured.value.error.code == "EVENT_UNKNOWN_TURN"


def test_rejects_model_supplied_event_id_as_an_extra_field():
    extractor = EventExtractor(
        ValidatingGateway(
            {"events": [candidate(eventId="MODEL-CONTROLLED")]}
        )
    )

    with pytest.raises(PipelineFailure) as captured:
        extractor.extract(make_request())

    assert captured.value.error.code == "EVENT_SCHEMA_INVALID"


def test_duplicate_candidates_are_merged_and_keep_higher_confidence():
    events = EventExtractor(
        ValidatingGateway(
            {
                "events": [
                    candidate(confidence=0.4, statement="较低置信度"),
                    candidate(confidence=0.9, statement="较高置信度"),
                ]
            }
        )
    ).extract(make_request())

    assert len(events) == 1
    assert events[0].confidence == 0.9
    assert events[0].statement == "较高置信度"
