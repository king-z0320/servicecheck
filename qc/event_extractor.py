from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from qc.errors import (
    AnalysisError,
    ErrorStage,
    OutputValidationError,
    PipelineFailure,
)
from qc.models import AnalysisRequest, EventType, QualityEvent, TranscriptTurn


EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [event_type.value for event_type in EventType],
                    },
                    "statement": {"type": "string", "minLength": 1},
                    "turnIds": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "ambiguous": {"type": "boolean"},
                },
                "required": [
                    "type",
                    "statement",
                    "turnIds",
                    "confidence",
                    "ambiguous",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}


class EventCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EventType
    statement: str
    turnIds: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguous: bool

    @field_validator("statement")
    @classmethod
    def require_statement(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("statement must not be blank")
        return value

    @field_validator("turnIds")
    @classmethod
    def require_unique_turn_ids(cls, values: list[str]) -> list[str]:
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            raise ValueError("turnIds must not contain blank values")
        if len(stripped) != len(set(stripped)):
            raise ValueError("turnIds must be unique")
        return stripped


class EventCandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[EventCandidate]


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def make_event_id(
    call_id: str,
    event_type: EventType,
    turn_ids: list[str],
    transcript: list[TranscriptTurn],
) -> tuple[str, list[str]]:
    turn_order = {turn.turnId: index for index, turn in enumerate(transcript)}
    turn_by_id = {turn.turnId: turn for turn in transcript}
    ordered_ids = sorted(set(turn_ids), key=turn_order.__getitem__)
    canonical = {
        "callId": _normalize(call_id),
        "eventType": event_type.value,
        "turnIds": ordered_ids,
        "sourceText": _normalize(
            "\n".join(turn_by_id[turn_id].text for turn_id in ordered_ids)
        ),
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:32].upper()
    return f"EVT-{digest}", ordered_ids


class EventExtractor:
    def __init__(self, gateway):
        self.gateway = gateway

    def extract(self, request: AnalysisRequest) -> list[QualityEvent]:
        valid_ids = {turn.turnId for turn in request.transcript}

        def validate(data: dict) -> EventCandidateBatch:
            try:
                batch = EventCandidateBatch.model_validate(data)
            except ValidationError as exc:
                raise OutputValidationError(
                    "EVENT_SCHEMA_INVALID",
                    "事件候选结构无效",
                ) from exc
            unknown = sorted(
                {
                    turn_id
                    for event in batch.events
                    for turn_id in event.turnIds
                    if turn_id not in valid_ids
                }
            )
            if unknown:
                raise OutputValidationError(
                    "EVENT_UNKNOWN_TURN",
                    "事件候选引用了不存在的转写证据",
                )
            return batch

        user = json.dumps(
            [turn.model_dump() for turn in request.transcript],
            ensure_ascii=False,
        )
        try:
            batch = self.gateway.complete_json(
                system=(
                    "你是催收通话质检事件提取器。只提取客户主张和疑似质检事件；"
                    "不得判断客户是否真实结清。turnIds只能引用输入中的ID。"
                ),
                user=user,
                schema=EVENT_SCHEMA,
                validate=validate,
            )
            if not isinstance(batch, EventCandidateBatch):
                batch = validate(batch)
        except OutputValidationError as exc:
            raise PipelineFailure(
                AnalysisError(
                    code=exc.code,
                    stage=ErrorStage.EVENT_EXTRACTION,
                    message=exc.safe_message,
                    retryable=False,
                    attempts=1,
                )
            ) from exc

        deduplicated: dict[str, QualityEvent] = {}
        for candidate in batch.events:
            event_id, ordered_ids = make_event_id(
                request.callId,
                candidate.type,
                candidate.turnIds,
                request.transcript,
            )
            event = QualityEvent(
                eventId=event_id,
                type=candidate.type,
                statement=candidate.statement,
                turnIds=ordered_ids,
                confidence=candidate.confidence,
                ambiguous=candidate.ambiguous,
            )
            current = deduplicated.get(event_id)
            if current is None or event.confidence > current.confidence:
                deduplicated[event_id] = event
        return list(deduplicated.values())
