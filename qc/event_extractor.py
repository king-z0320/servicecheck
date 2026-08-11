from __future__ import annotations

import json

from pydantic import ValidationError

from qc.models import EventType, QualityEvent, TranscriptTurn


EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "eventId": {"type": "string"},
                    # 枚举约束写在 schema 里：json_schema 模式下强约束；
                    # 降级到 json_object 时也作为文本约束，避免模型把 type 填成中文。
                    "type": {
                        "type": "string",
                        "enum": [event_type.value for event_type in EventType],
                    },
                    "statement": {"type": "string"},
                    "turnIds": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                    "ambiguous": {"type": "boolean"},
                },
                "required": [
                    "eventId",
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


class EventExtractor:
    def __init__(self, gateway):
        self.gateway = gateway

    def extract(
        self,
        turns: list[TranscriptTurn],
    ) -> list[QualityEvent]:
        valid_ids = {turn.turnId for turn in turns}
        user = json.dumps(
            [turn.model_dump() for turn in turns],
            ensure_ascii=False,
        )
        data = self.gateway.complete_json(
            system=(
                "你是催收通话质检事件提取器。只提取客户主张和疑似质检事件；"
                "不得判断客户是否真实结清。turnIds只能引用输入中的ID。"
            ),
            user=user,
            schema=EVENT_SCHEMA,
        )
        events = []
        for item in data.get("events", []):
            turn_ids = item.get("turnIds", [])
            if (
                not turn_ids
                or any(turn_id not in valid_ids for turn_id in turn_ids)
            ):
                continue
            try:
                events.append(QualityEvent.model_validate(item))
            except ValidationError:
                continue
        return events
