from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from qc.models import EventType, QualityEvent, Violation


class QualityRule(BaseModel):
    ruleId: str
    name: str
    eventTypes: list[EventType]
    description: str
    penalty: int
    version: str
    effectiveFrom: datetime
    effectiveTo: datetime | None = None
    sourceDocumentId: str


class RuleRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._rules = [
            QualityRule.model_validate(item)
            for item in json.loads(self.path.read_text(encoding="utf-8"))
        ]

    def get(self, rule_id: str) -> QualityRule:
        for rule in self._rules:
            if rule.ruleId == rule_id:
                return rule
        raise KeyError(rule_id)

    def source_document_ids(self) -> set[str]:
        return {rule.sourceDocumentId for rule in self._rules}

    def load_active(
        self,
        event_type: EventType,
        at_time: datetime,
    ) -> list[QualityRule]:
        if at_time.tzinfo is None:
            raise ValueError("at_time must include timezone")
        at_time = at_time.astimezone(timezone.utc)
        return [
            rule
            for rule in self._rules
            if event_type in rule.eventTypes
            and rule.effectiveFrom <= at_time
            and (rule.effectiveTo is None or at_time < rule.effectiveTo)
        ]

    def get_active(
        self,
        rule_id: str,
        event_type: EventType,
        at_time: datetime,
    ) -> QualityRule | None:
        rule = self.get(rule_id)
        if at_time.tzinfo is None or at_time.utcoffset() is None:
            raise ValueError("at_time must include timezone")
        at_time = at_time.astimezone(timezone.utc)
        if event_type not in rule.eventTypes:
            return None
        if rule.effectiveFrom > at_time:
            return None
        if rule.effectiveTo is not None and at_time >= rule.effectiveTo:
            return None
        return rule


def deduplicate_violations(
    violations: list[Violation],
    event_order: dict[str, int] | None = None,
) -> list[Violation]:
    merged: dict[tuple, Violation] = {}
    first_position: dict[tuple, int] = {}
    for position, violation in enumerate(violations):
        if violation.eventId:
            key = ("event", violation.eventId, violation.ruleId)
        else:
            key = (
                "legacy",
                violation.ruleId,
                tuple(sorted(set(violation.evidenceTurnIds))),
            )
        current = merged.get(key)
        if current is None:
            merged[key] = violation.model_copy(deep=True)
            first_position[key] = position
            continue
        known = set(current.knowledgeDocumentIds)
        current.knowledgeDocumentIds.extend(
            document_id
            for document_id in violation.knowledgeDocumentIds
            if document_id not in known
        )

    order = event_order or {}
    return [
        merged[key]
        for key in sorted(
            merged,
            key=lambda item: (
                order.get(merged[item].eventId or "", first_position[item]),
                merged[item].ruleId,
                first_position[item],
            ),
        )
    ]


def calculate_score(
    violations: list[Violation],
    repository: RuleRepository,
    call_started_at: datetime,
    events: list[QualityEvent],
) -> int:
    event_by_id = {event.eventId: event for event in events}
    event_order = {
        event.eventId: index
        for index, event in enumerate(events)
    }
    penalties = []
    for violation in deduplicate_violations(violations, event_order):
        repository.get(violation.ruleId)  # unknown IDs fail closed
        if not violation.eventId or violation.eventId not in event_by_id:
            raise KeyError(violation.eventId or "missing eventId")
        event = event_by_id[violation.eventId]
        rule = repository.get_active(
            violation.ruleId,
            event.type,
            call_started_at,
        )
        if rule is not None:
            penalties.append(rule.penalty)
    return max(0, 100 - sum(penalties))
