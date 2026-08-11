from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from qc.models import EventType, Violation


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


def calculate_score(
    violations: list[Violation],
    repository: RuleRepository,
) -> int:
    penalties = [
        repository.get(violation.ruleId).penalty
        for violation in violations
    ]
    return max(0, 100 - sum(penalties))
