from __future__ import annotations

from datetime import datetime, timezone

from qc.models import KnowledgeHit, QualityEvent, Violation
from qc.rules import QualityRule


def document_is_active(hit: KnowledgeHit, at_time: datetime) -> bool:
    if at_time.tzinfo is None or at_time.utcoffset() is None:
        raise ValueError("at_time must include timezone")
    try:
        effective_from = datetime.fromisoformat(
            str(hit.metadata["effectiveFrom"]).replace("Z", "+00:00")
        )
        effective_to_raw = hit.metadata.get("effectiveTo")
        effective_to = (
            datetime.fromisoformat(str(effective_to_raw).replace("Z", "+00:00"))
            if effective_to_raw
            else None
        )
    except (KeyError, TypeError, ValueError):
        return False
    at_time = at_time.astimezone(timezone.utc)
    return effective_from <= at_time and (
        effective_to is None or at_time < effective_to
    )


def document_relates_to_rule(hit: KnowledgeHit, rule: QualityRule) -> bool:
    if hit.documentId in {rule.ruleId, rule.sourceDocumentId}:
        return True
    related = hit.metadata.get("relatedRuleIds") or hit.metadata.get("ruleIds") or []
    if isinstance(related, str):
        related = [related]
    return rule.ruleId in related


def supporting_hits(
    *,
    violation: Violation,
    event: QualityEvent,
    rule: QualityRule,
    hits: list[KnowledgeHit],
    at_time: datetime,
    min_score: float,
) -> list[KnowledgeHit]:
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between zero and one")
    requested_ids = set(violation.knowledgeDocumentIds)
    supported = []
    for hit in hits:
        if requested_ids and hit.documentId not in requested_ids:
            continue
        if hit.score < min_score:
            continue
        if hit.metadata.get("eventType") != event.type.value:
            continue
        if not document_is_active(hit, at_time):
            continue
        if not document_relates_to_rule(hit, rule):
            continue
        supported.append(hit)
    return supported
