from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HumanOutcome(str, Enum):
    CONFIRMED_PASS = "CONFIRMED_PASS"
    CONFIRMED_VIOLATION = "CONFIRMED_VIOLATION"
    UNRESOLVED = "UNRESOLVED"


class ReviewTaskStatus(str, Enum):
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


class ReasonCode(str, Enum):
    PASS_CONFIRMED = "PASS_CONFIRMED"
    VIOLATION_CONFIRMED = "VIOLATION_CONFIRMED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ASR_UNCLEAR = "ASR_UNCLEAR"
    OTHER = "OTHER"


class DecisionSource(str, Enum):
    HUMAN = "HUMAN"


class ContextSource(str, Enum):
    CONFIGURED_DEMO = "CONFIGURED_DEMO"


REASON_CODES_BY_OUTCOME = {
    HumanOutcome.CONFIRMED_PASS: {ReasonCode.PASS_CONFIRMED},
    HumanOutcome.CONFIRMED_VIOLATION: {ReasonCode.VIOLATION_CONFIRMED},
    HumanOutcome.UNRESOLVED: {
        ReasonCode.INSUFFICIENT_EVIDENCE,
        ReasonCode.ASR_UNCLEAR,
        ReasonCode.OTHER,
    },
}

PROTECTED_SUBMIT_FIELDS = {
    "score",
    "disposition",
    "violations",
    "events",
    "knowledgeHits",
    "auditSnapshot",
    "report",
    "reportJson",
    "rules",
    "transcript",
    "reviewerId",
    "reviewer_id",
    "decisionSource",
    "effectiveRevisionId",
}


class RouteReason(BaseModel):
    code: str
    stage: str
    eventId: str | None = None
    ruleId: str | None = None
    reportPath: str | None = None


class ReviewerContext(BaseModel):
    reviewerId: str = "configured-demo-reviewer"
    contextSource: ContextSource = ContextSource.CONFIGURED_DEMO
    decisionSource: DecisionSource = DecisionSource.HUMAN


class ReviewSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expectedVersion: int = Field(ge=1)
    outcome: HumanOutcome
    reasonCode: ReasonCode
    note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_reason_matches_outcome(self):
        allowed = REASON_CODES_BY_OUTCOME[self.outcome]
        if self.reasonCode not in allowed:
            raise ValueError("reasonCode is not valid for outcome")
        return self


class ReviewRevisionView(BaseModel):
    revisionId: str
    taskId: str
    runId: str
    outcome: HumanOutcome
    reasonCode: ReasonCode
    note: str
    reviewerId: str
    contextSource: ContextSource
    decisionSource: DecisionSource
    createdAt: datetime


class ReviewTaskSummary(BaseModel):
    reviewTaskId: str
    runId: str
    batchItemId: int | None = None
    callId: str | None = None
    status: ReviewTaskStatus
    version: int
    routeReasons: list[RouteReason] = Field(default_factory=list)
    score: int | None = None
    createdAt: datetime | None = None
    updatedAt: datetime | None = None
    effectiveRevisionId: str | None = None


class ReviewTaskListResponse(BaseModel):
    items: list[ReviewTaskSummary]
    page: int
    pageSize: int
    total: int


def canonical_submit_hash(outcome: str, reason_code: str, note: str) -> str:
    payload = json.dumps(
        {"note": note, "outcome": outcome, "reasonCode": reason_code},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configured_reviewer_context() -> ReviewerContext:
    return ReviewerContext()


def revision_view(row: dict[str, Any]) -> ReviewRevisionView:
    return ReviewRevisionView.model_validate(row)
