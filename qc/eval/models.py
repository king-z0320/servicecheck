"""Pydantic data models used by the evaluation package."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qc.models import AnalysisRequest, QualityReport, TranscriptTurn


class EvalSplit(str, Enum):
    DEV = "dev"
    REGRESSION = "regression"
    CHALLENGE = "challenge"


class EvalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    caseId: str
    callId: str
    callStartedAt: datetime
    transcript: list[TranscriptTurn] = Field(min_length=1)

    @field_validator("callStartedAt")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("callStartedAt must include timezone")
        return value.astimezone(timezone.utc)

    def to_request(self) -> AnalysisRequest:
        return AnalysisRequest.model_validate(self.model_dump(mode="json"))


class ExpectedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    eventType: str = Field(min_length=1)
    requiredTurnIds: list[str] = Field(default_factory=list)


class ExpectedEval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[ExpectedEvent]
    ruleIds: list[str]
    allowedDispositions: list[str] = Field(min_length=1)
    requiredContextIds: list[str] = Field(default_factory=list)
    relevantContextIds: list[str] = Field(default_factory=list)
    forbiddenContextIds: list[str] = Field(default_factory=list)
    referenceAnswerPoints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_context_labels(self):
        if set(self.requiredContextIds) & set(self.forbiddenContextIds):
            raise ValueError("requiredContextIds and forbiddenContextIds overlap")
        return self


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    caseId: str = Field(min_length=1)
    split: EvalSplit
    source: dict[str, Any]
    labelNotes: str = Field(min_length=1)
    input: EvalInput
    expected: ExpectedEval
    caseHash: str | None = None

    @model_validator(mode="after")
    def validate_references(self):
        turn_ids = {item.turnId for item in self.input.transcript}
        for event in self.expected.events:
            missing = set(event.requiredTurnIds) - turn_ids
            if missing:
                raise ValueError(f"requiredTurnIds refer to missing turns: {sorted(missing)}")
        if self.input.caseId != self.caseId:
            raise ValueError("caseId must match input.caseId")
        return self


class EvalManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evalRunId: str
    createdAt: datetime
    codeCommitOrSourceHash: str
    datasetHash: str
    split: EvalSplit
    executionMode: Literal["replay", "model", "e2e"]
    judgeMode: Literal["none", "fake", "live"]
    targetModel: dict[str, Any] = Field(default_factory=dict)
    judgeModel: dict[str, Any] = Field(default_factory=dict)
    promptVersion: str | None = None
    ruleVersion: str | None = None
    knowledgeVersion: str | None = None
    retrievalConfig: dict[str, Any] = Field(default_factory=dict)
    changeSummary: str = ""
    changes: list[dict[str, Any]] = Field(default_factory=list)
    expectedImpact: str = ""


class EvalExecutionResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    report: QualityReport
    runId: str | None = None
    traceId: str | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    usage: list[dict[str, Any]] = Field(default_factory=list)


class CaseMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deterministic: dict[str, Any]
    rag: dict[str, Any]
    judgeResult: dict[str, Any] = Field(default_factory=dict)
    status: str = "passed"
    failureReasons: list[dict[str, Any]] = Field(default_factory=list)


class EvalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    caseId: str
    caseHash: str | None = None
    status: str
    metrics: CaseMetrics
    judgeResult: dict[str, Any] = Field(default_factory=dict)
    traceId: str | None = None
    runId: str | None = None


class EvalRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evalRunId: str
    status: str
    manifest: EvalManifest
    caseResults: list[EvalCaseResult]
    aggregateMetrics: dict[str, Any] = Field(default_factory=dict)
    failures: list[dict[str, Any]] = Field(default_factory=list)
