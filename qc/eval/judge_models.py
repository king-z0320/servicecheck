"""Data contracts for semantic LLM-as-a-Judge evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JudgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dimension: Literal["faithfulness", "answer_relevancy"]
    questionSummary: str = ""
    reportSummary: dict
    referenceAnswerPoints: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["completed", "unavailable", "invalid", "not_run"]
    dimension: str
    score: int | None = Field(default=None, ge=0, le=4)
    reason: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    provider: str = "unknown"
    model: str = "unknown"
    promptVersion: str = "unknown"
    rubricVersion: str = "v1"
    invocationId: str | None = None
    tokenSource: str = "unknown"

    @field_validator("evidenceIds")
    @classmethod
    def evidence_ids_are_strings(cls, values):
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("evidenceIds must contain non-empty strings")
        return values
