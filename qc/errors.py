from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ErrorStage(str, Enum):
    VALIDATION = "VALIDATION"
    ASR = "ASR"
    EVENT_EXTRACTION = "EVENT_EXTRACTION"
    RAG = "RAG"
    AUDIT = "AUDIT"
    RULE_EVALUATION = "RULE_EVALUATION"
    QUALITY_GATE = "QUALITY_GATE"
    AGENT_LOOP = "AGENT_LOOP"
    PERSISTENCE = "PERSISTENCE"
    API = "API"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    EMPTY_TRANSCRIPT = "EMPTY_TRANSCRIPT"
    DUPLICATE_TURN_ID = "DUPLICATE_TURN_ID"
    EMPTY_TURN_TEXT = "EMPTY_TURN_TEXT"
    INVALID_TURN_TIME = "INVALID_TURN_TIME"
    INVALID_CALL_STARTED_AT = "INVALID_CALL_STARTED_AT"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
    LLM_AUTH_FAILED = "LLM_AUTH_FAILED"
    LLM_UPSTREAM_ERROR = "LLM_UPSTREAM_ERROR"
    LLM_INVALID_OUTPUT = "LLM_INVALID_OUTPUT"
    EVENT_UNKNOWN_TURN = "EVENT_UNKNOWN_TURN"
    EVENT_SCHEMA_INVALID = "EVENT_SCHEMA_INVALID"
    RAG_INDEX_NOT_BUILT = "RAG_INDEX_NOT_BUILT"
    RAG_WEAK_SUPPORT = "RAG_WEAK_SUPPORT"
    RAG_DOCUMENT_INACTIVE = "RAG_DOCUMENT_INACTIVE"
    NO_ACTIVE_RULE_SUPPORT = "NO_ACTIVE_RULE_SUPPORT"
    UNKNOWN_RULE = "UNKNOWN_RULE"
    DUPLICATE_VIOLATION = "DUPLICATE_VIOLATION"
    AUDIT_TIMEOUT = "AUDIT_TIMEOUT"
    AUDIT_NOT_FOUND = "AUDIT_NOT_FOUND"
    AUDIT_AUTH_FAILED = "AUDIT_AUTH_FAILED"
    AUDIT_UPSTREAM_ERROR = "AUDIT_UPSTREAM_ERROR"
    AUDIT_INVALID_RESPONSE = "AUDIT_INVALID_RESPONSE"
    QUALITY_GATE_FAILED = "QUALITY_GATE_FAILED"
    LOOP_BUDGET_EXHAUSTED = "LOOP_BUDGET_EXHAUSTED"
    LOOP_UNSUPPORTED_ACTION = "LOOP_UNSUPPORTED_ACTION"
    SQLITE_LOCKED = "SQLITE_LOCKED"
    PERSISTENCE_WRITE_FAILED = "PERSISTENCE_WRITE_FAILED"
    PROCESS_INTERRUPTED = "PROCESS_INTERRUPTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    ASR_FAILED = "ASR_FAILED"
    ASR_EMPTY_TRANSCRIPT = "ASR_EMPTY_TRANSCRIPT"


class AnalysisError(BaseModel):
    code: str
    stage: ErrorStage
    message: str
    retryable: bool
    attempts: int = Field(default=0, ge=0)


class PipelineFailure(RuntimeError):
    """Expected pipeline failure with a client-safe structured error."""

    def __init__(self, error: AnalysisError):
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


class OutputValidationError(ValueError):
    """Safe domain-validation signal used inside the bounded LLM budget."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.safe_message = message
        super().__init__(message)
