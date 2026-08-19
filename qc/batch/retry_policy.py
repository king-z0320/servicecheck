from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from qc.batch.models import StageName
from qc.errors import PipelineFailure


@dataclass(frozen=True, slots=True)
class StageFailure:
    code: str
    stage: StageName
    message: str
    retryable: bool
    cause_type: str = "unknown"


def classify_error(exc: Exception, stage: StageName) -> StageFailure:
    """Map an exception to a safe, stage-local retry decision."""
    if all(hasattr(exc, name) for name in ("code", "message", "retryable")):
        return StageFailure(
            code=str(getattr(exc, "code")),
            stage=stage,
            message=str(getattr(exc, "message")),
            retryable=bool(getattr(exc, "retryable")),
            cause_type="structured",
        )
    if isinstance(exc, PipelineFailure):
        error = exc.error
        return StageFailure(
            code=str(error.code),
            stage=stage,
            message=error.message,
            retryable=bool(error.retryable),
            cause_type="pipeline",
        )
    if isinstance(exc, TimeoutError):
        message = str(exc).lower()
        if "429" in message or "rate" in message or "limit" in message:
            return StageFailure("UPSTREAM_RATE_LIMITED", stage, "上游限流", True, "rate_limit")
        return StageFailure("UPSTREAM_TIMEOUT", stage, "上游调用超时", True, "timeout")
    if isinstance(exc, (FileNotFoundError, PermissionError, ValueError)):
        return StageFailure("INVALID_INPUT", stage, "输入或路径不合法", False, "input")
    return StageFailure("INTERNAL_ERROR", stage, "批量阶段发生内部错误", False, "internal")


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.1
    random_fn: Callable[[], float] = random.random

    def should_retry(self, failure: StageFailure, attempts: int) -> bool:
        return bool(failure.retryable and attempts < max(1, self.max_attempts))

    def delay_for(self, attempts: int) -> float:
        base = min(
            max(0.0, self.initial_delay) * (2 ** max(0, attempts - 1)),
            max(0.0, self.max_delay),
        )
        return base + max(0.0, self.jitter) * self.random_fn()
