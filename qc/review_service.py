from __future__ import annotations

from typing import Any, Iterable

from qc.errors import AnalysisError, ErrorStage
from qc.models import QualityReport, ReviewDisposition
from qc.review_models import (
    ContextSource,
    ReviewSubmitRequest,
    ReviewerContext,
    RouteReason,
    canonical_submit_hash,
    configured_reviewer_context,
)


class ReviewError(Exception):
    def __init__(self, code: str, message: str, **details: Any):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


class ReviewNotFound(ReviewError):
    def __init__(self, review_task_id: str):
        super().__init__("REVIEW_NOT_FOUND", "review task not found", reviewTaskId=review_task_id)


class ReviewVersionConflict(ReviewError):
    def __init__(self, current_version: int):
        super().__init__(
            "REVIEW_VERSION_CONFLICT",
            "review task version conflict",
            currentVersion=current_version,
        )


class ReviewIdempotencyConflict(ReviewError):
    def __init__(self):
        super().__init__("IDEMPOTENCY_CONFLICT", "idempotency key reused with a different payload")


class ReviewStateConflict(ReviewError):
    def __init__(self, status: str):
        super().__init__(
            "REVIEW_STATE_CONFLICT",
            "review task is no longer pending",
            status=status,
        )


class ReviewValidationError(ReviewError):
    def __init__(self, message: str):
        super().__init__("INVALID_REQUEST", message)


def needs_review_task(status: str, report: QualityReport | None) -> bool:
    if report is None:
        return False
    if status == "FAILED":
        return False
    if status == "COMPLETED" and report.disposition in {
        ReviewDisposition.AUTO_PASS,
        ReviewDisposition.AUTO_VIOLATION,
    }:
        return False
    if status == "PARTIAL":
        return True
    return report.disposition == ReviewDisposition.HUMAN_REVIEW_REQUIRED


def compute_route_reasons(
    status: str,
    report: QualityReport | None,
    errors: Iterable[AnalysisError] | None = None,
) -> list[RouteReason]:
    reasons: list[RouteReason] = []
    seen: set[tuple[str, str, str | None, str | None]] = set()

    def add(
        code: str,
        stage: str,
        event_id: str | None = None,
        rule_id: str | None = None,
        report_path: str | None = None,
    ) -> None:
        key = (code, stage, event_id, rule_id)
        if key in seen:
            return
        seen.add(key)
        reasons.append(
            RouteReason(
                code=code,
                stage=stage,
                eventId=event_id,
                ruleId=rule_id,
                reportPath=report_path,
            )
        )

    for error in errors or []:
        stage = error.stage.value if isinstance(error.stage, ErrorStage) else str(error.stage)
        add(error.code, stage)

    if report is not None:
        for event in report.events:
            if event.ambiguous:
                add("AMBIGUOUS_EVENT", ErrorStage.EVENT_EXTRACTION.value, event_id=event.eventId)
        if report.auditSnapshot is not None:
            for error in report.auditSnapshot.errors:
                stage = error.stage.value if isinstance(error.stage, ErrorStage) else str(error.stage)
                add(error.code, stage)
            if report.auditSnapshot.errors:
                add("AUDIT_ERROR_REQUIRES_REVIEW", ErrorStage.AUDIT.value)
        pending = report.summary.get("pendingReviewIssues") or []
        for item in pending:
            if isinstance(item, dict):
                add(
                    str(item.get("code") or "QUALITY_GATE_FAILED"),
                    ErrorStage.QUALITY_GATE.value,
                    event_id=item.get("eventId"),
                    rule_id=item.get("ruleId"),
                    report_path="summary.pendingReviewIssues",
                )
            else:
                add(str(item), ErrorStage.QUALITY_GATE.value)
        if report.disposition == ReviewDisposition.HUMAN_REVIEW_REQUIRED and not reasons:
            add("HUMAN_REVIEW_REQUIRED", ErrorStage.QUALITY_GATE.value)

    if status == "PARTIAL" and not reasons:
        add("HUMAN_REVIEW_REQUIRED", ErrorStage.QUALITY_GATE.value)
    return reasons


class ReviewService:
    def __init__(self, store):
        self.store = store

    def list_tasks(
        self,
        status: str | None = "PENDING",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = min(100, max(1, int(page_size)))
        if status is not None and status not in {"PENDING", "RESOLVED", "UNRESOLVED"}:
            raise ReviewValidationError("invalid review task status")
        return self.store.list_tasks(status=status, page=page, page_size=page_size)

    def get_task(self, review_task_id: str) -> dict[str, Any]:
        detail = self.store.get_task_detail(review_task_id)
        if detail is None:
            raise ReviewNotFound(review_task_id)
        return detail

    def submit(
        self,
        review_task_id: str,
        request: ReviewSubmitRequest,
        idempotency_key: str | None,
        reviewer: ReviewerContext | None = None,
    ) -> dict[str, Any]:
        if not idempotency_key or not str(idempotency_key).strip():
            raise ReviewValidationError("Idempotency-Key is required")
        key = str(idempotency_key).strip()
        if len(key) > 256:
            raise ReviewValidationError("Idempotency-Key is too long")
        context = reviewer or configured_reviewer_context()
        if context.contextSource != ContextSource.CONFIGURED_DEMO:
            context = configured_reviewer_context()
        request_hash = canonical_submit_hash(
            request.outcome.value,
            request.reasonCode.value,
            request.note,
        )
        return self.store.submit(
            review_task_id=review_task_id,
            expected_version=request.expectedVersion,
            outcome=request.outcome.value,
            reason_code=request.reasonCode.value,
            note=request.note,
            idempotency_key=key,
            request_hash=request_hash,
            reviewer_id=context.reviewerId,
            context_source=context.contextSource.value,
        )

    def summaries_for_run(self, run_id: str) -> dict[str, Any]:
        return self.store.summaries_for_run(run_id)
