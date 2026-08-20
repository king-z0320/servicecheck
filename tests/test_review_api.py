from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from qc.errors import AnalysisError, ErrorStage
from qc.models import AnalysisResult, QualityReport, ReviewDisposition
from qc.review_models import ReviewSubmitRequest
from qc.review_service import (
    ReviewIdempotencyConflict,
    ReviewNotFound,
    ReviewStateConflict,
    ReviewValidationError,
    ReviewVersionConflict,
)


class FakeReviewService:
    def __init__(self):
        now = datetime.now(timezone.utc).isoformat()
        self.tasks = {
            "REVIEW-001": {
                "reviewTask": {
                    "reviewTaskId": "REVIEW-001",
                    "runId": "RUN-001",
                    "batchItemId": None,
                    "callId": "CALL-001",
                    "status": "PENDING",
                    "version": 1,
                    "routeReasons": [{"code": "RAG_WEAK_SUPPORT", "stage": "RAG"}],
                    "score": 80,
                    "createdAt": now,
                    "updatedAt": now,
                    "effectiveRevisionId": None,
                },
                "originalReport": {
                    "callId": "CALL-001",
                    "score": 80,
                    "disposition": "HUMAN_REVIEW_REQUIRED",
                    "violations": [],
                    "events": [],
                    "knowledgeHits": [{"documentId": "POLICY-1", "title": "规则"}],
                },
                "transcript": [{"turnId": "T1", "speaker": "客户", "text": "已还完"}],
                "knowledgeHits": [{"documentId": "POLICY-1", "title": "规则"}],
                "auditSnapshot": {"callId": "CALL-001", "errors": []},
                "trace": [{"phase": "FINALIZE", "message": "done"}],
                "errors": [{"code": "RAG_WEAK_SUPPORT"}],
                "effectiveRevision": None,
                "reviewHistory": [],
                "audioAvailable": False,
                "callId": "CALL-001",
                "caseId": "CASE-001",
            }
        }
        self.idempotency = {}

    def list_tasks(self, status="PENDING", page=1, page_size=20):
        items = [
            item["reviewTask"]
            for item in self.tasks.values()
            if status is None or item["reviewTask"]["status"] == status
        ]
        return {
            "items": items[(page - 1) * page_size : page * page_size],
            "page": page,
            "pageSize": page_size,
            "total": len(items),
        }

    def get_task(self, review_task_id: str):
        if review_task_id not in self.tasks:
            raise ReviewNotFound(review_task_id)
        return deepcopy(self.tasks[review_task_id])

    def submit(self, review_task_id, request: ReviewSubmitRequest, idempotency_key, reviewer):
        if not idempotency_key:
            raise ReviewValidationError("Idempotency-Key is required")
        if review_task_id not in self.tasks:
            raise ReviewNotFound(review_task_id)
        key = (review_task_id, idempotency_key)
        payload = {
            "outcome": request.outcome.value,
            "reasonCode": request.reasonCode.value,
            "note": request.note,
        }
        if key in self.idempotency:
            if self.idempotency[key] != payload:
                raise ReviewIdempotencyConflict()
            return deepcopy(self.tasks[review_task_id])
        task = self.tasks[review_task_id]
        summary = task["reviewTask"]
        if summary["status"] != "PENDING":
            raise ReviewStateConflict(summary["status"])
        if request.expectedVersion != summary["version"]:
            raise ReviewVersionConflict(summary["version"])
        revision = {
            "revisionId": "REV-001",
            "taskId": review_task_id,
            "runId": summary["runId"],
            "outcome": request.outcome.value,
            "reasonCode": request.reasonCode.value,
            "note": request.note,
            "reviewerId": reviewer.reviewerId,
            "contextSource": reviewer.contextSource.value,
            "decisionSource": "HUMAN",
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        resolved = request.outcome.value != "UNRESOLVED"
        summary["status"] = "RESOLVED" if resolved else "UNRESOLVED"
        summary["version"] += 1
        summary["effectiveRevisionId"] = revision["revisionId"] if resolved else None
        task["reviewHistory"] = [revision]
        task["effectiveRevision"] = revision if resolved else None
        self.idempotency[key] = payload
        return deepcopy(task)


class FakeService:
    def analyze(self, request):
        return AnalysisResult(
            runId="RUN-001",
            status="COMPLETED",
            loopUsed=False,
            report=QualityReport(callId=request.callId),
        )

    def get_run(self, run_id):
        return {
            "runId": run_id,
            "status": "PARTIAL",
            "result": {"disposition": "HUMAN_REVIEW_REQUIRED", "score": 80},
            "errors": [],
            "reviewTask": {
                "reviewTaskId": "REVIEW-001",
                "runId": run_id,
                "status": "PENDING",
                "version": 1,
            },
            "effectiveRevision": None,
            "reviewHistory": [],
        }


def test_review_list_and_detail_contract(app_factory):
    client = app_factory(FakeService(), review_service=FakeReviewService())
    listed = client.get("/api/review-tasks", params={"status": "PENDING"})
    assert listed.status_code == 200
    assert listed.json()["items"][0]["reviewTaskId"] == "REVIEW-001"
    assert "transcript" not in listed.json()["items"][0]
    detail = client.get("/api/review-tasks/REVIEW-001")
    assert detail.status_code == 200
    body = detail.json()
    assert body["reviewTask"]["status"] == "PENDING"
    assert body["effectiveRevision"] is None
    assert body["originalReport"]["disposition"] == "HUMAN_REVIEW_REQUIRED"
    assert body["transcript"][0]["turnId"] == "T1"
    assert "contenteditable" not in str(body)


def test_submit_three_outcomes_and_protected_fields(app_factory):
    client = app_factory(FakeService(), review_service=FakeReviewService())
    blocked = client.post(
        "/api/review-tasks/REVIEW-001/submit",
        json={
            "expectedVersion": 1,
            "outcome": "CONFIRMED_PASS",
            "reasonCode": "PASS_CONFIRMED",
            "score": 99,
        },
        headers={"Idempotency-Key": "k1"},
    )
    assert blocked.status_code == 400
    passed = client.post(
        "/api/review-tasks/REVIEW-001/submit",
        json={
            "expectedVersion": 1,
            "outcome": "CONFIRMED_PASS",
            "reasonCode": "PASS_CONFIRMED",
            "note": "确认无违规",
        },
        headers={"Idempotency-Key": "k1"},
    )
    assert passed.status_code == 200
    assert passed.json()["effectiveRevision"]["outcome"] == "CONFIRMED_PASS"
    assert passed.json()["originalReport"]["score"] == 80
    replay = client.post(
        "/api/review-tasks/REVIEW-001/submit",
        json={
            "expectedVersion": 1,
            "outcome": "CONFIRMED_PASS",
            "reasonCode": "PASS_CONFIRMED",
            "note": "确认无违规",
        },
        headers={"Idempotency-Key": "k1"},
    )
    assert replay.status_code == 200
    assert replay.json()["reviewHistory"][0]["revisionId"] == "REV-001"
    conflict = client.post(
        "/api/review-tasks/REVIEW-001/submit",
        json={
            "expectedVersion": 1,
            "outcome": "CONFIRMED_PASS",
            "reasonCode": "PASS_CONFIRMED",
            "note": "不同内容",
        },
        headers={"Idempotency-Key": "k1"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "IDEMPOTENCY_CONFLICT"
    assert "originalReport" not in conflict.json()


def test_submit_unresolved_has_no_effective_revision(app_factory):
    client = app_factory(FakeService(), review_service=FakeReviewService())
    response = client.post(
        "/api/review-tasks/REVIEW-001/submit",
        json={
            "expectedVersion": 1,
            "outcome": "UNRESOLVED",
            "reasonCode": "ASR_UNCLEAR",
            "note": "听不清",
        },
        headers={"Idempotency-Key": "k-unresolved"},
    )
    assert response.status_code == 200
    assert response.json()["reviewTask"]["status"] == "UNRESOLVED"
    assert response.json()["effectiveRevision"] is None


def test_version_conflict_and_missing_task(app_factory):
    client = app_factory(FakeService(), review_service=FakeReviewService())
    missing = client.get("/api/review-tasks/REVIEW-MISSING")
    assert missing.status_code == 404
    conflict = client.post(
        "/api/review-tasks/REVIEW-001/submit",
        json={
            "expectedVersion": 9,
            "outcome": "CONFIRMED_VIOLATION",
            "reasonCode": "VIOLATION_CONFIRMED",
        },
        headers={"Idempotency-Key": "k-conflict"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "REVIEW_VERSION_CONFLICT"
    assert conflict.json()["currentVersion"] == 1


def test_old_run_query_keeps_existing_fields_and_adds_review_summary(app_factory):
    client = app_factory(FakeService(), review_service=FakeReviewService())
    response = client.get("/api/agent/runs/RUN-001")
    body = response.json()
    assert body["runId"] == "RUN-001"
    assert body["status"] == "PARTIAL"
    assert "reviewTask" in body
    assert "effectiveRevision" in body
    assert "reviewHistory" in body


def test_missing_idempotency_key_is_400(app_factory):
    client = app_factory(FakeService(), review_service=FakeReviewService())
    response = client.post(
        "/api/review-tasks/REVIEW-001/submit",
        json={
            "expectedVersion": 1,
            "outcome": "CONFIRMED_PASS",
            "reasonCode": "PASS_CONFIRMED",
        },
    )
    assert response.status_code == 400
