import pytest

from qc.errors import AnalysisError, ErrorStage
from qc.models import AnalysisResult, QualityReport, ReviewDisposition


CALL_STARTED_AT = "2025-10-15T10:25:11+08:00"


class FakeService:
    def analyze(self, request):
        return AnalysisResult(
            runId="RUN-001",
            status="COMPLETED",
            loopUsed=False,
            report=QualityReport(callId=request.callId),
        )

    def get_run(self, run_id):
        if run_id == "MISSING":
            raise KeyError(run_id)
        return {"runId": run_id, "status": "COMPLETED"}


def test_new_analyze_endpoint_returns_run_and_report(app_factory):
    client = app_factory(FakeService()).test_client()
    response = client.post(
        "/api/agent/analyze",
        json={
            "caseId": "CASE",
            "callId": "CALL",
            "callStartedAt": CALL_STARTED_AT,
            "transcript": [
                {
                    "turnId": "T0001",
                    "speaker": "客户",
                    "text": "测试",
                    "start": 0,
                    "end": 1,
                }
            ],
        },
    )
    assert response.status_code == 200
    assert response.json["runId"] == "RUN-001"
    assert response.json["loopUsed"] is False
    assert response.json["report"]["callId"] == "CALL"


def test_get_run_endpoint_returns_persisted_state(app_factory):
    client = app_factory(FakeService()).test_client()
    response = client.get("/api/agent/runs/RUN-001")
    assert response.status_code == 200
    assert response.json == {"runId": "RUN-001", "status": "COMPLETED"}


def test_get_run_endpoint_returns_404_for_unknown_run(app_factory):
    client = app_factory(FakeService()).test_client()
    response = client.get("/api/agent/runs/MISSING")
    assert response.status_code == 404
    assert response.json["error"] == "run not found"


def test_legacy_endpoint_remains_compatible(app_factory):
    client = app_factory(FakeService()).test_client()
    response = client.post(
        "/api/analyze",
        json={
            "transcript": [
                {"speaker": "客户", "text": "测试", "start": 0, "end": 1}
            ]
        },
    )
    assert response.status_code == 200
    assert set(response.json) == {"success", "summary", "qcReport"}


def test_new_endpoint_returns_400_for_invalid_payload(app_factory):
    client = app_factory(FakeService()).test_client()
    response = client.post("/api/agent/analyze", json={"caseId": "CASE"})
    assert response.status_code == 400
    assert response.json["error"] == "invalid request"
    assert response.json["status"] == "FAILED"
    assert set(response.json["errors"][0]) == {
        "code",
        "stage",
        "message",
        "retryable",
        "attempts",
    }


def test_new_endpoint_requires_explicit_call_started_at_before_service_call(
    app_factory,
):
    class SpyService(FakeService):
        called = False

        def analyze(self, request):
            self.called = True
            return super().analyze(request)

    service = SpyService()
    client = app_factory(service).test_client()
    response = client.post(
        "/api/agent/analyze",
        json={
            "caseId": "CASE",
            "callId": "CALL",
            "transcript": [
                {
                    "turnId": "T0001",
                    "speaker": "客户",
                    "text": "测试",
                    "start": 0,
                    "end": 1,
                }
            ],
        },
    )

    assert response.status_code == 400
    assert service.called is False


def test_new_endpoint_classifies_naive_call_started_at_as_call_time_error(
    app_factory,
):
    client = app_factory(FakeService()).test_client()
    response = client.post(
        "/api/agent/analyze",
        json={
            "caseId": "CASE",
            "callId": "CALL",
            "callStartedAt": "2025-10-15T10:25:11",
            "transcript": [
                {
                    "turnId": "T0001",
                    "speaker": "customer",
                    "text": "test",
                    "start": 0,
                    "end": 1,
                }
            ],
        },
    )

    assert response.status_code == 400
    assert response.json["errors"][0]["code"] == "INVALID_CALL_STARTED_AT"


def test_legacy_endpoint_returns_400_for_empty_transcript(app_factory):
    client = app_factory(FakeService()).test_client()
    response = client.post("/api/analyze", json={"transcript": []})
    assert response.status_code == 400
    assert response.json["error"] == "invalid request"


def failure(code, stage=ErrorStage.EVENT_EXTRACTION):
    return AnalysisError(
        code=code,
        stage=stage,
        message="安全错误摘要",
        retryable=True,
        attempts=2,
    )


@pytest.mark.parametrize(
    ("code", "expected_http"),
    [
        ("LLM_TIMEOUT", 503),
        ("LLM_AUTH_FAILED", 503),
        ("RAG_INDEX_NOT_BUILT", 500),
        ("INTERNAL_ERROR", 500),
    ],
)
def test_failed_result_uses_error_aware_http_mapping(
    app_factory,
    code,
    expected_http,
):
    class FailedService(FakeService):
        def analyze(self, request):
            return AnalysisResult(
                runId="RUN-FAILED",
                status="FAILED",
                loopUsed=False,
                report=None,
                errors=[failure(code)],
            )

    response = app_factory(FailedService()).test_client().post(
        "/api/agent/analyze",
        json={
            "caseId": "CASE",
            "callId": "CALL",
            "callStartedAt": CALL_STARTED_AT,
            "transcript": [
                {
                    "turnId": "T0001",
                    "speaker": "客户",
                    "text": "测试",
                    "start": 0,
                    "end": 1,
                }
            ],
        },
    )

    assert response.status_code == expected_http
    assert response.json["status"] == "FAILED"
    assert response.json["errors"][0]["code"] == code


def test_partial_result_is_200_and_never_claims_auto_pass(app_factory):
    class PartialService(FakeService):
        def analyze(self, request):
            return AnalysisResult(
                runId="RUN-PARTIAL",
                status="PARTIAL",
                loopUsed=True,
                report=QualityReport(
                    callId=request.callId,
                    disposition=ReviewDisposition.HUMAN_REVIEW_REQUIRED,
                ),
                errors=[failure("RAG_WEAK_SUPPORT", ErrorStage.RAG)],
            )

    response = app_factory(PartialService()).test_client().post(
        "/api/agent/analyze",
        json={
            "caseId": "CASE",
            "callId": "CALL",
            "callStartedAt": CALL_STARTED_AT,
            "transcript": [
                {
                    "turnId": "T0001",
                    "speaker": "客户",
                    "text": "测试",
                    "start": 0,
                    "end": 1,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json["status"] == "PARTIAL"
    assert response.json["report"]["disposition"] == "HUMAN_REVIEW_REQUIRED"
#这个测试文件的作用是：测试 api_server.py 中的 Flask API 接口是否能正确处理请求和返回响应，包括新接口 /api/agent/analyze、/api/agent/runs/<run_id> 和旧接口 /api/analyze 的兼容性，以及对无效请求的处理。
#测试方法：使用 pytest 框架，通过创建测试客户端来发送 HTTP 请求，并断言返回的响应是否符合预期。
