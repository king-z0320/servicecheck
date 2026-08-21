import pytest
from fastapi import FastAPI

from api_server import create_app
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
    client = app_factory(FakeService())
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
    assert response.json()["runId"] == "RUN-001"
    assert response.json()["loopUsed"] is False
    assert response.json()["report"]["callId"] == "CALL"


def test_get_run_endpoint_returns_persisted_state(app_factory):
    client = app_factory(FakeService())
    response = client.get("/api/agent/runs/RUN-001")
    assert response.status_code == 200
    assert response.json() == {"runId": "RUN-001", "status": "COMPLETED"}


def test_get_run_endpoint_returns_404_for_unknown_run(app_factory):
    client = app_factory(FakeService())
    response = client.get("/api/agent/runs/MISSING")
    assert response.status_code == 404
    assert response.json()["error"] == "run not found"


def test_metrics_endpoint_is_available_without_business_identifiers(app_factory):
    response = app_factory(FakeService()).get("/metrics")
    assert response.status_code == 200
    assert "servicecheck_stage_duration_seconds" in response.text
    assert "run_id" not in response.text


def test_legacy_endpoint_remains_compatible(app_factory):
    client = app_factory(FakeService())
    response = client.post(
        "/api/analyze",
        json={
            "transcript": [
                {"speaker": "客户", "text": "测试", "start": 0, "end": 1}
            ]
        },
    )
    assert response.status_code == 200
    assert set(response.json()) == {"success", "summary", "qcReport"}


def test_new_endpoint_returns_400_for_invalid_payload(app_factory):
    client = app_factory(FakeService())
    response = client.post("/api/agent/analyze", json={"caseId": "CASE"})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid request"
    assert response.json()["status"] == "FAILED"
    assert set(response.json()["errors"][0]) == {
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
    client = app_factory(service)
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
    client = app_factory(FakeService())
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
    assert response.json()["errors"][0]["code"] == "INVALID_CALL_STARTED_AT"


def test_legacy_endpoint_returns_400_for_empty_transcript(app_factory):
    client = app_factory(FakeService())
    response = client.post("/api/analyze", json={"transcript": []})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid request"


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

    response = app_factory(FailedService()).post(
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
    assert response.json()["status"] == "FAILED"
    assert response.json()["errors"][0]["code"] == code


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

    response = app_factory(PartialService()).post(
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
    assert response.json()["status"] == "PARTIAL"
    assert response.json()["report"]["disposition"] == "HUMAN_REVIEW_REQUIRED"


def test_application_factory_returns_fastapi():
    assert isinstance(create_app(service=FakeService()), FastAPI)


def test_malformed_json_returns_contract_400_instead_of_fastapi_422(app_factory):
    response = app_factory(FakeService()).post(
        "/api/agent/analyze",
        content="{not-json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid request"
    assert response.json()["status"] == "FAILED"


def test_openapi_documents_compatible_routes_and_success_models(app_factory):
    schema = app_factory(FakeService()).get("/openapi.json").json()

    assert {
        "/api/health",
        "/api/agent/analyze",
        "/api/agent/runs/{run_id}",
        "/api/analyze",
    }.issubset(schema["paths"])
    analyze_post = schema["paths"]["/api/agent/analyze"]["post"]
    assert "requestBody" in analyze_post
    assert "schema" in analyze_post["responses"]["200"]["content"]["application/json"]
    assert {"200", "400", "500", "503"}.issubset(analyze_post["responses"])
    assert "422" not in analyze_post["responses"]

    request_schema = analyze_post["requestBody"]["content"]["application/json"]["schema"]
    component_name = request_schema["$ref"].rsplit("/", 1)[-1]
    assert "callStartedAt" in schema["components"]["schemas"][component_name]["required"]


def test_api_module_does_not_import_audio_or_batch_execution_paths():
    source = __import__("inspect").getsource(__import__("api_server"))

    for forbidden in ("process_audio", "FunASR", "BatchOrchestrator", "FileCheckpointSession"):
        assert forbidden not in source


@pytest.mark.parametrize(
    ("transcript", "expected_code"),
    [
        (
            [
                {"turnId": "T1", "speaker": "客户", "text": "一", "start": 0, "end": 1},
                {"turnId": "T1", "speaker": "坐席", "text": "二", "start": 1, "end": 2},
            ],
            "DUPLICATE_TURN_ID",
        ),
        (
            [{"turnId": "T1", "speaker": "客户", "text": "  ", "start": 0, "end": 1}],
            "EMPTY_TURN_TEXT",
        ),
        (
            [{"turnId": "T1", "speaker": "客户", "text": "测试", "start": 2, "end": 1}],
            "INVALID_TURN_TIME",
        ),
    ],
)
def test_agent_validation_error_codes_remain_compatible(
    app_factory,
    transcript,
    expected_code,
):
    response = app_factory(FakeService()).post(
        "/api/agent/analyze",
        json={
            "caseId": "CASE",
            "callId": "CALL",
            "callStartedAt": CALL_STARTED_AT,
            "transcript": transcript,
        },
    )

    assert response.status_code == 400
    assert response.json()["errors"][0]["code"] == expected_code


def test_flask_is_not_a_runtime_dependency():
    from pathlib import Path

    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    api_source = Path("api_server.py").read_text(encoding="utf-8").lower()

    assert "flask" not in requirements
    assert "from flask" not in api_source


def test_production_service_assembly_uses_postgres_not_sqlite():
    import ast
    import inspect
    import textwrap
    from pathlib import Path

    import api_server

    source = Path("api_server.py").read_text(encoding="utf-8")

    assert "from qc.postgres_run_store import PostgresRunStore" in source
    assert "from qc.run_store import RunStore" not in source
    assert 'ROOT / "data/qc_runs.db"' not in source

    build_tree = ast.parse(textwrap.dedent(inspect.getsource(api_server.build_service)))
    build_function = build_tree.body[0]
    assert isinstance(build_function.body[-1], ast.Return)
    assert isinstance(build_function.body[-1].value, ast.Call)
    assert build_function.body[-1].value.func.id == "QualityAnalysisService"
