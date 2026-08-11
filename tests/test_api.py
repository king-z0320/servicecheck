from qc.models import AnalysisResult, QualityReport


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


def test_legacy_endpoint_returns_400_for_empty_transcript(app_factory):
    client = app_factory(FakeService()).test_client()
    response = client.post("/api/analyze", json={"transcript": []})
    assert response.status_code == 400
    assert response.json["error"] == "invalid request"
#这个测试文件的作用是：测试 api_server.py 中的 Flask API 接口是否能正确处理请求和返回响应，包括新接口 /api/agent/analyze、/api/agent/runs/<run_id> 和旧接口 /api/analyze 的兼容性，以及对无效请求的处理。
#测试方法：使用 pytest 框架，通过创建测试客户端来发送 HTTP 请求，并断言返回的响应是否符合预期。