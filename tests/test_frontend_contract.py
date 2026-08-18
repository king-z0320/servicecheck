from pathlib import Path


HTML_PATH = Path(__file__).resolve().parents[1] / "催收质检.html"


def test_frontend_calls_agent_endpoint_and_renders_trace_panels():
    html = HTML_PATH.read_text(encoding="utf-8")
    assert "/api/agent/analyze" in html
    assert 'id="analysis-route"' in html
    assert 'id="quality-events"' in html
    assert 'id="agent-analysis-trace"' in html
    assert 'id="knowledge-evidence"' in html
    assert 'id="action-audit"' in html
    assert 'id="analysis-disposition"' in html
    assert "Agent Loop：未启动" in html


def test_frontend_sends_stable_call_context_and_uses_structured_report():
    html = HTML_PATH.read_text(encoding="utf-8")
    assert "caseId: caseData.caseInfo.id" in html
    assert "callId: caseData.caseInfo.callId || `CALL-${caseId}`" in html
    assert "callStartedAt: resolveCallStartedAt(caseData.caseInfo)" in html
    assert "function resolveCallStartedAt(caseInfo)" in html
    assert "+08:00" in html
    assert "new Date().toISOString()" not in html
    assert "turnId: turn.turnId || `T${String(index + 1).padStart(4, '0')}`" in html
    assert "renderAgentAnalysis(result)" in html
    assert "result.report" in html


def test_frontend_interprets_status_before_violations_and_preserves_old_result():
    html = HTML_PATH.read_text(encoding="utf-8")
    assert "function interpretAnalysisState(result)" in html
    assert "result.status === 'FAILED'" in html
    assert "result.status === 'PARTIAL'" in html
    assert "disposition === 'HUMAN_REVIEW_REQUIRED'" in html
    assert "result.status === 'COMPLETED' && disposition === 'AUTO_PASS'" in html
    assert "result.status === 'COMPLETED' && disposition === 'AUTO_VIOLATION'" in html
    assert "证据不完整，需要人工复核" in html
    assert "分析失败" in html
    assert "if (!state.canApplyReport)" in html
    assert "runState?.status === 'PARTIAL'" in html
    assert "证据不完整，不能按无违规自动通过" in html
    assert "error.code" in html
    assert "error.message" in html


def test_frontend_marks_rule_and_unconnected_actions_as_demo_only():
    html = HTML_PATH.read_text(encoding="utf-8")
    assert "演示功能：仅修改当前浏览器内存，不会写入后端规则库，刷新后丢失" in html
    assert "规则已添加（仅演示，未保存到后端）" in html
    assert "规则已更新（仅演示，未保存到后端）" in html
    assert "规则已删除（仅演示，未保存到后端）" in html
    assert "仅演示，未提交到后端" in html
    assert "仅演示，未同步到 CRM" in html
    assert "演示视图，未接后端" in html
    assert "Object.entries(LEGACY_DEMO_TEMPLATES)" in html


def test_agent_workbench_uses_backend_as_its_only_fact_source():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "const CASE_CACHE = {};" in html
    assert "MOCK_DATA" not in html
    for path in (
        "/api/cases",
        "/api/cases/",
        "/api/calls/",
        "/transcript",
        "/runs",
        "/api/reports/",
        "/audio",
    ):
        assert path in html
    assert "async function initializeWorkbench()" in html


def test_frontend_restores_and_switches_immutable_run_history():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'id="run-history"' in html
    assert "function renderRunHistory" in html
    assert "async function selectHistoricalRun" in html
    assert "new URLSearchParams(window.location.search)" in html
    assert "runId" in html
    assert "history.replaceState" in html
