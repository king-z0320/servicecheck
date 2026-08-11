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
    assert "turnId: turn.turnId || `T${String(index + 1).padStart(4, '0')}`" in html
    assert "renderAgentAnalysis(result)" in html
    assert "result.report" in html
