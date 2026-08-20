from pathlib import Path


HTML_PATH = Path(__file__).resolve().parents[1] / "review.html"
WORKBENCH = Path(__file__).resolve().parents[1] / "催收质检.html"


def test_review_page_loads_pending_list_and_detail_contract():
    html = HTML_PATH.read_text(encoding="utf-8")
    assert "/api/review-tasks" in html
    assert 'id="review-task-list"' in html
    assert 'id="review-detail"' in html
    assert 'id="review-form"' in html
    assert "CONFIRMED_PASS" in html
    assert "CONFIRMED_VIOLATION" in html
    assert "UNRESOLVED" in html
    assert "Idempotency-Key" in html
    assert "REVIEW_VERSION_CONFLICT" in html
    assert "escapeHTML" in html
    assert "localStorage.setItem" not in html
    assert "contenteditable" not in html
    assert "vue" not in html.lower()
    assert "react" not in html.lower()
    assert 'name="score"' not in html
    assert "仍无法安全判定" in html
    assert "人工确认" in html
    assert "originalReport" in html
    assert "readonly" in html


def test_review_page_keeps_draft_on_version_conflict():
    html = HTML_PATH.read_text(encoding="utf-8")
    assert "localDraft" in html
    assert "重新加载" in html
    assert "草稿" in html


def test_workbench_keeps_report_view_and_links_to_review_page():
    html = WORKBENCH.read_text(encoding="utf-8")
    assert "review.html" in html
    assert "function interpretAnalysisState(result)" in html
    assert "POST /api/review-tasks" not in html
