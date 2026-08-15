import json

import pytest

from qc.config import CALIBRATION_PATH, Settings, calibrated_support_score


def test_versioned_calibration_artifact_has_real_separating_threshold():
    artifact = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    threshold = calibrated_support_score()

    assert artifact["embedder"] == "BAAI/bge-small-zh-v1.5"
    assert artifact["purpose"].endswith("not violation classification")
    assert artifact["negativeMax"] < threshold < artifact["positiveMin"]
    assert artifact["positiveCount"] >= 6
    assert artifact["negativeCount"] >= 6


def test_settings_use_calibrated_default_and_validate_override(monkeypatch):
    monkeypatch.delenv("RAG_MIN_SUPPORT_SCORE", raising=False)
    assert Settings.from_env().rag_min_support_score == calibrated_support_score()

    monkeypatch.setenv("RAG_MIN_SUPPORT_SCORE", "1.1")
    with pytest.raises(ValueError):
        Settings.from_env()
