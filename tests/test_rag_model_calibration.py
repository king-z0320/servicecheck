from pathlib import Path

import pytest

from scripts.calibrate_rag import calibrate


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.rag_model
def test_real_embedder_has_a_nonzero_separating_threshold():
    result = calibrate(ROOT / "tests/gold/rag_calibration_cases.json")

    assert result["threshold"] > 0
    assert result["negativeMax"] < result["threshold"] < result["positiveMin"]
    assert result["positiveCount"] >= 6
    assert result["negativeCount"] >= 6
    assert result["excludedCaseCount"] >= 2
