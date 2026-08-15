import pytest
from pydantic import ValidationError


def test_analysis_error_rejects_negative_attempts():
    from qc.errors import AnalysisError, ErrorStage

    with pytest.raises(ValidationError):
        AnalysisError(
            code="LLM_TIMEOUT",
            stage=ErrorStage.EVENT_EXTRACTION,
            message="大模型请求超时",
            retryable=True,
            attempts=-1,
        )


def test_pipeline_failure_exposes_only_structured_error():
    from qc.errors import AnalysisError, ErrorStage, PipelineFailure

    error = AnalysisError(
        code="LLM_AUTH_FAILED",
        stage=ErrorStage.EVENT_EXTRACTION,
        message="大模型鉴权失败",
        retryable=False,
        attempts=1,
    )
    failure = PipelineFailure(error)

    assert failure.error == error
    assert "secret" not in str(failure).lower()
