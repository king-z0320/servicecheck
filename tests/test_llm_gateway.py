import json

import pytest
import requests

from qc.errors import PipelineFailure
from qc.llm_gateway import DeepSeekGateway


SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


class FakeResponse:
    def __init__(self, status_code=200, content=None, error_message=""):
        self.status_code = status_code
        self.headers = {}
        self._content = content
        self._error_message = error_message

    def json(self):
        if self.status_code >= 400:
            return {"error": {"message": self._error_message}}
        return {
            "choices": [
                {"message": {"content": self._content}}
            ]
        }


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def gateway_for(responses, sleeper=lambda _: None):
    session = FakeSession(responses)
    gateway = DeepSeekGateway(
        "top-secret-api-key",
        session=session,
        sleeper=sleeper,
    )
    return gateway, session


def complete(gateway, validate=lambda data: data):
    return gateway.complete_json(
        system="system",
        user="user",
        schema=SCHEMA,
        validate=validate,
    )


def test_timeout_then_success_uses_exactly_two_network_attempts():
    gateway, session = gateway_for(
        [
            requests.Timeout("secret timeout detail"),
            FakeResponse(content=json.dumps({"value": "ok"})),
        ]
    )

    assert complete(gateway) == {"value": "ok"}
    assert len(session.calls) == 2


@pytest.mark.parametrize(
    ("status", "expected_code", "expected_attempts"),
    [
        (500, "LLM_UPSTREAM_ERROR", 2),
        (503, "LLM_UPSTREAM_ERROR", 2),
        (429, "LLM_RATE_LIMITED", 2),
        (401, "LLM_AUTH_FAILED", 1),
        (403, "LLM_AUTH_FAILED", 1),
        (400, "LLM_UPSTREAM_ERROR", 1),
    ],
)
def test_http_failures_have_bounded_attempts_and_typed_errors(
    status,
    expected_code,
    expected_attempts,
):
    gateway, session = gateway_for(
        [FakeResponse(status, error_message="upstream secret body")] * 2
    )

    with pytest.raises(PipelineFailure) as captured:
        complete(gateway)

    assert len(session.calls) == expected_attempts
    assert captured.value.error.code == expected_code
    assert captured.value.error.attempts == expected_attempts
    serialized = captured.value.error.model_dump_json().lower()
    assert "top-secret-api-key" not in serialized
    assert "upstream secret body" not in serialized
    assert "authorization" not in serialized


def test_schema_capability_fallback_stays_inside_two_request_budget():
    gateway, session = gateway_for(
        [
            FakeResponse(
                400,
                error_message="This response_format type is unavailable now",
            ),
            FakeResponse(content=json.dumps({"value": "fallback"})),
        ]
    )

    assert complete(gateway) == {"value": "fallback"}
    assert len(session.calls) == 2
    assert session.calls[0]["json"]["response_format"]["type"] == "json_schema"
    assert session.calls[1]["json"]["response_format"]["type"] == "json_object"


def test_invalid_json_is_repaired_once_inside_two_request_budget():
    gateway, session = gateway_for(
        [
            FakeResponse(content="not-json"),
            FakeResponse(content=json.dumps({"value": "repaired"})),
        ]
    )

    assert complete(gateway) == {"value": "repaired"}
    assert len(session.calls) == 2
    assert "修复" in session.calls[1]["json"]["messages"][-1]["content"]


def test_domain_validation_is_repaired_once_then_fails():
    gateway, session = gateway_for(
        [
            FakeResponse(content=json.dumps({"value": "bad"})),
            FakeResponse(content=json.dumps({"value": "still-bad"})),
        ]
    )

    def validate(data):
        if data["value"] != "ok":
            raise ValueError("private domain detail")
        return data

    with pytest.raises(PipelineFailure) as captured:
        complete(gateway, validate=validate)

    assert len(session.calls) == 2
    assert captured.value.error.code == "LLM_INVALID_OUTPUT"
    assert captured.value.error.attempts == 2
    assert "private domain detail" not in captured.value.error.message


def test_missing_choices_is_not_interpreted_as_valid_empty_output():
    class MissingChoicesResponse(FakeResponse):
        def json(self):
            return {}

    gateway, session = gateway_for(
        [MissingChoicesResponse(), MissingChoicesResponse()]
    )

    with pytest.raises(PipelineFailure) as captured:
        complete(gateway)

    assert len(session.calls) == 2
    assert captured.value.error.code == "LLM_INVALID_OUTPUT"
