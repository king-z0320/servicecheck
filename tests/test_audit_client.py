import pytest
import requests

from qc.audit_client import AuditClient


class FakeResponse:
    def __init__(self, data=None, status_code=200):
        self.data = data if data is not None else {}
        self.status_code = status_code

    def json(self):
        return self.data


class ScriptedSession:
    def __init__(self, scripts=None):
        self.scripts = {key: list(value) for key, value in (scripts or {}).items()}
        self.calls = []

    def get(self, url, timeout):
        resource = url.rsplit("/", 1)[-1]
        self.calls.append(resource)
        scripted = self.scripts.get(resource)
        if scripted:
            item = scripted.pop(0)
        else:
            item = self._default(resource)
        if isinstance(item, Exception):
            raise item
        return item

    @staticmethod
    def _default(resource):
        values = {
            "crm-summary": {"summary": "客户拒绝还款"},
            "dispute-tickets": {"tickets": []},
            "follow-up-tasks": {"tasks": [{"type": "CONTINUE_COLLECTION"}]},
            "agent-actions": {"actions": [{"type": "SAVE_SUMMARY"}]},
        }
        return FakeResponse(values[resource])


def test_fetches_combined_read_only_snapshot():
    snapshot = AuditClient(
        "http://mock",
        session=ScriptedSession(),
    ).fetch_snapshot("CALL-002")

    assert snapshot.crmSummary == "客户拒绝还款"
    assert snapshot.disputeTicketCreated is False
    assert snapshot.followUpType == "CONTINUE_COLLECTION"
    assert snapshot.errors == []


def test_timeout_then_success_retries_only_failed_resource():
    session = ScriptedSession(
        {
            "dispute-tickets": [
                requests.Timeout("private timeout detail"),
                FakeResponse({"tickets": [{"id": "D-1"}]}),
            ]
        }
    )

    snapshot = AuditClient(
        "http://mock",
        session=session,
        sleeper=lambda _: None,
    ).fetch_snapshot("CALL-002")

    assert session.calls.count("dispute-tickets") == 2
    assert session.calls.count("crm-summary") == 1
    assert snapshot.disputeTicketCreated is True
    assert snapshot.errors == []


@pytest.mark.parametrize(
    ("status", "code", "attempts"),
    [
        (503, "AUDIT_UPSTREAM_ERROR", 2),
        (429, "AUDIT_UPSTREAM_ERROR", 2),
        (401, "AUDIT_AUTH_FAILED", 1),
        (403, "AUDIT_AUTH_FAILED", 1),
        (404, "AUDIT_NOT_FOUND", 1),
    ],
)
def test_http_failures_are_classified_and_bounded(status, code, attempts):
    session = ScriptedSession(
        {"dispute-tickets": [FakeResponse(status_code=status)] * 2}
    )

    snapshot = AuditClient(
        "http://mock",
        session=session,
        sleeper=lambda _: None,
    ).fetch_snapshot("CALL-002")

    assert session.calls.count("dispute-tickets") == attempts
    assert snapshot.disputeTicketCreated is None
    assert len(snapshot.errors) == 1
    assert snapshot.errors[0].code == code
    assert snapshot.errors[0].stage == "AUDIT"
    assert snapshot.errors[0].attempts == attempts


def test_one_resource_failure_keeps_other_three_successes():
    session = ScriptedSession(
        {
            "dispute-tickets": [
                requests.Timeout("private timeout detail"),
                requests.Timeout("private timeout detail"),
            ]
        }
    )

    snapshot = AuditClient(
        "http://mock",
        session=session,
        sleeper=lambda _: None,
    ).fetch_snapshot("CALL-002")

    assert snapshot.crmSummary == "客户拒绝还款"
    assert snapshot.disputeTicketCreated is None
    assert snapshot.followUpType == "CONTINUE_COLLECTION"
    assert snapshot.actions == [{"type": "SAVE_SUMMARY"}]
    assert snapshot.errors[0].code == "AUDIT_TIMEOUT"
    assert snapshot.errors[0].attempts == 2
    assert "private timeout detail" not in snapshot.errors[0].model_dump_json()


def test_invalid_json_shape_is_typed_and_not_retried():
    session = ScriptedSession(
        {"crm-summary": [FakeResponse(["not", "an", "object"])]}
    )

    snapshot = AuditClient(
        "http://mock",
        session=session,
        sleeper=lambda _: None,
    ).fetch_snapshot("CALL-002")

    assert session.calls.count("crm-summary") == 1
    assert snapshot.crmSummary is None
    assert snapshot.errors[0].code == "AUDIT_INVALID_RESPONSE"
    assert snapshot.errors[0].attempts == 1
