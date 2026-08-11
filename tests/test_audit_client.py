from qc.audit_client import AuditClient


class FakeResponse:
    def __init__(self, data, status_code=200):
        self.data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(str(self.status_code))

    def json(self):
        return self.data


class FakeSession:
    def get(self, url, timeout):
        if url.endswith("/crm-summary"):
            return FakeResponse({"summary": "客户拒绝还款"})
        if url.endswith("/dispute-tickets"):
            return FakeResponse({"tickets": []})
        if url.endswith("/follow-up-tasks"):
            return FakeResponse({"tasks": [{"type": "CONTINUE_COLLECTION"}]})
        return FakeResponse({"actions": [{"type": "SAVE_SUMMARY"}]})


def test_fetches_combined_read_only_snapshot():
    snapshot = AuditClient(
        "http://mock",
        session=FakeSession(),
    ).fetch_snapshot("CALL-002")
    assert snapshot.crmSummary == "客户拒绝还款"
    assert snapshot.disputeTicketCreated is False
    assert snapshot.followUpType == "CONTINUE_COLLECTION"


def test_partial_failure_is_exposed_not_silenced():
    class FailingSession(FakeSession):
        def get(self, url, timeout):
            if url.endswith("/dispute-tickets"):
                raise RuntimeError("timeout")
            return super().get(url, timeout)

    snapshot = AuditClient(
        "http://mock",
        session=FailingSession(),
    ).fetch_snapshot("CALL-002")
    assert snapshot.disputeTicketCreated is None
    assert snapshot.errors == ["dispute-tickets: timeout"]
