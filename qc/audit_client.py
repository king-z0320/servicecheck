import requests

from qc.models import AuditSnapshot


class AuditClient:
    def __init__(
        self,
        base_url: str,
        session=None,
        timeout: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def _get(self, call_id: str, resource: str):
        response = self.session.get(
            f"{self.base_url}/mock/calls/{call_id}/{resource}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def fetch_snapshot(self, call_id: str) -> AuditSnapshot:
        values = {}
        errors = []
        resources = (
            "crm-summary",
            "dispute-tickets",
            "follow-up-tasks",
            "agent-actions",
        )
        for resource in resources:
            try:
                values[resource] = self._get(call_id, resource)
            except (requests.RequestException, RuntimeError) as exc:
                errors.append(f"{resource}: {exc}")

        tickets = values.get("dispute-tickets", {}).get("tickets")
        tasks = values.get("follow-up-tasks", {}).get("tasks", [])
        return AuditSnapshot(
            callId=call_id,
            crmSummary=values.get("crm-summary", {}).get("summary"),
            disputeTicketCreated=(
                None if tickets is None else bool(tickets)
            ),
            followUpType=tasks[0]["type"] if tasks else None,
            actions=values.get("agent-actions", {}).get("actions", []),
            errors=errors,
        )
