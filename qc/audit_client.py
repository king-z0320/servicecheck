from __future__ import annotations

from time import sleep
from typing import Callable

import requests

from qc.errors import AnalysisError, ErrorStage
from qc.models import AuditSnapshot


class AuditClient:
    _TRANSIENT_STATUSES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str,
        session=None,
        timeout: float = 5,
        *,
        sleeper: Callable[[float], None] = sleep,
        retry_delay_seconds: float = 0.1,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.sleeper = sleeper
        self.retry_delay_seconds = retry_delay_seconds

    def _get_with_retry(
        self,
        call_id: str,
        resource: str,
    ) -> tuple[dict | None, AnalysisError | None]:
        for attempt in (1, 2):
            try:
                response = self.session.get(
                    f"{self.base_url}/mock/calls/{call_id}/{resource}",
                    timeout=self.timeout,
                )
            except requests.Timeout:
                if attempt < 2:
                    self.sleeper(self.retry_delay_seconds)
                    continue
                return None, self._error(
                    "AUDIT_TIMEOUT",
                    "审计服务请求超时",
                    True,
                    attempt,
                )
            except requests.RequestException:
                if attempt < 2:
                    self.sleeper(self.retry_delay_seconds)
                    continue
                return None, self._error(
                    "AUDIT_UPSTREAM_ERROR",
                    "审计服务暂时不可用",
                    True,
                    attempt,
                )

            status = int(response.status_code)
            if status in self._TRANSIENT_STATUSES:
                if attempt < 2:
                    self.sleeper(self.retry_delay_seconds)
                    continue
                return None, self._error(
                    "AUDIT_UPSTREAM_ERROR",
                    "审计服务暂时不可用",
                    True,
                    attempt,
                )
            if status in {401, 403}:
                return None, self._error(
                    "AUDIT_AUTH_FAILED",
                    "审计服务鉴权失败",
                    False,
                    attempt,
                )
            if status == 404:
                return None, self._error(
                    "AUDIT_NOT_FOUND",
                    "审计资源不存在",
                    False,
                    attempt,
                )
            if status >= 400:
                return None, self._error(
                    "AUDIT_UPSTREAM_ERROR",
                    "审计服务拒绝了请求",
                    False,
                    attempt,
                )
            try:
                data = response.json()
            except (TypeError, ValueError):
                data = None
            if not isinstance(data, dict):
                return None, self._error(
                    "AUDIT_INVALID_RESPONSE",
                    "审计服务返回的数据结构无效",
                    False,
                    attempt,
                )
            return data, None

        raise AssertionError("bounded audit retry loop did not terminate")

    @staticmethod
    def _error(
        code: str,
        message: str,
        retryable: bool,
        attempts: int,
    ) -> AnalysisError:
        return AnalysisError(
            code=code,
            stage=ErrorStage.AUDIT,
            message=message,
            retryable=retryable,
            attempts=attempts,
        )

    def fetch_snapshot(self, call_id: str) -> AuditSnapshot:
        values: dict[str, dict] = {}
        errors: list[AnalysisError] = []
        resources = (
            "crm-summary",
            "dispute-tickets",
            "follow-up-tasks",
            "agent-actions",
        )
        for resource in resources:
            value, error = self._get_with_retry(call_id, resource)
            if value is not None:
                values[resource] = value
            if error is not None:
                errors.append(error)

        tickets = values.get("dispute-tickets", {}).get("tickets")
        tasks = values.get("follow-up-tasks", {}).get("tasks", [])
        return AuditSnapshot(
            callId=call_id,
            crmSummary=values.get("crm-summary", {}).get("summary"),
            disputeTicketCreated=(
                None if tickets is None else bool(tickets)
            ),
            followUpType=(
                tasks[0].get("type")
                if tasks and isinstance(tasks[0], dict)
                else None
            ),
            actions=values.get("agent-actions", {}).get("actions", []),
            errors=errors,
        )
