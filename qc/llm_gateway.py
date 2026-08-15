from __future__ import annotations

import json
from time import sleep
from typing import Any, Callable, TypeVar

import requests

from qc.errors import (
    AnalysisError,
    ErrorStage,
    OutputValidationError,
    PipelineFailure,
)


T = TypeVar("T")


class DeepSeekGateway:
    """Bounded DeepSeek JSON gateway with safe, typed failure semantics."""

    _JSON_SCHEMA_UNAVAILABLE_HINTS = (
        "response_format type is unavailable",
        "json_schema is unavailable",
    )
    _TRANSIENT_STATUSES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1/chat/completions",
        timeout: float = 60,
        *,
        session=None,
        sleeper: Callable[[float], None] = sleep,
        retry_delay_seconds: float = 0.1,
        max_retry_after_seconds: float = 2.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.session = session or requests.Session()
        self.sleeper = sleeper
        self.retry_delay_seconds = retry_delay_seconds
        self.max_retry_after_seconds = max_retry_after_seconds

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        validate: Callable[[dict[str, Any]], T] | None = None,
        *,
        stage: ErrorStage = ErrorStage.EVENT_EXTRACTION,
    ) -> T | dict[str, Any]:
        validate = validate or (lambda data: data)
        use_json_object = False
        repair_requested = False
        last_validation_error: OutputValidationError | None = None

        for attempt in (1, 2):
            payload = self._payload(
                system=system,
                user=user,
                schema=schema,
                use_json_object=use_json_object,
                repair_requested=repair_requested,
            )
            try:
                response = self.session.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.Timeout as exc:
                if attempt < 2:
                    self.sleeper(self.retry_delay_seconds)
                    continue
                raise self._failure(
                    "LLM_TIMEOUT",
                    stage,
                    "大模型请求超时",
                    True,
                    attempt,
                    exc,
                )
            except requests.RequestException as exc:
                if attempt < 2:
                    self.sleeper(self.retry_delay_seconds)
                    continue
                raise self._failure(
                    "LLM_UPSTREAM_ERROR",
                    stage,
                    "大模型服务暂时不可用",
                    True,
                    attempt,
                    exc,
                )

            status = int(response.status_code)
            if status == 400 and self._is_schema_unavailable(response):
                if attempt < 2:
                    use_json_object = True
                    continue
                raise self._failure(
                    "LLM_UPSTREAM_ERROR",
                    stage,
                    "大模型暂不支持所需结构化输出能力",
                    False,
                    attempt,
                )
            if status in self._TRANSIENT_STATUSES:
                if attempt < 2:
                    self.sleeper(self._retry_delay(response))
                    continue
                code = "LLM_RATE_LIMITED" if status == 429 else "LLM_UPSTREAM_ERROR"
                message = "大模型请求频率受限" if status == 429 else "大模型服务暂时不可用"
                raise self._failure(code, stage, message, True, attempt)
            if status in {401, 403}:
                raise self._failure(
                    "LLM_AUTH_FAILED",
                    stage,
                    "大模型鉴权失败",
                    False,
                    attempt,
                )
            if 400 <= status < 500 or status >= 400:
                raise self._failure(
                    "LLM_UPSTREAM_ERROR",
                    stage,
                    "大模型服务拒绝了请求",
                    False,
                    attempt,
                )

            try:
                envelope = response.json()
                content = envelope["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("missing response content")
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise ValueError("structured response root must be an object")
                return validate(data)
            except OutputValidationError as exc:
                last_validation_error = exc
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                last_validation_error = None

            if attempt < 2:
                repair_requested = True
                use_json_object = True
                continue
            code = (
                last_validation_error.code
                if last_validation_error is not None
                else "LLM_INVALID_OUTPUT"
            )
            message = (
                last_validation_error.safe_message
                if last_validation_error is not None
                else "大模型返回的结构化结果无效"
            )
            raise self._failure(code, stage, message, False, attempt)

        raise AssertionError("bounded attempt loop did not terminate")

    def _payload(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        use_json_object: bool,
        repair_requested: bool,
    ) -> dict[str, Any]:
        schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True)
        system_content = system
        if use_json_object:
            system_content = (
                f"{system}\n\n只输出符合如下 JSON Schema 的 JSON，不要输出解释或 markdown：\n"
                f"{schema_text}"
            )
        user_content = user
        if repair_requested:
            user_content = (
                f"{user}\n\n上一次输出未通过结构校验。请修复输出，只返回符合 Schema 的 JSON。"
            )
        response_format: dict[str, Any]
        if use_json_object:
            response_format = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "quality_output",
                    "strict": True,
                    "schema": schema,
                },
            }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "response_format": response_format,
        }

    def _is_schema_unavailable(self, response) -> bool:
        try:
            message = (response.json().get("error") or {}).get("message", "")
        except (AttributeError, TypeError, ValueError):
            return False
        lowered = str(message).lower()
        return any(hint in lowered for hint in self._JSON_SCHEMA_UNAVAILABLE_HINTS)

    def _retry_delay(self, response) -> float:
        try:
            value = float((response.headers or {}).get("Retry-After", ""))
        except (TypeError, ValueError):
            return self.retry_delay_seconds
        return max(0.0, min(value, self.max_retry_after_seconds))

    @staticmethod
    def _failure(
        code: str,
        stage: ErrorStage,
        message: str,
        retryable: bool,
        attempts: int,
        cause: Exception | None = None,
    ) -> PipelineFailure:
        failure = PipelineFailure(
            AnalysisError(
                code=code,
                stage=stage,
                message=message,
                retryable=retryable,
                attempts=attempts,
            )
        )
        if cause is not None:
            failure.__cause__ = cause
        return failure


# One-release import compatibility. New construction paths use DeepSeekGateway.
OpenRouterGateway = DeepSeekGateway
