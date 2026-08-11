from __future__ import annotations

import json
from typing import Any

import requests


class OpenRouterError(RuntimeError):
    pass


class OpenRouterGateway:
    """LLM 网关。

    历史：项目最早经 OpenRouter 中转调用 DeepSeek。现改为 DeepSeek 官方直连
    (api.deepseek.com)，不再依赖 OpenRouter 凭证。类名保留以避免大面积重命名。

    结构化输出兼容：OpenRouter 支持 response_format=json_schema；DeepSeek 官方
    某些档位（如 deepseek-chat 当前指向的模型）不支持 json_schema，会返回
    "This response_format type is unavailable now"。因此 complete_json 先尝试
    json_schema，遇到该错误时自动降级为 json_object（纯 JSON 模式），并把 JSON
    Schema 以文本形式写入 system 指令，要求模型只输出符合该 Schema 的 JSON。
    两者返回内容都是 JSON 字符串，解析方式一致。
    """

    # DeepSeek 在不支持 json_schema 时返回的错误特征。
    _JSON_SCHEMA_UNAVAILABLE_HINTS = ("response_format type is unavailable",)

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1/chat/completions",
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        # DeepSeek 官方不使用 OpenRouter 的 HTTP-Referer / X-Title 头；
        # 仅当显式提供（例如仍走 OpenRouter 中转）时才附带。
        self.extra_headers = {}
        if "openrouter.ai" in base_url:
            self.extra_headers = {
                "HTTP-Referer": "http://localhost:8080",
                "X-Title": "DebtCollectionQC",
            }

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        return requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.extra_headers,
            },
            json=payload,
            timeout=self.timeout,
        )

    def complete_json( #这个函数的作用：调用 LLM API，生成符合指定 JSON Schema 的结构化输出。
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            # 第一次尝试：json_schema（OpenRouter 及部分 DeepSeek 档位支持）。
            response = self._post({
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "quality_output",
                        "strict": True,
                        "schema": schema,
                    },
                },
            })
            if (
                response.status_code == 400
                and self._is_json_schema_unavailable(response)
            ):
                # 降级到 json_object：把 Schema 写进 system 指令。
                response = self._post({
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"{system}\n\n只输出符合如下 JSON Schema 的 JSON，"
                                f"不要输出任何解释或 markdown：\n"
                                f"{json.dumps(schema, ensure_ascii=False)}"
                            ),
                        },
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                })
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except (
            requests.RequestException,
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            raise OpenRouterError(self._error_detail(exc, response)) from exc

    def _is_json_schema_unavailable(self, response: requests.Response) -> bool:
        try:
            message = (response.json().get("error") or {}).get("message", "")
        except ValueError:
            return False
        return any(hint in message for hint in self._JSON_SCHEMA_UNAVAILABLE_HINTS)

    @staticmethod
    def _error_detail(exc: Exception, response: requests.Response | None) -> str:
        if response is not None:
            try:
                return f"{exc} | body={response.text[:300]}"
            except Exception:  # pragma: no cover - defensive
                pass
        return str(exc)
