from __future__ import annotations

import hashlib
import re
from typing import Any

SENSITIVE_KEYS = {"authorization", "api_key", "apikey", "token", "password", "secret", "prompt", "transcript", "content", "observation", "customerinfo"}
PHONE = re.compile(r"(?<!\d)1\d{10}(?!\d)")


def _hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def redact_for_observability(value: Any, *, _key: str | None = None) -> Any:
    key = (_key or "").lower().replace("_", "")
    if key in SENSITIVE_KEYS or any(part in key for part in ("authorization", "apikey", "password", "secret")):
        return {"redacted": True, "length": len(str(value)), "contentHash": _hash(value)}
    if isinstance(value, dict):
        result = {}
        for name, item in value.items():
            if str(name).lower().replace("_", "") in SENSITIVE_KEYS:
                result[name] = {"redacted": True, "length": len(str(item)), "contentHash": _hash(item)}
            else:
                result[name] = redact_for_observability(item, _key=str(name))
        return result
    if isinstance(value, list):
        return [redact_for_observability(item) for item in value]
    if isinstance(value, str):
        return PHONE.sub(lambda match: f"<phone:{_hash(match.group(0))}>", value)
    return value

