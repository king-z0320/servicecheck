"""Correlation context helpers used by trace/log adapters."""

from __future__ import annotations

from contextvars import ContextVar

_values: ContextVar[dict[str, str]] = ContextVar("servicecheck_context", default={})


def set_context(**values: str) -> None:
    current = _values.get().copy()
    current.update({key: str(value) for key, value in values.items() if value is not None})
    _values.set(current)


def get_context() -> dict[str, str]:
    return dict(_values.get())


def clear_context() -> None:
    _values.set({})

