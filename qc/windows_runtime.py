from __future__ import annotations

import os
from types import ModuleType
from typing import Any


def disable_python_wmi_queries_when_requested(
    *,
    platform_module: ModuleType | Any | None = None,
    os_name: str | None = None,
) -> bool:
    """Make stdlib platform helpers use their supported non-WMI fallback.

    Python 3.14 queries WMI from ``platform.machine()``. If the local WMI
    provider is unhealthy, imports such as SQLAlchemy can otherwise block
    indefinitely while asking only for system metadata. Both Python versions
    used by this project already treat ``OSError`` from ``_wmi_query`` as the
    signal to use their registry and system-API fallback paths.
    """
    enabled = os.getenv("SERVICECHECK_DISABLE_PYTHON_WMI", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return False
    if (os_name or os.name) != "nt":
        return False

    if platform_module is None:
        import platform as platform_module

    if not hasattr(platform_module, "_wmi_query"):
        return False

    def wmi_disabled(*_args, **_kwargs):
        raise OSError("Python WMI platform queries disabled by serviceCheck")

    platform_module._wmi_query = wmi_disabled
    return True
