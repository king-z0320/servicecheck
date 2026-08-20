from types import SimpleNamespace

from qc.windows_runtime import disable_python_wmi_queries_when_requested


def test_disable_python_wmi_queries_when_requested(monkeypatch):
    monkeypatch.setenv("SERVICECHECK_DISABLE_PYTHON_WMI", "1")
    marker = object()
    platform_module = SimpleNamespace(_wmi=marker, _wmi_query=lambda: "blocked")

    disabled = disable_python_wmi_queries_when_requested(
        platform_module=platform_module,
        os_name="nt",
    )

    assert disabled is True
    assert platform_module._wmi is marker
    try:
        platform_module._wmi_query("OS", "Version")
    except OSError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("WMI query compatibility hook did not raise OSError")


def test_keep_python_wmi_without_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("SERVICECHECK_DISABLE_PYTHON_WMI", raising=False)
    original_query = lambda: "original"
    platform_module = SimpleNamespace(_wmi_query=original_query)

    disabled = disable_python_wmi_queries_when_requested(
        platform_module=platform_module,
        os_name="nt",
    )

    assert disabled is False
    assert platform_module._wmi_query is original_query
