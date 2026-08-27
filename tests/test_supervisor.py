from __future__ import annotations

import pytest

from tandemn_worker import supervisor


@pytest.mark.parametrize(
    ("configured_executable", "expected_executable"),
    [
        (None, "vllm"),
        ("/opt/vllm/bin/vllm", "/opt/vllm/bin/vllm"),
    ],
)
def test_build_vllm_command_uses_configured_executable(
    monkeypatch: pytest.MonkeyPatch,
    configured_executable: str | None,
    expected_executable: str,
) -> None:
    if configured_executable is None:
        monkeypatch.delenv("TD_VLLM_EXECUTABLE", raising=False)
    else:
        monkeypatch.setenv("TD_VLLM_EXECUTABLE", configured_executable)
    monkeypatch.setenv("TD_VLLM_MODEL", "test-model")
    monkeypatch.setenv("TD_VLLM_HOST", "127.0.0.1")
    monkeypatch.setenv("TD_VLLM_PORT", "8080")
    monkeypatch.setenv("TD_VLLM_EXTRA_ARGS", "--tensor-parallel-size 2")

    assert supervisor.build_vllm_command() == [
        expected_executable,
        "serve",
        "test-model",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--tensor-parallel-size",
        "2",
    ]
