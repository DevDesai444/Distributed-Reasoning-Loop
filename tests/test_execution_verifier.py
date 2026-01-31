"""
Tests for the subprocess-backed execution verifier.
"""

from types import SimpleNamespace
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verifier.execution_verifier import (
    ExecutionVerificationStatus,
    ExecutionVerifier,
)


class TestExecutionVerifier:
    def test_verify_pass(self, monkeypatch):
        verifier = ExecutionVerifier(docker_image="sandbox:test")

        def fake_run(*args, **kwargs):
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = verifier.verify("prompt", "print('ok')", "assert True")

        assert result.status == ExecutionVerificationStatus.PASS
        assert result.reward == 1.0
        assert result.stdout == "ok\n"

    def test_verify_compile_error(self, monkeypatch):
        verifier = ExecutionVerifier(docker_image="sandbox:test")

        def fake_run(*args, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="SyntaxError: invalid syntax")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = verifier.verify("prompt", "def broken(", "assert True")

        assert result.status == ExecutionVerificationStatus.COMPILE_ERROR
        assert result.reward == -0.1

    def test_verify_timeout(self, monkeypatch):
        verifier = ExecutionVerifier(docker_image="sandbox:test", timeout=1)

        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="docker", timeout=1, output="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = verifier.verify("prompt", "while True: pass", "assert True")

        assert result.status == ExecutionVerificationStatus.TIMEOUT
        assert result.reward == -0.05
