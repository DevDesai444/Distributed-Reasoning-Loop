"""
Parallel code execution verifier backed by sandboxed Docker subprocesses.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional

from wandb_utils import make_histogram, log_to_wandb

logger = logging.getLogger(__name__)


class ExecutionVerificationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    COMPILE_ERROR = "compile_error"


@dataclass
class ExecutionVerificationResult:
    status: ExecutionVerificationStatus
    reward: float
    stdout: str = ""
    stderr: str = ""
    latency_s: float = 0.0
    error_type: Optional[str] = None


class ExecutionVerifier:
    """
    Verifies generated code by piping the program and test harness to a
    sandboxed Python process inside Docker.
    """

    def __init__(
        self,
        timeout: int = 10,
        docker_image: str = "distributed-reasoning-loop-sandbox:latest",
        memory_limit: str = "512m",
        max_workers: int = 8,
        docker_binary: str = "docker",
    ):
        self.timeout = timeout
        self.docker_image = docker_image
        self.memory_limit = memory_limit
        self.max_workers = max_workers
        self.docker_binary = docker_binary

    def _normalize_test_cases(self, test_cases: Any) -> str:
        """Convert supported test-case payloads into Python test code."""
        if test_cases is None:
            return ""

        if isinstance(test_cases, str):
            return test_cases.strip()

        if isinstance(test_cases, dict):
            if "code" in test_cases:
                return str(test_cases["code"]).strip()
            if "test" in test_cases:
                return str(test_cases["test"]).strip()

        if isinstance(test_cases, Iterable):
            blocks: list[str] = []
            for case in test_cases:
                if isinstance(case, str):
                    blocks.append(case.strip())
                    continue
                if isinstance(case, dict):
                    if "code" in case:
                        blocks.append(str(case["code"]).strip())
                        continue
                    if "test" in case:
                        blocks.append(str(case["test"]).strip())
                        continue
                    expression = case.get("expression")
                    expected = case.get("expected")
                    if expression is not None and expected is not None:
                        blocks.append(f"assert ({expression}) == {repr(expected)}")
                        continue
                    if "input" in case and "expected_output" in case:
                        blocks.append(
                            "raise AssertionError("
                            f"'IO-style test cases are unsupported without a harness: {case}'"
                            ")"
                        )
                        continue
                blocks.append(str(case).strip())
            return "\n".join(block for block in blocks if block)

        return str(test_cases).strip()

    def _build_program(self, generated_code: str, test_cases: Any) -> str:
        test_block = self._normalize_test_cases(test_cases)
        parts = [generated_code.rstrip()]
        if test_block:
            parts.append("")
            parts.append("# --- injected tests ---")
            parts.append(test_block.rstrip())
        return "\n".join(parts) + "\n"

    def _build_command(self) -> list[str]:
        return [
            self.docker_binary,
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--memory",
            self.memory_limit,
            "--cpus",
            "1",
            "--pids-limit",
            "64",
            "--read-only",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            self.docker_image,
            "python3",
            "-I",
            "-u",
            "-",
        ]

    def _classify_failure(self, stderr: str) -> tuple[ExecutionVerificationStatus, str, float]:
        stderr_lower = stderr.lower()
        compile_markers = ("syntaxerror", "indentationerror", "taberror")
        if any(marker in stderr_lower for marker in compile_markers):
            return ExecutionVerificationStatus.COMPILE_ERROR, "compile_error", -0.1
        return ExecutionVerificationStatus.FAIL, "fail", 0.0

    def verify(
        self,
        prompt: str,
        generated_code: str,
        test_cases: Any,
    ) -> ExecutionVerificationResult:
        """
        Execute generated code plus tests in a Docker sandbox and map the result
        onto a reward-bearing verification outcome.
        """
        del prompt  # Prompt is kept in the signature for parity with trainer call sites.

        program = self._build_program(generated_code, test_cases)
        start_time = time.time()

        try:
            result = subprocess.run(
                self._build_command(),
                input=program,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            latency = time.time() - start_time
            return ExecutionVerificationResult(
                status=ExecutionVerificationStatus.TIMEOUT,
                reward=-0.05,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                latency_s=latency,
                error_type="timeout",
            )
        except FileNotFoundError as exc:
            latency = time.time() - start_time
            logger.error("Docker executable not found: %s", exc)
            return ExecutionVerificationResult(
                status=ExecutionVerificationStatus.COMPILE_ERROR,
                reward=-0.1,
                stdout="",
                stderr=str(exc),
                latency_s=latency,
                error_type="docker_missing",
            )
        except Exception as exc:  # pragma: no cover - depends on host Docker runtime
            latency = time.time() - start_time
            logger.error("Sandbox execution failed: %s", exc)
            return ExecutionVerificationResult(
                status=ExecutionVerificationStatus.COMPILE_ERROR,
                reward=-0.1,
                stdout="",
                stderr=str(exc),
                latency_s=latency,
                error_type="sandbox_error",
            )

        latency = time.time() - start_time
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode == 0:
            return ExecutionVerificationResult(
                status=ExecutionVerificationStatus.PASS,
                reward=1.0,
                stdout=stdout,
                stderr=stderr,
                latency_s=latency,
                error_type="pass",
            )

        status, error_type, reward = self._classify_failure(stderr)
        return ExecutionVerificationResult(
            status=status,
            reward=reward,
            stdout=stdout,
            stderr=stderr,
            latency_s=latency,
            error_type=error_type,
        )

    def batch_verify(self, problems: list[dict[str, Any]]) -> list[ExecutionVerificationResult]:
        """
        Verify many problems in parallel. Every verification call starts its own
        Docker container, so thread workers naturally get isolated container instances.
        """
        results: list[Optional[ExecutionVerificationResult]] = [None] * len(problems)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self.verify,
                    problem.get("prompt", ""),
                    problem.get("generated_code", problem.get("code", "")),
                    problem.get("test_cases"),
                ): idx
                for idx, problem in enumerate(problems)
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Batch verification worker failed: %s", exc)
                    results[idx] = ExecutionVerificationResult(
                        status=ExecutionVerificationStatus.COMPILE_ERROR,
                        reward=-0.1,
                        stderr=str(exc),
                        error_type="worker_error",
                    )

        finalized = [result for result in results if result is not None]
        if finalized:
            latencies = [result.latency_s for result in finalized]
            status_counts = Counter(result.status.value for result in finalized)
            pass_rate = sum(result.status == ExecutionVerificationStatus.PASS for result in finalized) / len(finalized)
            log_payload = {
                "verification/code_pass_rate": pass_rate,
                "verification/code_latency_histogram": make_histogram(latencies),
            }
            for status, count in status_counts.items():
                log_payload[f"verification/error_type/{status}"] = count
            log_to_wandb(log_payload)

            for index, result in enumerate(finalized):
                log_to_wandb(
                    {
                        "verification/problem_index": index,
                        "verification/problem_latency_s": result.latency_s,
                        "verification/problem_reward": result.reward,
                    }
                )

        return finalized

    def image_available(self) -> bool:
        """Best-effort check that the sandbox image is present locally."""
        try:
            result = subprocess.run(
                [self.docker_binary, "image", "inspect", self.docker_image],
                text=True,
                capture_output=True,
                check=False,
                timeout=min(self.timeout, 10),
            )
            return result.returncode == 0
        except Exception:  # pragma: no cover - defensive
            return False


def get_default_sandbox_image() -> str:
    """Return the configured Docker image for code execution."""
    return os.getenv("REASONING_LOOP_SANDBOX_IMAGE", "distributed-reasoning-loop-sandbox:latest")
