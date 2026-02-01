"""
Verifier module for validating generated reasoning paths.
Supports both mathematical and code verification.
"""

from .math_verifier import (
    MathVerifier,
    GSM8KVerifier,
    VerificationResult,
    VerificationStatus,
)

from .code_verifier import (
    CodeVerifier,
    HumanEvalVerifier,
    DockerSandbox,
    ExecutionResult,
    ExecutionStatus,
    TestCase,
)

from .execution_verifier import (
    ExecutionVerifier,
    ExecutionVerificationResult,
    ExecutionVerificationStatus,
    get_default_sandbox_image,
)

from .step_extractor import extract_steps


def create_verifier(verifier_type: str = "math", **kwargs):
    """Factory for the main verifier types used across training and evaluation."""
    if verifier_type == "code":
        return ExecutionVerifier(**kwargs)
    return MathVerifier(**kwargs)

__all__ = [
    # Math verification
    "MathVerifier",
    "GSM8KVerifier",
    "VerificationResult",
    "VerificationStatus",
    # Code verification
    "CodeVerifier",
    "HumanEvalVerifier",
    "DockerSandbox",
    "ExecutionResult",
    "ExecutionStatus",
    "TestCase",
    "ExecutionVerifier",
    "ExecutionVerificationResult",
    "ExecutionVerificationStatus",
    "get_default_sandbox_image",
    "extract_steps",
    "create_verifier",
]
