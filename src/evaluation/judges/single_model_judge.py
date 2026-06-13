"""
Single-model judge. Runs one open-weights LM with a rubric prompt and parses
the verdict.

Backend resolution: we try vLLM first (matches the rest of the project) and
fall back to a transformers pipeline if vLLM isn't importable. For CPU smoke
runs the transformers path is what actually runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .base import Judge, JudgeVerdict, Verdict, parse_judge_response
from .prompts import get_rubric

logger = logging.getLogger(__name__)


DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class JudgeConfig:
    model_name: str = DEFAULT_JUDGE_MODEL
    backend: str = "auto"  # auto | vllm | transformers | stub
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 7
    # Used for the rare deterministic-stub mode in tests.
    stub_response: Optional[str] = None
    # Pull at most this many tokens of the response into the prompt; long
    # generations otherwise blow the context window.
    max_response_chars: int = 4000
    extra: Dict[str, Any] = field(default_factory=dict)


class SingleModelJudge(Judge):
    """One model, one rubric, one verdict per trace."""

    def __init__(self, config: Optional[JudgeConfig] = None, *, judge_id: Optional[str] = None):
        self.config = config or JudgeConfig()
        self.judge_id = judge_id or f"single::{self.config.model_name}"
        self._backend: Optional[str] = None
        self._llm: Any = None
        self._tokenizer: Any = None
        self._sampler: Any = None
        self._stub_fn: Optional[Callable[[str], str]] = None

    # ---- backend wiring -------------------------------------------------

    def _resolve_backend(self) -> str:
        if self._backend is not None:
            return self._backend
        choice = self.config.backend
        if choice == "stub":
            self._backend = "stub"
            return self._backend
        if choice in ("vllm", "auto"):
            try:
                import vllm  # noqa: F401
                self._backend = "vllm"
                return self._backend
            except ImportError:
                if choice == "vllm":
                    raise
        # fall through to transformers
        self._backend = "transformers"
        return self._backend

    def _setup(self):
        backend = self._resolve_backend()
        if backend == "stub":
            if self._stub_fn is None:
                fixed = self.config.stub_response or ""

                def _const(_prompt: str, _fixed: str = fixed) -> str:
                    return _fixed

                self._stub_fn = _const
            return
        if backend == "vllm" and self._llm is None:
            from vllm import LLM, SamplingParams

            self._llm = LLM(
                model=self.config.model_name,
                trust_remote_code=True,
                enforce_eager=True,
            )
            self._sampler = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_new_tokens,
                seed=self.config.seed,
            )
            return
        if backend == "transformers" and self._llm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            self._llm.eval()

    def set_stub(self, fn: Callable[[str], str]):
        """Inject a deterministic response generator for tests."""
        self.config.backend = "stub"
        self._backend = "stub"
        self._stub_fn = fn

    # ---- generation -----------------------------------------------------

    def _generate(self, prompt: str) -> str:
        self._setup()
        backend = self._backend
        if backend == "stub":
            assert self._stub_fn is not None
            return self._stub_fn(prompt)
        if backend == "vllm":
            outputs = self._llm.generate([prompt], self._sampler)
            return outputs[0].outputs[0].text
        if backend == "transformers":
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            with torch.no_grad():
                out = self._llm.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.temperature > 0,
                    temperature=max(self.config.temperature, 1e-5),
                    top_p=self.config.top_p,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            text = self._tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return text
        raise RuntimeError(f"Unknown judge backend {backend!r}")

    # ---- public API -----------------------------------------------------

    def _build_prompt(
        self,
        *,
        problem: str,
        response: str,
        reference_answer: Optional[str],
        problem_type: str,
    ) -> tuple[str, str]:
        version_tag, template = get_rubric(problem_type)
        clipped_response = response or ""
        if len(clipped_response) > self.config.max_response_chars:
            clipped_response = clipped_response[: self.config.max_response_chars] + "\n[...truncated]"
        filled = template.format(
            problem=problem,
            response=clipped_response,
            reference_answer=reference_answer if reference_answer is not None else "(not provided)",
        )
        return version_tag, filled

    def judge(
        self,
        *,
        problem: str,
        response: str,
        reference_answer: Optional[str] = None,
        problem_type: str = "math",
    ) -> JudgeVerdict:
        version_tag, prompt = self._build_prompt(
            problem=problem,
            response=response,
            reference_answer=reference_answer,
            problem_type=problem_type,
        )
        try:
            raw = self._generate(prompt)
        except Exception as exc:  # pragma: no cover - guard for runtime failures
            logger.exception("Judge generation failed: %s", exc)
            return JudgeVerdict(
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                rationale=f"generation_error: {exc}",
                reason_code="OTHER",
                raw_response="",
                judge_id=self.judge_id,
                prompt_version=version_tag,
            )

        parsed = parse_judge_response(raw)
        return JudgeVerdict(
            verdict=parsed["verdict"],
            confidence=parsed["confidence"],
            rationale=parsed["rationale"],
            reason_code=parsed["reason_code"],
            raw_response=raw[:4000],
            judge_id=self.judge_id,
            prompt_version=version_tag,
        )
