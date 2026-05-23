"""
Shared prompt construction utilities.

Keeping training and inference on the same chat scaffold reduces prompt-shape
drift between synthetic data generation, post-training, and evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional


MATH_SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "double-check the arithmetic, and end with the final numeric answer after '#### '."
)

CODE_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Explain the approach briefly, "
    "then provide the complete solution in a Python code block."
)


def get_system_prompt(problem_type: str = "math") -> str:
    """Return the default system prompt for a problem type."""
    return MATH_SYSTEM_PROMPT if problem_type == "math" else CODE_SYSTEM_PROMPT


def build_messages(
    problem: str,
    problem_type: str = "math",
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build a canonical chat transcript for a reasoning task."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": get_system_prompt(problem_type)}
    ]

    if few_shot_examples:
        for example in few_shot_examples:
            messages.append({"role": "user", "content": example["problem"]})
            messages.append({"role": "assistant", "content": example["solution"]})

    messages.append({"role": "user", "content": problem})
    return messages


def _manual_chat_fallback(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    formatted = []
    for msg in messages:
        formatted.append(f"<|{msg['role']}|>\n{msg['content']}")
    if add_generation_prompt:
        formatted.append("<|assistant|>\n")
    return "\n".join(formatted)


def apply_chat_format(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    """
    Render messages with the tokenizer's chat template when available.

    Falls back to a simple role-tagged format when the tokenizer does not
    expose a working chat template.
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return _manual_chat_fallback(messages, add_generation_prompt)


def format_prompt(
    tokenizer,
    problem: str,
    *,
    problem_type: str = "math",
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Format a prompt ready for generation."""
    return apply_chat_format(
        tokenizer,
        build_messages(problem, problem_type, few_shot_examples),
        add_generation_prompt=True,
    )


def format_conversation(
    tokenizer,
    prompt: str,
    response: str,
    *,
    problem_type: str = "math",
) -> str:
    """Format a full prompt/response conversation for SFT-style training."""
    messages = build_messages(prompt, problem_type)
    messages.append({"role": "assistant", "content": response})
    return apply_chat_format(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
