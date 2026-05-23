"""
Tests for shared prompt formatting utilities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompting import build_messages, format_conversation, format_prompt


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        roles = [message["role"] for message in messages]
        suffix = "GEN" if add_generation_prompt else "FULL"
        return f"{suffix}:{'|'.join(roles)}"


def test_format_prompt_includes_system_message():
    tokenizer = DummyTokenizer()
    prompt = format_prompt(tokenizer, "What is 2+2?", problem_type="math")
    assert prompt == "GEN:system|user"


def test_format_conversation_preserves_system_user_assistant_structure():
    tokenizer = DummyTokenizer()
    transcript = format_conversation(
        tokenizer,
        "What is 2+2?",
        "We compute carefully. #### 4",
        problem_type="math",
    )
    assert transcript == "FULL:system|user|assistant"


def test_build_messages_supports_few_shot_examples():
    messages = build_messages(
        "Solve 3+3",
        few_shot_examples=[
            {"problem": "Solve 1+1", "solution": "2"},
        ],
    )
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
