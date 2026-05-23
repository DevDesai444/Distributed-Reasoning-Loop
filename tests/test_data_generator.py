"""
Tests for the data generator module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import CoTGenerator, GenerationConfig, ReasoningPath
from data_generator.cot_generator import InferenceBackend
from data_generator.dataset_loader import GSM8KLoader, HumanEvalLoader, Problem


class TestGenerationConfig:
    """Tests for GenerationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.num_paths == 10
        assert config.temperature == 0.8
        assert config.max_new_tokens == 2048
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = GenerationConfig(
            model_name="test-model",
            num_paths=5,
            temperature=0.5,
        )
        assert config.model_name == "test-model"
        assert config.num_paths == 5
        assert config.temperature == 0.5


class TestPromptFormatting:
    """Tests for prompt construction behavior."""

    def test_format_prompt_prefers_chat_template(self, monkeypatch):
        """Inference prompt formatting should use the model chat template when available."""
        generator = CoTGenerator(
            GenerationConfig(
                model_name="test-model",
                backend=InferenceBackend.VLLM,
            )
        )

        class DummyTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                assert tokenize is False
                assert add_generation_prompt is True
                assert messages[0]["role"] == "system"
                assert messages[-1]["role"] == "user"
                return "formatted-with-chat-template"

        monkeypatch.setattr(generator, "_get_formatter_tokenizer", lambda: DummyTokenizer())
        prompt = generator._format_prompt("What is 2+2?")
        assert prompt == "formatted-with-chat-template"


class TestReasoningPath:
    """Tests for ReasoningPath."""
    
    def test_path_creation(self):
        """Test reasoning path creation."""
        path = ReasoningPath(
            problem_id="test_1",
            problem="What is 2+2?",
            reasoning="2+2 = 4",
            final_answer="4",
            is_correct=True,
        )
        assert path.problem_id == "test_1"
        assert path.is_correct == True
    
    def test_path_hash(self):
        """Test path hash generation."""
        path = ReasoningPath(
            problem_id="test_1",
            problem="What is 2+2?",
            reasoning="2+2 = 4",
        )
        assert len(path.path_hash) == 16
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        path = ReasoningPath(
            problem_id="test_1",
            problem="What is 2+2?",
            reasoning="2+2 = 4",
        )
        d = path.to_dict()
        assert "problem_id" in d
        assert "reasoning" in d
        assert "path_hash" in d


class TestProblem:
    """Tests for Problem dataclass."""
    
    def test_problem_creation(self):
        """Test problem creation."""
        problem = Problem(
            id="test_1",
            problem="What is 2+2?",
            answer="4",
        )
        assert problem.id == "test_1"
        assert problem.answer == "4"
    
    def test_problem_with_metadata(self):
        """Test problem with metadata."""
        problem = Problem(
            id="test_1",
            problem="What is 2+2?",
            answer="4",
            metadata={"difficulty": "easy"},
        )
        assert problem.metadata["difficulty"] == "easy"


# Note: These tests require network access to download datasets
# They are marked as integration tests

@pytest.mark.integration
class TestGSM8KLoader:
    """Integration tests for GSM8KLoader."""
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = GSM8KLoader(subset_size=5)
        assert loader.subset_size == 5
    
    @pytest.mark.slow
    def test_load_subset(self):
        """Test loading a subset of GSM8K."""
        loader = GSM8KLoader(subset_size=5)
        try:
            problems = loader.load()
        except Exception as exc:
            pytest.skip(f"GSM8K download unavailable in this environment: {exc}")
        assert len(problems) == 5
        assert all(isinstance(p, Problem) for p in problems)


@pytest.mark.integration
class TestHumanEvalLoader:
    """Integration tests for HumanEvalLoader."""
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = HumanEvalLoader(subset_size=5)
        assert loader.subset_size == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
