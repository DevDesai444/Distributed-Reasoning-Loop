"""
Tests for evaluation integrity and TTC selection behavior.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.benchmarks import GSM8KEvaluator
from evaluation.test_time_compute import GeneratedPath, TestTimeCompute, TestTimeComputeConfig
from verifier import GSM8KVerifier


def test_ttc_weighted_vote_prefers_consensus_answer():
    ttc = TestTimeCompute(
        "dummy-model",
        TestTimeComputeConfig(num_samples=3, aggregation_method="weighted_vote"),
        verifier_type="math",
    )
    ttc.verifier = GSM8KVerifier()
    ttc.generate_paths = lambda problem: [
        GeneratedPath(reasoning="Reasoning A\n#### 12"),
        GeneratedPath(reasoning="Reasoning B\n#### 12"),
        GeneratedPath(reasoning="Reasoning C\n#### 9"),
    ]

    best, paths = ttc.solve("What is 6+6?")

    assert best.final_answer == "12"
    assert len(paths) == 3


def test_gsm8k_evaluator_ttc_does_not_use_ground_truth_without_oracle(monkeypatch):
    captured = {}

    class FakeLoader:
        def __init__(self, split="test", subset_size=None):
            self.split = split
            self.subset_size = subset_size

        def load(self):
            return [
                types.SimpleNamespace(
                    id="gsm8k_test_0",
                    problem="What is 2+2?",
                    answer="4",
                    metadata={},
                )
            ]

    class FakeTTC:
        def solve(self, prompt, expected_answer=None):
            captured["expected_answer"] = expected_answer
            return GeneratedPath(reasoning="Work\n#### 4", final_answer="4"), []

    monkeypatch.setitem(sys.modules, "data_generator", types.SimpleNamespace(GSM8KLoader=FakeLoader))

    evaluator = GSM8KEvaluator(
        "dummy-model",
        use_test_time_compute=True,
        ttc_samples=4,
        ttc_oracle_verify=False,
    )
    evaluator.setup = lambda: (
        setattr(evaluator, "ttc", FakeTTC()),
        setattr(evaluator, "verifier", GSM8KVerifier()),
    )

    result = evaluator.evaluate(subset_size=1)

    assert captured["expected_answer"] is None
    assert result.correct == 1


def test_gsm8k_evaluator_ttc_can_use_ground_truth_in_explicit_oracle_mode(monkeypatch):
    captured = {}

    class FakeLoader:
        def __init__(self, split="test", subset_size=None):
            self.split = split
            self.subset_size = subset_size

        def load(self):
            return [
                types.SimpleNamespace(
                    id="gsm8k_test_0",
                    problem="What is 2+2?",
                    answer="4",
                    metadata={},
                )
            ]

    class FakeTTC:
        def solve(self, prompt, expected_answer=None):
            captured["expected_answer"] = expected_answer
            return GeneratedPath(reasoning="Work\n#### 4", final_answer="4"), []

    monkeypatch.setitem(sys.modules, "data_generator", types.SimpleNamespace(GSM8KLoader=FakeLoader))

    evaluator = GSM8KEvaluator(
        "dummy-model",
        use_test_time_compute=True,
        ttc_samples=4,
        ttc_oracle_verify=True,
    )
    evaluator.setup = lambda: (
        setattr(evaluator, "ttc", FakeTTC()),
        setattr(evaluator, "verifier", GSM8KVerifier()),
    )

    result = evaluator.evaluate(subset_size=1)

    assert captured["expected_answer"] == "4"
    assert result.correct == 1
