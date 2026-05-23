"""
Tests for evaluation integrity and TTC selection behavior.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.benchmarks import BenchmarkResult, GSM8KEvaluator
from evaluation.test_time_compute import GeneratedPath, TestTimeCompute, TestTimeComputeConfig
from scripts.evaluate import finalize_result
from verifier import GSM8KVerifier


def test_benchmark_result_marks_failed_runs_invalid(tmp_path):
    result = BenchmarkResult(
        benchmark_name="GSM8K",
        total_problems=100,
        correct=0,
        incorrect=0,
        errors=100,
        accuracy=0.0,
        avg_time_per_problem=0.0,
        metadata={"model": "dummy"},
    )

    exit_code = finalize_result(
        result,
        str(tmp_path / "gsm8k_results.json"),
        str(tmp_path),
        fail_on_errors=False,
    )

    assert exit_code == 2
    assert result.status == "failed"
    assert result.valid_run is False
    saved = (tmp_path / "gsm8k_results.json").read_text()
    assert '"status": "failed"' in saved
    assert '"valid_run": false' in saved


def test_benchmark_result_marks_partial_runs_when_requested(tmp_path):
    result = BenchmarkResult(
        benchmark_name="GSM8K",
        total_problems=10,
        correct=7,
        incorrect=1,
        errors=2,
        accuracy=0.7,
        avg_time_per_problem=1.0,
        metadata={"model": "dummy"},
    )

    exit_code = finalize_result(
        result,
        str(tmp_path / "gsm8k_results.json"),
        str(tmp_path),
        fail_on_errors=True,
    )

    assert exit_code == 2
    assert result.status == "partial"
    assert result.valid_run is False


def test_base_evaluator_setup_skips_standard_generator_in_ttc_mode(monkeypatch):
    calls = {"generator_init": 0, "ttc_setup": 0}

    class FakeCoTGenerator:
        def __init__(self, config):
            calls["generator_init"] += 1

        def initialize(self):
            calls["generator_init"] += 100

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTTC:
        def __init__(self, model_name, config):
            self.model_name = model_name
            self.config = config

        def setup(self):
            calls["ttc_setup"] += 1

    monkeypatch.setitem(
        sys.modules,
        "data_generator",
        types.SimpleNamespace(
            CoTGenerator=FakeCoTGenerator,
            GenerationConfig=FakeGenerationConfig,
        ),
    )

    import evaluation.benchmarks as benchmarks

    monkeypatch.setattr(benchmarks, "TestTimeCompute", FakeTTC, raising=False)
    monkeypatch.setattr(
        benchmarks,
        "TestTimeComputeConfig",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "evaluation.test_time_compute",
        types.SimpleNamespace(
            TestTimeCompute=FakeTTC,
            TestTimeComputeConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        ),
    )

    evaluator = GSM8KEvaluator("dummy-model", use_test_time_compute=True, ttc_samples=3)
    evaluator.setup()

    assert calls["generator_init"] == 0
    assert calls["ttc_setup"] == 1


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
