"""
End-to-end pipeline integration test.

Runs generation -> train (1 epoch) -> evaluation through scripts/run_pipeline.py
using lightweight fakes for model-heavy dependencies.
"""

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _load_run_pipeline_module():
    """Load scripts/run_pipeline.py as an importable module."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline_integration_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pipeline_generate_train_eval_end_to_end(monkeypatch, tmp_path):
    """Integration: generate 10 samples -> train 1 epoch -> evaluate."""
    state = {
        "generated_samples": 0,
        "generated_pairs": 0,
        "sft_trained": False,
        "dpo_trained": False,
        "dpo_train_rows": 0,
        "eval_called": False,
    }

    # Stub wandb integration.
    monkeypatch.setitem(
        sys.modules,
        "wandb_utils",
        types.SimpleNamespace(ensure_wandb_run=lambda *args, **kwargs: None, get_wandb=lambda: None),
    )

    # Fake data generator module used by run_data_generation.
    fake_data_generator = types.ModuleType("data_generator")

    @dataclass
    class FakeGenerationConfig:
        model_name: str
        backend: str
        num_paths: int
        max_new_tokens: int
        temperature: float
        top_p: float = 0.95
        tensor_parallel_size: int = 1
        gpu_memory_utilization: float = 0.9

    class FakePreprocessConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSample:
        def __init__(self, idx: int):
            self.idx = idx

        def to_dict(self):
            return {
                "problem_id": f"p{self.idx}",
                "problem": f"What is {self.idx}+{self.idx}?",
                "reasoning": f"Compute {self.idx}+{self.idx} = {self.idx * 2}. #### {self.idx * 2}",
                "path_hash": f"h{self.idx}",
                "final_answer": str(self.idx * 2),
                "expected_answer": str(self.idx * 2),
                "is_correct": True,
            }

    class FakePair:
        def __init__(self, idx: int):
            self.idx = idx

        def to_dict(self):
            return {
                "problem_id": f"p{self.idx}",
                "problem": f"What is {self.idx}+{self.idx}?",
                "chosen": f"Correct reasoning for {self.idx}",
                "rejected": f"Wrong reasoning for {self.idx}",
            }

    class FakeSyntheticDataPipeline:
        def __init__(self, generator_config, dataset_name, output_dir, preprocess_config=None):
            self.generator_config = generator_config
            self.dataset_name = dataset_name
            self.output_dir = Path(output_dir)
            self.preprocess_config = preprocess_config
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def run(self, subset_size, batch_size):
            assert subset_size == 10
            assert batch_size > 0

            samples = [FakeSample(i) for i in range(10)]
            pairs = [FakePair(i) for i in range(10)]
            state["generated_samples"] = len(samples)
            state["generated_pairs"] = len(pairs)

            dpo_path = self.output_dir / "dpo_pairs.jsonl"
            with dpo_path.open("w") as handle:
                for pair in pairs:
                    row = pair.to_dict()
                    handle.write(
                        json.dumps(
                            {
                                "prompt": row["problem"],
                                "chosen": row["chosen"],
                                "rejected": row["rejected"],
                            }
                        )
                        + "\n"
                    )

            correct_path = self.output_dir / "correct_samples.jsonl"
            with correct_path.open("w") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample.to_dict()) + "\n")

            return samples, pairs

    fake_data_generator.GenerationConfig = FakeGenerationConfig
    fake_data_generator.PreprocessConfig = FakePreprocessConfig
    fake_data_generator.SyntheticDataPipeline = FakeSyntheticDataPipeline
    monkeypatch.setitem(sys.modules, "data_generator", fake_data_generator)

    # Fake enum container for run_data_generation import.
    fake_cot_generator = types.ModuleType("data_generator.cot_generator")
    fake_cot_generator.InferenceBackend = types.SimpleNamespace(VLLM="vllm")
    monkeypatch.setitem(sys.modules, "data_generator.cot_generator", fake_cot_generator)

    # Fake training module used by run_sft_training and run_dpo_training.
    fake_training = types.ModuleType("training")

    @dataclass
    class FakeSFTTrainerConfig:
        model_name: str
        learning_rate: float
        batch_size: int
        num_epochs: int
        output_dir: str

    class FakeSFTFromSyntheticData:
        def __init__(self, cfg, data_path):
            assert cfg.num_epochs == 1
            self.cfg = cfg
            self.data_path = Path(data_path)

        def train(self):
            assert self.data_path.exists()
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.output_dir) / "fake_sft_done.txt").write_text("ok")
            state["sft_trained"] = True

    @dataclass
    class FakeDPOTrainerConfig:
        model_name: str
        beta: float
        learning_rate: float
        batch_size: int
        num_epochs: int
        max_length: int
        output_dir: str

    class FakeReasoningDPOTrainer:
        def __init__(self, cfg):
            assert cfg.num_epochs == 1
            self.cfg = cfg

        def train(self, data):
            assert len(data) == 10
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.output_dir) / "fake_dpo_done.txt").write_text("ok")
            state["dpo_trained"] = True
            state["dpo_train_rows"] = len(data)

    fake_training.SFTTrainerConfig = FakeSFTTrainerConfig
    fake_training.SFTFromSyntheticData = FakeSFTFromSyntheticData
    fake_training.DPOTrainerConfig = FakeDPOTrainerConfig
    fake_training.ReasoningDPOTrainer = FakeReasoningDPOTrainer
    monkeypatch.setitem(sys.modules, "training", fake_training)

    # Fake evaluation module used by run_evaluation.
    fake_evaluation = types.ModuleType("evaluation")

    class FakeEvalResult:
        def __init__(self, accuracy):
            self.accuracy = accuracy
            self.errors = 0

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps({"summary": {"accuracy": self.accuracy}}))

        def to_dict(self):
            return {
                "benchmark": "GSM8K",
                "total": 10,
                "completed": 10,
                "correct": 7,
                "incorrect": 3,
                "errors": 0,
                "accuracy": self.accuracy,
                "avg_time": 1.0,
                "success_rate": 1.0,
                "status": "success",
                "valid_run": True,
            }

    class FakeGSM8KEvaluator:
        def __init__(self, model_name, use_test_time_compute, ttc_samples):
            self.model_name = model_name
            self.use_test_time_compute = use_test_time_compute
            self.ttc_samples = ttc_samples

        def evaluate(self, subset_size):
            state["eval_called"] = True
            state["eval_subset"] = subset_size
            return FakeEvalResult(accuracy=0.7)

    fake_evaluation.GSM8KEvaluator = FakeGSM8KEvaluator
    fake_evaluation.HumanEvalEvaluator = object
    fake_evaluation.MATHEvaluator = object
    monkeypatch.setitem(sys.modules, "evaluation", fake_evaluation)

    config = OmegaConf.create(
        {
            "data_generator": {
                "teacher_model": "teacher",
                "student_model": "student",
                "backend": "vllm",
                "tensor_parallel_size": "auto",
                "gpu_memory_utilization": 0.9,
                "num_cot_paths": 2,
                "max_new_tokens": 32,
                "temperature": 0.7,
                "top_p": 0.95,
            },
            "verifier": {
                "type": "math",
                "math": {"timeout": 1},
                "code": {"timeout": 1, "docker_image": "sandbox", "memory_limit": "256m"},
            },
            "orchestration": {
                "kafka": {
                    "enabled": False,
                    "bootstrap_servers": ["localhost:9092"],
                    "topics": {
                        "raw_reasoning_data": "raw_reasoning_data",
                        "verified_paths": "verified_paths",
                        "training_data": "training_data",
                    },
                }
            },
            "training": {
                "method": "dpo",
                "batch_size": 2,
                "learning_rate": 1e-6,
                "num_epochs": 1,
                "dpo": {"beta": 0.1, "max_length": 128},
                "grpo": {
                    "group_size": 4,
                    "kl_threshold": 0.1,
                    "eval_interval_steps": 10,
                    "heldout_eval_size": 5,
                    "eval_max_new_tokens": 64,
                    "online_max_new_tokens": 64,
                    "online_temperature": 0.8,
                    "online_top_p": 0.95,
                    "online_resample_attempts": 1,
                    "online_min_reward_std": 1e-6,
                    "enable_ray_verification": False,
                    "ray_verifier_workers": 1,
                },
                "evaluation": {"num_paths": 4},
                "wandb": {"enabled": False, "project": "test", "mode": "offline"},
            },
            "general": {"output_dir": str(tmp_path / "outputs"), "cache_dir": str(tmp_path / "cache")},
        }
    )
    config_path = tmp_path / "integration_config.yaml"
    OmegaConf.save(config, config_path)

    module = _load_run_pipeline_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--config",
            str(config_path),
            "--dataset",
            "gsm8k",
            "--subset-size",
            "10",
            "--batch-size",
            "5",
            "--run-sft",
            "--training-method",
            "dpo",
        ],
    )
    module.main()

    assert state["generated_samples"] == 10
    assert state["generated_pairs"] == 10
    assert state["sft_trained"] is True
    assert state["dpo_trained"] is True
    assert state["dpo_train_rows"] == 10
    assert state["eval_called"] is True

    latest_run = json.loads((tmp_path / "outputs" / "latest_run.json").read_text())
    run_dir = Path(latest_run["run_dir"])

    assert (run_dir / "dpo_model" / "fake_dpo_done.txt").exists()
    assert (run_dir / "gsm8k_results.json").exists()
    assert (run_dir / "pipeline_summary.json").exists()
    assert (run_dir / "run_manifest.json").exists()


def test_pipeline_prefers_best_checkpoint_for_final_evaluation(monkeypatch, tmp_path):
    state = {"eval_model_name": None}

    monkeypatch.setitem(
        sys.modules,
        "wandb_utils",
        types.SimpleNamespace(ensure_wandb_run=lambda *args, **kwargs: None, get_wandb=lambda: None),
    )

    fake_data_generator = types.ModuleType("data_generator")

    @dataclass
    class FakeGenerationConfig:
        model_name: str
        backend: str
        num_paths: int
        max_new_tokens: int
        temperature: float
        top_p: float = 0.95
        tensor_parallel_size: int = 1
        gpu_memory_utilization: float = 0.9

    class FakePreprocessConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSample:
        def __init__(self, idx: int):
            self.idx = idx

        def to_dict(self):
            return {
                "problem_id": f"p{self.idx}",
                "problem": f"What is {self.idx}+{self.idx}?",
                "reasoning": f"Compute {self.idx}+{self.idx} = {self.idx * 2}. #### {self.idx * 2}",
                "path_hash": f"h{self.idx}",
                "final_answer": str(self.idx * 2),
                "expected_answer": str(self.idx * 2),
                "is_correct": True,
            }

    class FakePair:
        def __init__(self, idx: int):
            self.idx = idx

        def to_dict(self):
            return {
                "problem_id": f"p{self.idx}",
                "problem": f"What is {self.idx}+{self.idx}?",
                "chosen": f"Correct reasoning for {self.idx}",
                "rejected": f"Wrong reasoning for {self.idx}",
            }

    class FakeSyntheticDataPipeline:
        def __init__(self, generator_config, dataset_name, output_dir, preprocess_config=None):
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def run(self, subset_size, batch_size):
            samples = [FakeSample(i) for i in range(4)]
            pairs = [FakePair(i) for i in range(4)]
            with (self.output_dir / "dpo_pairs.jsonl").open("w") as handle:
                for pair in pairs:
                    row = pair.to_dict()
                    handle.write(json.dumps({"prompt": row["problem"], "chosen": row["chosen"], "rejected": row["rejected"]}) + "\n")
            with (self.output_dir / "correct_samples.jsonl").open("w") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample.to_dict()) + "\n")
            return samples, pairs

    fake_data_generator.GenerationConfig = FakeGenerationConfig
    fake_data_generator.PreprocessConfig = FakePreprocessConfig
    fake_data_generator.SyntheticDataPipeline = FakeSyntheticDataPipeline
    monkeypatch.setitem(sys.modules, "data_generator", fake_data_generator)

    fake_cot_generator = types.ModuleType("data_generator.cot_generator")
    fake_cot_generator.InferenceBackend = types.SimpleNamespace(VLLM="vllm")
    monkeypatch.setitem(sys.modules, "data_generator.cot_generator", fake_cot_generator)

    fake_training = types.ModuleType("training")

    @dataclass
    class FakeDPOTrainerConfig:
        model_name: str
        beta: float
        learning_rate: float
        batch_size: int
        num_epochs: int
        max_length: int
        output_dir: str

    class FakeReasoningDPOTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def train(self, data):
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.output_dir) / "fake_dpo_done.txt").write_text("ok")

    @dataclass
    class FakeSFTTrainerConfig:
        model_name: str
        learning_rate: float
        batch_size: int
        num_epochs: int
        output_dir: str

    class FakeSFTFromSyntheticData:
        def __init__(self, cfg, data_path):
            self.cfg = cfg

        def train(self):
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.output_dir) / "fake_sft_done.txt").write_text("ok")

    fake_training.DPOTrainerConfig = FakeDPOTrainerConfig
    fake_training.ReasoningDPOTrainer = FakeReasoningDPOTrainer
    fake_training.SFTTrainerConfig = FakeSFTTrainerConfig
    fake_training.SFTFromSyntheticData = FakeSFTFromSyntheticData
    monkeypatch.setitem(sys.modules, "training", fake_training)

    fake_grpo_trainer = types.ModuleType("training.grpo_trainer")

    @dataclass
    class FakeGRPOConfig:
        model_name: str
        learning_rate: float
        batch_size: int
        num_epochs: int
        max_length: int
        group_size: int
        verifier_type: str
        verifier_timeout: int
        code_docker_image: str
        code_memory_limit: str
        kl_threshold: float
        eval_interval_steps: int
        heldout_eval_size: int
        heldout_dataset: str
        eval_max_new_tokens: int
        save_best_checkpoint: bool
        best_checkpoint_metric: str
        min_eval_improvement: float
        early_stop_patience: int
        online_max_new_tokens: int
        online_temperature: float
        online_top_p: float
        online_resample_attempts: int
        online_min_reward_std: float
        enable_ray_verification: bool
        ray_verifier_workers: int
        prompt_problem_type: str
        wandb_project: str
        wandb_mode: str
        output_dir: str

    class FakeReasoningGRPOTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def train(self, data):
            output_dir = Path(self.cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "fake_grpo_done.txt").write_text("ok")
            best_dir = output_dir / "best_checkpoint"
            best_dir.mkdir(parents=True, exist_ok=True)
            (best_dir / "best_marker.txt").write_text("best")
            (output_dir / "checkpoint_selection.json").write_text(
                json.dumps(
                    {
                        "best_eval_score": 0.75,
                        "best_eval_step": 10,
                        "best_checkpoint_metric": "pass_at_1",
                    }
                )
            )

    fake_grpo_trainer.GRPOConfig = FakeGRPOConfig
    fake_grpo_trainer.ReasoningGRPOTrainer = FakeReasoningGRPOTrainer
    monkeypatch.setitem(sys.modules, "training.grpo_trainer", fake_grpo_trainer)

    fake_evaluation = types.ModuleType("evaluation")

    class FakeEvalResult:
        def __init__(self, accuracy):
            self.accuracy = accuracy
            self.errors = 0

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps({"summary": {"accuracy": self.accuracy}}))

        def to_dict(self):
            return {
                "benchmark": "GSM8K",
                "total": 4,
                "completed": 4,
                "correct": 3,
                "incorrect": 1,
                "errors": 0,
                "accuracy": self.accuracy,
                "avg_time": 1.0,
                "success_rate": 1.0,
                "status": "success",
                "valid_run": True,
            }

    class FakeGSM8KEvaluator:
        def __init__(self, model_name, use_test_time_compute, ttc_samples):
            state["eval_model_name"] = model_name

        def evaluate(self, subset_size):
            return FakeEvalResult(accuracy=0.75)

    fake_evaluation.GSM8KEvaluator = FakeGSM8KEvaluator
    fake_evaluation.HumanEvalEvaluator = object
    fake_evaluation.MATHEvaluator = object
    monkeypatch.setitem(sys.modules, "evaluation", fake_evaluation)

    config = OmegaConf.create(
        {
            "data_generator": {
                "teacher_model": "teacher",
                "student_model": "student",
                "backend": "vllm",
                "tensor_parallel_size": "auto",
                "gpu_memory_utilization": 0.9,
                "num_cot_paths": 2,
                "max_new_tokens": 32,
                "temperature": 0.7,
                "top_p": 0.95,
                "preprocessing": {},
            },
            "verifier": {
                "type": "math",
                "math": {"timeout": 1},
                "code": {"timeout": 1, "docker_image": "sandbox", "memory_limit": "256m"},
            },
            "orchestration": {
                "kafka": {
                    "enabled": False,
                    "bootstrap_servers": ["localhost:9092"],
                    "topics": {
                        "raw_reasoning_data": "raw_reasoning_data",
                        "verified_paths": "verified_paths",
                        "training_data": "training_data",
                    },
                }
            },
            "training": {
                "method": "best",
                "batch_size": 2,
                "learning_rate": 1e-6,
                "num_epochs": 1,
                "dpo": {"beta": 0.1, "max_length": 128},
                "grpo": {
                    "group_size": 4,
                    "kl_threshold": 0.1,
                    "eval_interval_steps": 10,
                    "heldout_eval_size": 5,
                    "heldout_dataset": "gsm8k",
                    "eval_max_new_tokens": 64,
                    "save_best_checkpoint": True,
                    "best_checkpoint_metric": "pass_at_1",
                    "min_eval_improvement": 0.0001,
                    "early_stop_patience": 2,
                    "online_max_new_tokens": 64,
                    "online_temperature": 0.8,
                    "online_top_p": 0.95,
                    "online_resample_attempts": 1,
                    "online_min_reward_std": 1e-6,
                    "enable_ray_verification": False,
                    "ray_verifier_workers": 1,
                },
                "evaluation": {"num_paths": 2, "oracle_verify": False},
                "wandb": {"enabled": False, "project": "test-project", "mode": "offline"},
            },
            "general": {"output_dir": str(tmp_path / "outputs")},
        }
    )

    pipeline_module = _load_run_pipeline_module()
    monkeypatch.setattr(pipeline_module.OmegaConf, "load", lambda _: config)

    argv = [
        "run_pipeline.py",
        "--config", "ignored.yaml",
        "--dataset", "gsm8k",
        "--subset-size", "4",
        "--training-method", "best",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    pipeline_module.main()

    assert state["eval_model_name"] is not None
    assert state["eval_model_name"].endswith("best_checkpoint")


def test_resolve_evaluation_model_path_carries_checkpoint_selection_metadata(tmp_path):
    pipeline_module = _load_run_pipeline_module()
    config = OmegaConf.create(
        {
            "training": {
                "grpo": {
                    "save_best_checkpoint": True,
                }
            }
        }
    )

    model_dir = tmp_path / "grpo_model"
    best_dir = model_dir / "best_checkpoint"
    best_dir.mkdir(parents=True)
    (model_dir / "checkpoint_selection.json").write_text(
        json.dumps(
            {
                "best_eval_score": 0.74,
                "best_eval_step": 120,
                "best_checkpoint_metric": "pass_at_1",
            }
        )
    )

    selected_path, metadata = pipeline_module.resolve_evaluation_model_path(
        config,
        str(model_dir),
    )

    assert selected_path == str(best_dir)
    assert metadata["selection_reason"] == "best_checkpoint"
    assert metadata["selection_metric"] == "pass_at_1"
    assert metadata["selection_score"] == 0.74
    assert metadata["selection_step"] == 120


def test_run_evaluation_rejects_invalid_benchmark_runs(monkeypatch, tmp_path):
    pipeline_module = _load_run_pipeline_module()

    class FakeEvalResult:
        accuracy = 0.0
        errors = 4
        valid_run = False
        status = "failed"

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("{}")

        def to_dict(self):
            return {
                "benchmark": "GSM8K",
                "total": 4,
                "completed": 0,
                "correct": 0,
                "incorrect": 0,
                "errors": 4,
                "accuracy": 0.0,
                "avg_time": 0.0,
                "success_rate": 0.0,
                "status": "failed",
                "valid_run": False,
            }

    class FakeGSM8KEvaluator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def evaluate(self, subset_size):
            return FakeEvalResult()

    fake_evaluation = types.ModuleType("evaluation")
    fake_evaluation.GSM8KEvaluator = FakeGSM8KEvaluator
    fake_evaluation.HumanEvalEvaluator = object
    fake_evaluation.MATHEvaluator = object
    monkeypatch.setitem(sys.modules, "evaluation", fake_evaluation)

    config = OmegaConf.create(
        {
            "training": {
                "grpo": {"save_best_checkpoint": False},
                "evaluation": {
                    "num_paths": 2,
                    "oracle_verify": False,
                    "require_valid_run": True,
                },
            },
            "general": {"output_dir": str(tmp_path / "outputs")},
        }
    )

    class Args:
        dataset = "gsm8k"
        use_ttc = False
        ttc_oracle_verify = False
        eval_subset_size = 4

    with pytest.raises(RuntimeError, match="invalid benchmark runs"):
        pipeline_module.run_evaluation(config, "dummy-model", Args())
