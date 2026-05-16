"""
Tests for optional Kafka publishing hook in the pipeline script.
"""

from pathlib import Path
import importlib.util
import sys
import types

from omegaconf import OmegaConf


def _load_run_pipeline_module():
    """Load scripts/run_pipeline.py as a module for direct function testing."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"

    # run_pipeline imports wandb_utils at module import time.
    sys.modules.setdefault(
        "wandb_utils",
        types.SimpleNamespace(ensure_wandb_run=lambda *args, **kwargs: None, get_wandb=lambda: None),
    )

    spec = importlib.util.spec_from_file_location("run_pipeline_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_kafka_hook_disabled_noop():
    module = _load_run_pipeline_module()
    config = OmegaConf.create(
        {
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
            }
        }
    )

    module.stream_pipeline_data_to_kafka(config, samples=[], pairs=[])


def test_kafka_hook_enabled_streams_samples_and_pairs(monkeypatch):
    module = _load_run_pipeline_module()
    calls = {"raw": 0, "verified": 0, "training": 0, "topics_setup": 0, "closed": 0}

    class FakeKafkaConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeKafkaAdminClient:
        def __init__(self, cfg):
            self.cfg = cfg

        def setup_pipeline_topics(self):
            calls["topics_setup"] += 1

    class FakeReasoningDataProducer:
        def __init__(self, cfg):
            self.cfg = cfg

        def send_raw_reasoning(self, _payload):
            calls["raw"] += 1

        def send_verified_path(self, _payload):
            calls["verified"] += 1

        def send_training_sample(self, _payload):
            calls["training"] += 1

        def close(self):
            calls["closed"] += 1

    fake_mod = types.ModuleType("orchestration.kafka_streaming")
    fake_mod.KafkaConfig = FakeKafkaConfig
    fake_mod.KafkaAdminClient = FakeKafkaAdminClient
    fake_mod.ReasoningDataProducer = FakeReasoningDataProducer
    monkeypatch.setitem(sys.modules, "orchestration.kafka_streaming", fake_mod)

    config = OmegaConf.create(
        {
            "orchestration": {
                "kafka": {
                    "enabled": True,
                    "bootstrap_servers": ["localhost:9092"],
                    "topics": {
                        "raw_reasoning_data": "raw_reasoning_data",
                        "verified_paths": "verified_paths",
                        "training_data": "training_data",
                    },
                }
            }
        }
    )

    samples = [
        {"problem_id": "p1", "problem": "2+2", "reasoning": "4", "path_hash": "h1"},
        {"problem_id": "p2", "problem": "3+3", "reasoning": "6", "path_hash": "h2"},
    ]
    pairs = [
        {"problem_id": "p1", "problem": "2+2", "chosen": "4", "rejected": "5"},
    ]

    module.stream_pipeline_data_to_kafka(config, samples=samples, pairs=pairs)

    assert calls["topics_setup"] == 1
    assert calls["raw"] == len(samples)
    assert calls["verified"] == len(samples)
    assert calls["training"] == len(pairs)
    assert calls["closed"] == 1
