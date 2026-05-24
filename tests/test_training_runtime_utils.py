"""
Tests for adapter-aware model loading during continued training.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.dpo_trainer import DPOTrainerConfig, ReasoningDPOTrainer
from training.runtime_utils import load_causal_lm_for_training


class _FakeModel:
    def __init__(self):
        self._params = {
            "base.weight": types.SimpleNamespace(requires_grad=True),
            "lora_A.weight": types.SimpleNamespace(requires_grad=True),
        }

    def named_parameters(self):
        return list(self._params.items())


def test_load_causal_lm_for_training_detects_adapter_checkpoint(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter_model"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")

    calls = {"base_model_name": None, "adapter_path": None}

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["base_model_name"] = model_name
            return _FakeModel()

    class FakePeftConfig:
        @staticmethod
        def from_pretrained(path):
            assert path == str(adapter_dir)
            return types.SimpleNamespace(base_model_name_or_path="base-model")

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, path, is_trainable=True):
            calls["adapter_path"] = path
            assert is_trainable is True
            return base_model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(PeftConfig=FakePeftConfig, PeftModel=FakePeftModel),
    )

    model, use_8bit, loaded_adapter, base_model_name = load_causal_lm_for_training(
        str(adapter_dir),
        prefer_bf16=False,
        allow_8bit=False,
    )

    assert isinstance(model, _FakeModel)
    assert use_8bit is False
    assert loaded_adapter is True
    assert base_model_name == "base-model"
    assert calls["base_model_name"] == "base-model"
    assert calls["adapter_path"] == str(adapter_dir)


def test_dpo_train_preserves_non_trl_import_errors(monkeypatch):
    trainer = ReasoningDPOTrainer(DPOTrainerConfig())

    monkeypatch.setitem(
        sys.modules,
        "trl",
        types.SimpleNamespace(DPOTrainer=object, DPOConfig=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(Dataset=types.SimpleNamespace(from_list=lambda items: items)),
    )
    monkeypatch.setattr(trainer, "setup", lambda: (_ for _ in ()).throw(ImportError("adapter mismatch")))

    with pytest.raises(ImportError, match="adapter mismatch"):
        trainer.train([{"prompt": "p", "chosen": "c", "rejected": "r"}])
