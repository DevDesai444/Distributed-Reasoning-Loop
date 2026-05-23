"""
Tests for core/optional component registry.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from component_registry import component_registry_for_policy, core_component_registry
from tracks import TrackPolicy


def test_core_component_registry_has_major_domains():
    registry = core_component_registry()

    assert "training" in registry
    assert "evaluation" in registry
    assert "orchestration" in registry
    assert "inference" in registry
    assert "ReasoningGRPOTrainer" in registry["training"]


def test_component_registry_respects_optional_track_policy():
    registry = component_registry_for_policy(
        TrackPolicy(
            kafka_streaming=True,
            reward_model_reranking=True,
        )
    )

    assert registry["enabled_optional_tracks"] == [
        "kafka_streaming",
        "reward_model_reranking",
    ]
    assert "streaming" in registry["optional_enabled"]["orchestration"]
    assert "reward_modeling" in registry["optional_enabled"]["training"]
    assert registry["optional_enabled"]["inference"] == {}
