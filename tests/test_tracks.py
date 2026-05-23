"""
Tests for optional track policy.
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracks import TrackPolicy, apply_track_policy


def test_track_policy_core_only_disables_optional_features():
    config = OmegaConf.create(
        {
            "tracks": {
                "core_only": True,
                "optional": {
                    "kafka_streaming": True,
                    "ray_verification": True,
                    "reward_model_reranking": True,
                    "process_reward_model_reranking": True,
                    "speculative_decoding": True,
                },
            },
            "orchestration": {"kafka": {"enabled": True}},
            "training": {
                "grpo": {"enable_ray_verification": True},
                "evaluation": {
                    "reward_model": "./reward_model",
                    "process_reward_model": "./process_reward_model",
                },
            },
            "speculative_decoding": {"enabled": True},
        }
    )

    policy = apply_track_policy(config)

    assert policy.enabled_optional_tracks() == []
    assert config.orchestration.kafka.enabled is False
    assert config.training.grpo.enable_ray_verification is False
    assert config.training.evaluation.reward_model is None
    assert config.training.evaluation.process_reward_model is None
    assert config.speculative_decoding.enabled is False


def test_track_policy_preserves_explicit_optional_track_enablement():
    config = OmegaConf.create(
        {
            "tracks": {
                "core_only": False,
                "optional": {
                    "ray_verification": True,
                },
            },
            "orchestration": {"kafka": {"enabled": True}},
            "training": {
                "grpo": {"enable_ray_verification": False},
                "evaluation": {
                    "reward_model": "./reward_model",
                },
            },
            "speculative_decoding": {"enabled": False},
        }
    )

    policy = apply_track_policy(config)

    assert policy == TrackPolicy(core_only=False, ray_verification=True)
    assert config.training.grpo.enable_ray_verification is True
    assert config.orchestration.kafka.enabled is False
    assert config.training.evaluation.reward_model is None
