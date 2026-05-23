"""
Machine-readable registry of core and optional project components.
"""

from __future__ import annotations

from typing import Any, Dict

from evaluation import CORE_EVALUATION_COMPONENTS, OPTIONAL_EXPERIMENTAL_COMPONENTS as EVALUATION_OPTIONAL
from inference import CORE_INFERENCE_COMPONENTS, OPTIONAL_EXPERIMENTAL_COMPONENTS as INFERENCE_OPTIONAL
from orchestration import CORE_COMPONENTS as ORCHESTRATION_CORE, OPTIONAL_COMPONENTS as ORCHESTRATION_OPTIONAL
from tracks import TrackPolicy
from training import CORE_TRAINING_STACK, OPTIONAL_TRAINING_TRACKS


def core_component_registry() -> Dict[str, Any]:
    return {
        "orchestration": list(ORCHESTRATION_CORE),
        "training": list(CORE_TRAINING_STACK),
        "evaluation": list(CORE_EVALUATION_COMPONENTS),
        "inference": list(CORE_INFERENCE_COMPONENTS),
    }


def optional_component_registry() -> Dict[str, Any]:
    return {
        "orchestration": dict(ORCHESTRATION_OPTIONAL),
        "training": dict(OPTIONAL_TRAINING_TRACKS),
        "evaluation": dict(EVALUATION_OPTIONAL),
        "inference": dict(INFERENCE_OPTIONAL),
    }


def component_registry_for_policy(policy: TrackPolicy) -> Dict[str, Any]:
    policy = policy.normalized()
    enabled_optional = {
        "orchestration": {},
        "training": {},
        "evaluation": {},
        "inference": {},
    }

    if policy.kafka_streaming:
        enabled_optional["orchestration"]["streaming"] = ORCHESTRATION_OPTIONAL["streaming"]
    if policy.ray_verification:
        enabled_optional["orchestration"]["distributed_compute"] = ORCHESTRATION_OPTIONAL["distributed_compute"]
    if policy.reward_model_reranking:
        enabled_optional["training"]["reward_modeling"] = OPTIONAL_TRAINING_TRACKS["reward_modeling"]
    if policy.process_reward_model_reranking:
        enabled_optional["training"]["process_reward_modeling"] = OPTIONAL_TRAINING_TRACKS["process_reward_modeling"]
        enabled_optional["evaluation"]["search_strategies"] = EVALUATION_OPTIONAL["search_strategies"]
    if policy.speculative_decoding:
        enabled_optional["inference"]["throughput_experiments"] = INFERENCE_OPTIONAL["throughput_experiments"]

    return {
        "core": core_component_registry(),
        "optional_enabled": enabled_optional,
        "enabled_optional_tracks": policy.enabled_optional_tracks(),
    }
