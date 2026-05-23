"""
Machine-readable registry of core and optional project components.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

from tracks import TrackPolicy


def _module_attr(module_name: str, attr_name: str, default: Any) -> Any:
    try:
        module = import_module(module_name)
    except Exception:
        return default
    return getattr(module, attr_name, default)


def _core_training_stack() -> list[str]:
    return list(
        _module_attr(
            "training",
            "CORE_TRAINING_STACK",
            [
                "SFTTrainerConfig",
                "ReasoningSFTTrainer",
                "SFTFromSyntheticData",
                "DPOTrainerConfig",
                "ReasoningDPOTrainer",
                "GRPOConfig",
                "ReasoningGRPOTrainer",
            ],
        )
    )


def _optional_training_tracks() -> Dict[str, list[str]]:
    return dict(
        _module_attr(
            "training",
            "OPTIONAL_TRAINING_TRACKS",
            {
                "reward_modeling": ["RewardModel", "RewardModelConfig"],
                "process_reward_modeling": ["ProcessRewardModel", "ProcessRewardModelConfig"],
                "experimental_baselines": ["RejectionSamplingDPO"],
            },
        )
    )


def _core_evaluation_components() -> list[str]:
    return list(
        _module_attr(
            "evaluation",
            "CORE_EVALUATION_COMPONENTS",
            [
                "GSM8KEvaluator",
                "HumanEvalEvaluator",
                "MATHEvaluator",
                "BenchmarkResult",
                "run_all_benchmarks",
                "TestTimeCompute",
                "BestOfNSampler",
            ],
        )
    )


def _optional_evaluation_components() -> Dict[str, list[str]]:
    return dict(
        _module_attr(
            "evaluation",
            "OPTIONAL_EXPERIMENTAL_COMPONENTS",
            {
                "search_strategies": [
                    "BeamSearchReasoner",
                    "MCTSReasoner",
                ],
            },
        )
    )


def _core_inference_components() -> list[str]:
    return list(
        _module_attr(
            "inference",
            "CORE_INFERENCE_COMPONENTS",
            [
                "VLLMEngine",
                "VLLMConfig",
                "SGLangEngine",
                "SGLangConfig",
            ],
        )
    )


def _optional_inference_components() -> Dict[str, list[str]]:
    return dict(
        _module_attr(
            "inference",
            "OPTIONAL_EXPERIMENTAL_COMPONENTS",
            {
                "throughput_experiments": [
                    "SpeculativeDecoder",
                    "SpeculativeConfig",
                    "DraftTargetPair",
                ],
            },
        )
    )


def _core_orchestration_components() -> list[str]:
    return list(
        _module_attr(
            "orchestration",
            "CORE_COMPONENTS",
            [
                "KVCacheManager",
                "DistributedKVCache",
                "CacheEntry",
                "CacheStats",
            ],
        )
    )


def _optional_orchestration_components() -> Dict[str, list[str]]:
    return dict(
        _module_attr(
            "orchestration",
            "OPTIONAL_COMPONENTS",
            {
                "distributed_compute": [
                    "RayClusterManager",
                    "RayClusterConfig",
                    "DataProcessingWorker",
                    "TokenizationWorker",
                    "BatchPreparationWorker",
                ],
                "streaming": [
                    "KafkaProducer",
                    "KafkaConsumer",
                    "KafkaConfig",
                    "ReasoningDataProducer",
                    "ReasoningDataConsumer",
                ],
            },
        )
    )


def core_component_registry() -> Dict[str, Any]:
    return {
        "orchestration": _core_orchestration_components(),
        "training": _core_training_stack(),
        "evaluation": _core_evaluation_components(),
        "inference": _core_inference_components(),
    }


def optional_component_registry() -> Dict[str, Any]:
    return {
        "orchestration": _optional_orchestration_components(),
        "training": _optional_training_tracks(),
        "evaluation": _optional_evaluation_components(),
        "inference": _optional_inference_components(),
    }


def component_registry_for_policy(policy: TrackPolicy) -> Dict[str, Any]:
    policy = policy.normalized()
    orchestration_optional = _optional_orchestration_components()
    training_optional = _optional_training_tracks()
    evaluation_optional = _optional_evaluation_components()
    inference_optional = _optional_inference_components()

    enabled_optional = {
        "orchestration": {},
        "training": {},
        "evaluation": {},
        "inference": {},
    }

    if policy.kafka_streaming:
        enabled_optional["orchestration"]["streaming"] = orchestration_optional.get("streaming", [])
    if policy.ray_verification:
        enabled_optional["orchestration"]["distributed_compute"] = orchestration_optional.get(
            "distributed_compute", []
        )
    if policy.reward_model_reranking:
        enabled_optional["training"]["reward_modeling"] = training_optional.get("reward_modeling", [])
    if policy.process_reward_model_reranking:
        enabled_optional["training"]["process_reward_modeling"] = training_optional.get(
            "process_reward_modeling", []
        )
        enabled_optional["evaluation"]["search_strategies"] = evaluation_optional.get(
            "search_strategies", []
        )
    if policy.speculative_decoding:
        enabled_optional["inference"]["throughput_experiments"] = inference_optional.get(
            "throughput_experiments", []
        )

    return {
        "core": core_component_registry(),
        "optional_enabled": enabled_optional,
        "enabled_optional_tracks": policy.enabled_optional_tracks(),
    }
