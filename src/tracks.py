"""
Feature-track policy for keeping the core reasoning loop separate from optional extras.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def _set_path(obj: Any, path: Iterable[str], value: Any) -> None:
    keys = list(path)
    cursor = obj
    for key in keys[:-1]:
        next_value = _get_value(cursor, key)
        if next_value is None:
            next_value = {}
            try:
                cursor[key] = next_value
            except Exception:
                setattr(cursor, key, next_value)
        cursor = next_value

    last_key = keys[-1]
    try:
        cursor[last_key] = value
    except Exception:
        setattr(cursor, last_key, value)


@dataclass(frozen=True)
class TrackPolicy:
    """Optional-track policy applied to pipeline and evaluation configs."""

    core_only: bool = False
    kafka_streaming: bool = False
    ray_verification: bool = False
    reward_model_reranking: bool = False
    process_reward_model_reranking: bool = False
    speculative_decoding: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "TrackPolicy":
        tracks = _get_value(config, "tracks", {})
        optional = _get_value(tracks, "optional", {})
        return cls(
            core_only=bool(_get_value(tracks, "core_only", False)),
            kafka_streaming=bool(_get_value(optional, "kafka_streaming", False)),
            ray_verification=bool(_get_value(optional, "ray_verification", False)),
            reward_model_reranking=bool(_get_value(optional, "reward_model_reranking", False)),
            process_reward_model_reranking=bool(
                _get_value(optional, "process_reward_model_reranking", False)
            ),
            speculative_decoding=bool(_get_value(optional, "speculative_decoding", False)),
        )

    def normalized(self) -> "TrackPolicy":
        if not self.core_only:
            return self
        return replace(
            self,
            kafka_streaming=False,
            ray_verification=False,
            reward_model_reranking=False,
            process_reward_model_reranking=False,
            speculative_decoding=False,
        )

    def enabled_optional_tracks(self) -> list[str]:
        enabled = []
        if self.kafka_streaming:
            enabled.append("kafka_streaming")
        if self.ray_verification:
            enabled.append("ray_verification")
        if self.reward_model_reranking:
            enabled.append("reward_model_reranking")
        if self.process_reward_model_reranking:
            enabled.append("process_reward_model_reranking")
        if self.speculative_decoding:
            enabled.append("speculative_decoding")
        return enabled


def apply_track_policy(config: Any) -> TrackPolicy:
    """
    Apply optional-track policy directly onto a config-like object.

    This keeps the default path centered on the core reasoning loop while letting
    optional distributed or experimental features be turned on explicitly.
    """
    policy = TrackPolicy.from_config(config).normalized()

    _set_path(config, ("orchestration", "kafka", "enabled"), policy.kafka_streaming)
    _set_path(config, ("training", "grpo", "enable_ray_verification"), policy.ray_verification)
    if not policy.reward_model_reranking:
        _set_path(config, ("training", "evaluation", "reward_model"), None)
    if not policy.process_reward_model_reranking:
        _set_path(config, ("training", "evaluation", "process_reward_model"), None)
    _set_path(config, ("speculative_decoding", "enabled"), policy.speculative_decoding)
    return policy
