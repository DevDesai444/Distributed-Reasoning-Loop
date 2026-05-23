"""
Inference module with Speculative Decoding support.
Optimizes inference throughput for reasoning models.
"""

CORE_INFERENCE_COMPONENTS = [
    "VLLMEngine",
    "VLLMConfig",
    "SGLangEngine",
    "SGLangConfig",
]

OPTIONAL_EXPERIMENTAL_COMPONENTS = {
    "throughput_experiments": [
        "SpeculativeDecoder",
        "SpeculativeConfig",
        "DraftTargetPair",
    ],
}

from .speculative_decoding import (
    SpeculativeDecoder,
    SpeculativeConfig,
    DraftTargetPair,
)

from .vllm_engine import (
    VLLMEngine,
    VLLMConfig,
)

from .sglang_engine import (
    SGLangEngine,
    SGLangConfig,
)

__all__ = [
    "SpeculativeDecoder",
    "SpeculativeConfig",
    "DraftTargetPair",
    "VLLMEngine",
    "VLLMConfig",
    "SGLangEngine",
    "SGLangConfig",
    "CORE_INFERENCE_COMPONENTS",
    "OPTIONAL_EXPERIMENTAL_COMPONENTS",
]
