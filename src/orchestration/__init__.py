"""
Orchestration module for distributed data processing.
Kafka for streaming, Ray for distributed compute.
"""

CORE_COMPONENTS = [
    "KVCacheManager",
    "DistributedKVCache",
    "CacheEntry",
    "CacheStats",
]

OPTIONAL_COMPONENTS = {
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
}

# Kafka imports (optional)
try:
    from .kafka_streaming import (
        KafkaProducer,
        KafkaConsumer,
        KafkaConfig,
        ReasoningDataProducer,
        ReasoningDataConsumer,
    )
    _kafka_available = True
except ImportError:
    KafkaProducer = None
    KafkaConsumer = None
    KafkaConfig = None
    ReasoningDataProducer = None
    ReasoningDataConsumer = None
    _kafka_available = False

# Ray imports (optional)
try:
    from .ray_workers import (
        RayClusterManager,
        RayClusterConfig,
        DataProcessingWorker,
        TokenizationWorker,
        BatchPreparationWorker,
    )
    _ray_available = True
except ImportError:
    RayClusterManager = None
    RayClusterConfig = None
    DataProcessingWorker = None
    TokenizationWorker = None
    BatchPreparationWorker = None
    _ray_available = False

# KV Cache (always available)
from .kv_cache_manager import (
    KVCacheManager,
    DistributedKVCache,
    CacheEntry,
    CacheStats,
)

# Scaling utilities (pure Python, always available)
from .scaling import (
    StepRecord,
    ThroughputSummary,
    build_scaling_summary,
    measure_throughput,
    scaling_efficiency,
    summarize_records,
)

__all__ = [
    # Kafka
    "KafkaProducer",
    "KafkaConsumer",
    "KafkaConfig",
    "ReasoningDataProducer",
    "ReasoningDataConsumer",
    # Ray
    "RayClusterManager",
    "RayClusterConfig",
    "DataProcessingWorker",
    "TokenizationWorker",
    "BatchPreparationWorker",
    # KV Cache
    "KVCacheManager",
    "DistributedKVCache",
    "CacheEntry",
    "CacheStats",
    # Scaling
    "StepRecord",
    "ThroughputSummary",
    "build_scaling_summary",
    "measure_throughput",
    "scaling_efficiency",
    "summarize_records",
    # Availability flags
    "_kafka_available",
    "_ray_available",
    "CORE_COMPONENTS",
    "OPTIONAL_COMPONENTS",
]
