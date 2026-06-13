"""
Data Generator module for synthetic reasoning path generation.
Uses teacher models to create Chain-of-Thought solutions.
"""

from .cot_generator import (
    CoTGenerator,
    ReasoningPath,
    GenerationConfig,
)

from .dataset_loader import (
    DatasetLoader,
    GSM8KLoader,
    HumanEvalLoader,
    Problem,
)

from .synthetic_data_pipeline import (
    SyntheticDataPipeline,
    FilteredSample,
    SamplePair,
)

from .data_preprocessor import (
    DataPreprocessor,
    PreprocessConfig,
    preprocess_jsonl,
)

from .provenance import (
    PROVENANCE_SCHEMA_VERSION,
    ProvenanceMetadata,
    build_provenance,
    dedup_hash,
    pair_id,
)

from .quality_score import (
    QualityComponents,
    score_components,
    score_trace,
)

from .stats import RunStats, BackendCounters

from .checkpoint import PairCheckpointer, CheckpointState

__all__ = [
    "CoTGenerator",
    "ReasoningPath",
    "GenerationConfig",
    "DatasetLoader",
    "GSM8KLoader",
    "HumanEvalLoader",
    "Problem",
    "SyntheticDataPipeline",
    "FilteredSample",
    "SamplePair",
    "DataPreprocessor",
    "PreprocessConfig",
    "preprocess_jsonl",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceMetadata",
    "build_provenance",
    "dedup_hash",
    "pair_id",
    "QualityComponents",
    "score_components",
    "score_trace",
    "RunStats",
    "BackendCounters",
    "PairCheckpointer",
    "CheckpointState",
]
