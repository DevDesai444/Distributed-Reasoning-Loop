#!/usr/bin/env python3
"""
Script to generate synthetic reasoning data using the pipeline.
Generates Chain-of-Thought solutions and creates DPO training pairs.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import SyntheticDataPipeline, GenerationConfig, PreprocessConfig
from omegaconf import OmegaConf
from run_artifacts import RunArtifacts
from tracks import apply_track_policy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _resolve_tensor_parallel_size(requested: Any, backend_name: str) -> int:
    """Resolve tensor parallel size, including auto mode for vLLM."""
    if backend_name != "vllm":
        return 1

    requested_value = "auto" if requested is None else str(requested).strip().lower()
    if requested_value in {"auto", "all"}:
        try:
            import torch

            available = torch.cuda.device_count() if torch.cuda.is_available() else 0
            return max(1, available)
        except Exception:
            return 1

    try:
        parsed = int(requested_value)
    except ValueError:
        logger.warning(
            "Invalid tensor parallel size '%s'; falling back to 1.",
            requested,
        )
        return 1

    return max(1, parsed)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "humaneval", "math", "mbpp"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for generation (overrides config)",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=10,
        help="Number of reasoning paths per problem",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of problems to process (None for all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_data",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["vllm", "sglang", "transformers"],
        help="Inference backend to use",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=str,
        default=None,
        help="Tensor parallel size for vLLM (auto|all|1|2|...)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM GPU memory utilization target (0.0-1.0)",
    )
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = OmegaConf.load(args.config)
    else:
        config = OmegaConf.create({})
    track_policy = apply_track_policy(config)
    config_payload = OmegaConf.to_container(config, resolve=True)
    
    data_cfg = config.get("data_generator", {})
    run_artifacts = RunArtifacts(
        root_output_dir=args.output_dir,
        dataset=args.dataset,
        training_method="data_generation",
        pipeline="synthetic_data_generation",
        config=config_payload,
        nested=False,
    )

    # Override with command line args
    model_name = args.model or data_cfg.get(
        "teacher_model", "meta-llama/Llama-3-70B-Instruct"
    )
    
    # Create generation config
    from data_generator.cot_generator import InferenceBackend
    
    backend_map = {
        "vllm": InferenceBackend.VLLM,
        "sglang": InferenceBackend.SGLANG,
        "transformers": InferenceBackend.TRANSFORMERS,
    }
    backend_name = args.backend or data_cfg.get("backend", "vllm")
    tensor_parallel_size = _resolve_tensor_parallel_size(
        args.tensor_parallel_size if args.tensor_parallel_size is not None else data_cfg.get("tensor_parallel_size", "auto"),
        backend_name=backend_name,
    )
    gpu_memory_utilization = (
        args.gpu_memory_utilization
        if args.gpu_memory_utilization is not None
        else float(data_cfg.get("gpu_memory_utilization", 0.9))
    )
    
    gen_config = GenerationConfig(
        model_name=model_name,
        backend=backend_map[backend_name],
        num_paths=args.num_paths,
        max_new_tokens=data_cfg.get("max_new_tokens", 2048),
        temperature=data_cfg.get("temperature", 0.8),
        top_p=data_cfg.get("top_p", 0.95),
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    
    # Create pipeline
    pipeline = SyntheticDataPipeline(
        generator_config=gen_config,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        preprocess_config=PreprocessConfig(
            **OmegaConf.to_container(
                data_cfg.get("preprocessing", {}),
                resolve=True,
            )
        ) if data_cfg.get("preprocessing") else None,
    )
    
    # Run pipeline
    logger.info(f"Starting synthetic data generation for {args.dataset}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Backend: {backend_name}")
    logger.info(
        "Optional tracks enabled: %s",
        ", ".join(track_policy.enabled_optional_tracks()) or "none",
    )
    if backend_name == "vllm":
        logger.info(f"vLLM tensor_parallel_size: {tensor_parallel_size}")
        logger.info(f"vLLM gpu_memory_utilization: {gpu_memory_utilization}")
    logger.info(f"Paths per problem: {args.num_paths}")
    logger.info(f"Output directory: {args.output_dir}")
    
    samples, pairs = pipeline.run(
        subset_size=args.subset_size,
        batch_size=args.batch_size,
    )

    output_dir = Path(args.output_dir)
    for artifact_name in [
        "all_samples.jsonl",
        "filtered_samples.jsonl",
        "correct_samples.jsonl",
        "incorrect_samples.jsonl",
        "dpo_pairs.jsonl",
        "full_pairs.jsonl",
        "smart_pairs.jsonl",
        "problem_manifest.json",
        "stats.json",
        "preprocess_stats.json",
        "pair_quality_summary.json",
        "sample_quality_summary.json",
    ]:
        artifact_path = output_dir / artifact_name
        if artifact_path.exists():
            run_artifacts.record_artifact(
                stage="generation",
                name=artifact_name,
                path=artifact_path,
            )

    correct_count = sum(1 for sample in samples if sample.is_correct)
    run_artifacts.record_metric("sample_count", len(samples))
    run_artifacts.record_metric("correct_sample_count", correct_count)
    run_artifacts.record_metric("pair_count", len(pairs))
    run_artifacts.finalize(
        "completed",
        summary={
            "sample_count": len(samples),
            "correct_sample_count": correct_count,
            "pair_count": len(pairs),
            "enabled_optional_tracks": track_policy.enabled_optional_tracks(),
        },
    )
    
    logger.info(f"Generation complete!")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Correct samples: {sum(1 for s in samples if s.is_correct)}")
    logger.info(f"DPO pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
