#!/usr/bin/env python3
"""
Main script to run the full distributed reasoning loop pipeline.
Orchestrates data generation, training, and evaluation.
"""

import argparse
import inspect
import json
import logging
import sys
from pathlib import Path
import time
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import OmegaConf
from run_artifacts import RunArtifacts
from tracks import apply_track_policy
from wandb_utils import ensure_wandb_run, get_wandb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def stream_pipeline_data_to_kafka(config, samples, pairs):
    """Optionally stream pipeline artifacts to Kafka topics."""
    kafka_cfg = config.orchestration.kafka
    if not bool(kafka_cfg.get("enabled", False)):
        logger.info("Kafka streaming disabled; skipping orchestration stream publish.")
        return

    try:
        from orchestration.kafka_streaming import KafkaConfig, KafkaAdminClient, ReasoningDataProducer
    except Exception as exc:
        logger.warning("Kafka modules unavailable; skipping stream publish: %s", exc)
        return

    try:
        topic_cfg = kafka_cfg.topics
        producer_cfg = KafkaConfig(
            bootstrap_servers=list(kafka_cfg.bootstrap_servers),
            raw_reasoning_topic=str(topic_cfg.raw_reasoning_data),
            verified_paths_topic=str(topic_cfg.verified_paths),
            training_data_topic=str(topic_cfg.training_data),
        )
        admin = KafkaAdminClient(producer_cfg)
        admin.setup_pipeline_topics()
        producer = ReasoningDataProducer(producer_cfg)
    except Exception as exc:
        logger.warning("Failed to initialize Kafka producer; skipping stream publish: %s", exc)
        return

    try:
        for sample in samples:
            sample_dict = sample.to_dict() if hasattr(sample, "to_dict") else sample
            producer.send_raw_reasoning(
                {
                    "problem_id": sample_dict.get("problem_id"),
                    "problem": sample_dict.get("problem"),
                    "reasoning": sample_dict.get("reasoning"),
                    "path_hash": sample_dict.get("path_hash"),
                }
            )
            producer.send_verified_path(sample_dict)

        for pair in pairs:
            pair_dict = pair.to_dict() if hasattr(pair, "to_dict") else pair
            producer.send_training_sample(
                {
                    "problem_id": pair_dict.get("problem_id"),
                    "prompt": pair_dict.get("problem", ""),
                    "chosen": pair_dict.get("chosen", ""),
                    "rejected": pair_dict.get("rejected", ""),
                }
            )
        logger.info(
            "Published pipeline artifacts to Kafka: %s samples, %s training pairs",
            len(samples),
            len(pairs),
        )
    except Exception as exc:
        logger.warning("Kafka publish failed mid-stream: %s", exc)
    finally:
        producer.close()


def _dataset_problem_type(dataset_name: str) -> str:
    return "code" if dataset_name in {"humaneval", "mbpp"} else "math"


def _construct_with_supported_kwargs(factory, **kwargs):
    """Instantiate an object while tolerating older/fake constructors in tests."""
    signature = inspect.signature(factory)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return factory(**supported)


def resolve_evaluation_model_path(config, model_path: str) -> tuple[str, Dict[str, Any]]:
    """
    Choose the most appropriate checkpoint for final evaluation.

    If GRPO saved a dedicated best checkpoint, prefer that over the latest
    training output so benchmark numbers reflect the strongest held-out model.
    """
    metadata = {
        "selected_model_path": model_path,
        "selection_reason": "final_model",
    }
    model_dir = Path(model_path)
    best_checkpoint_dir = model_dir / "best_checkpoint"
    selection_state_path = model_dir / "checkpoint_selection.json"

    selection_state: Dict[str, Any] = {}
    if selection_state_path.exists():
        try:
            with open(selection_state_path) as handle:
                selection_state = json.load(handle)
        except Exception as exc:
            logger.warning("Unable to read checkpoint selection state from %s: %s", selection_state_path, exc)

    if (
        bool(config.training.grpo.get("save_best_checkpoint", False))
        and best_checkpoint_dir.exists()
        and selection_state.get("best_eval_step") is not None
    ):
        metadata["selected_model_path"] = str(best_checkpoint_dir)
        metadata["selection_reason"] = "best_checkpoint"
        metadata["selection_metric"] = selection_state.get("best_checkpoint_metric")
        metadata["selection_score"] = selection_state.get("best_eval_score")
        metadata["selection_step"] = selection_state.get("best_eval_step")
        return str(best_checkpoint_dir), metadata

    if selection_state:
        metadata["selection_metric"] = selection_state.get("best_checkpoint_metric")
        metadata["selection_score"] = selection_state.get("best_eval_score")
        metadata["selection_step"] = selection_state.get("best_eval_step")

    return model_path, metadata


def run_data_generation(config, args):
    """Run synthetic data generation phase."""
    from data_generator import SyntheticDataPipeline, GenerationConfig, PreprocessConfig
    from data_generator.cot_generator import InferenceBackend

    def resolve_tensor_parallel_size(requested, backend_name: str) -> int:
        if backend_name != "vllm":
            return 1
        value = "auto" if requested is None else str(requested).strip().lower()
        if value in {"auto", "all"}:
            try:
                import torch

                available = torch.cuda.device_count() if torch.cuda.is_available() else 0
                return max(1, available)
            except Exception:
                return 1
        try:
            return max(1, int(value))
        except ValueError:
            logger.warning("Invalid tensor_parallel_size=%s, defaulting to 1", requested)
            return 1
    
    logger.info("=" * 50)
    logger.info("Phase 1: Synthetic Data Generation")
    logger.info("=" * 50)

    backend_map = {}
    if hasattr(InferenceBackend, "VLLM"):
        backend_map["vllm"] = InferenceBackend.VLLM
    if hasattr(InferenceBackend, "SGLANG"):
        backend_map["sglang"] = InferenceBackend.SGLANG
    if hasattr(InferenceBackend, "TRANSFORMERS"):
        backend_map["transformers"] = InferenceBackend.TRANSFORMERS

    backend_name = str(config.data_generator.get("backend", "vllm")).lower()
    if backend_name not in backend_map:
        raise ValueError(f"Unsupported data_generator.backend='{backend_name}'")
    tensor_parallel_size = resolve_tensor_parallel_size(
        config.data_generator.get("tensor_parallel_size", "auto"),
        backend_name=backend_name,
    )
    gpu_memory_utilization = float(config.data_generator.get("gpu_memory_utilization", 0.9))
    
    gen_config = GenerationConfig(
        model_name=config.data_generator.teacher_model,
        backend=backend_map[backend_name],
        num_paths=config.data_generator.num_cot_paths,
        max_new_tokens=config.data_generator.max_new_tokens,
        temperature=config.data_generator.temperature,
        top_p=config.data_generator.get("top_p", 0.95),
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    logger.info("Generation backend: %s", backend_name)
    if backend_name == "vllm":
        logger.info(
            "vLLM tensor_parallel_size=%s gpu_memory_utilization=%.2f",
            tensor_parallel_size,
            gpu_memory_utilization,
        )
    
    pipeline = SyntheticDataPipeline(
        generator_config=gen_config,
        dataset_name=args.dataset,
        output_dir=f"{config.general.output_dir}/synthetic_data",
        preprocess_config=PreprocessConfig(
            **OmegaConf.to_container(
                config.data_generator.get("preprocessing", {}),
                resolve=True,
            )
        ) if config.data_generator.get("preprocessing") else None,
    )
    
    samples, pairs = pipeline.run(
        subset_size=args.subset_size,
        batch_size=args.batch_size,
    )
    stream_pipeline_data_to_kafka(config, samples, pairs)
    
    logger.info(f"Generated {len(samples)} samples, {len(pairs)} DPO pairs")
    return samples, pairs


def run_sft_training(config, data_path, dataset_name):
    """Run supervised fine-tuning phase."""
    from training import SFTTrainerConfig, SFTFromSyntheticData
    
    logger.info("=" * 50)
    logger.info("Phase 2a: Supervised Fine-Tuning")
    logger.info("=" * 50)
    
    sft_config = _construct_with_supported_kwargs(
        SFTTrainerConfig,
        model_name=config.data_generator.student_model,
        learning_rate=config.training.learning_rate * 2,  # Higher LR for SFT
        batch_size=config.training.batch_size,
        num_epochs=1,  # Quick SFT pass
        problem_type=_dataset_problem_type(dataset_name),
        output_dir=f"{config.general.output_dir}/sft_model",
    )
    
    trainer = SFTFromSyntheticData(sft_config, data_path)
    trainer.train()
    
    return sft_config.output_dir


def run_dpo_training(config, data_path, dataset_name, base_model=None):
    """Run DPO training phase."""
    from training import DPOTrainerConfig, ReasoningDPOTrainer
    import json
    
    logger.info("=" * 50)
    logger.info("Phase 2b: DPO Training")
    logger.info("=" * 50)
    
    # Load DPO data
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    model_name = base_model or config.data_generator.student_model
    
    dpo_config = _construct_with_supported_kwargs(
        DPOTrainerConfig,
        model_name=model_name,
        beta=config.training.dpo.beta,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        num_epochs=config.training.num_epochs,
        max_length=config.training.dpo.max_length,
        problem_type=_dataset_problem_type(dataset_name),
        output_dir=f"{config.general.output_dir}/dpo_model",
    )
    
    trainer = ReasoningDPOTrainer(dpo_config)
    trainer.train(data)
    
    return dpo_config.output_dir


def run_grpo_training(config, data_path, dataset_name, base_model=None):
    """Run GRPO training phase."""
    from training.grpo_trainer import GRPOConfig, ReasoningGRPOTrainer
    import json
    
    logger.info("=" * 50)
    logger.info("Phase 2b: GRPO Training")
    logger.info("=" * 50)
    
    # Load data
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    model_name = base_model or config.data_generator.student_model
    
    grpo_config = _construct_with_supported_kwargs(
        GRPOConfig,
        model_name=model_name,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        num_epochs=config.training.num_epochs,
        max_length=config.training.dpo.max_length,
        group_size=config.training.grpo.group_size,
        verifier_type=config.verifier.type,
        verifier_timeout=config.verifier[config.verifier.type].timeout,
        code_docker_image=config.verifier.code.docker_image,
        code_memory_limit=config.verifier.code.memory_limit,
        kl_threshold=config.training.grpo.kl_threshold,
        eval_interval_steps=config.training.grpo.eval_interval_steps,
        heldout_eval_size=config.training.grpo.heldout_eval_size,
        heldout_dataset=config.training.grpo.heldout_dataset,
        eval_max_new_tokens=config.training.grpo.eval_max_new_tokens,
        save_best_checkpoint=config.training.grpo.save_best_checkpoint,
        best_checkpoint_metric=config.training.grpo.best_checkpoint_metric,
        min_eval_improvement=config.training.grpo.min_eval_improvement,
        early_stop_patience=config.training.grpo.early_stop_patience,
        online_max_new_tokens=config.training.grpo.online_max_new_tokens,
        online_temperature=config.training.grpo.online_temperature,
        online_top_p=config.training.grpo.online_top_p,
        online_resample_attempts=config.training.grpo.online_resample_attempts,
        online_min_reward_std=config.training.grpo.online_min_reward_std,
        enable_ray_verification=config.training.grpo.enable_ray_verification,
        ray_verifier_workers=config.training.grpo.ray_verifier_workers,
        prompt_problem_type=_dataset_problem_type(dataset_name),
        wandb_project=config.training.wandb.project,
        wandb_mode=config.training.wandb.mode,
        output_dir=f"{config.general.output_dir}/grpo_model",
    )
    
    trainer = ReasoningGRPOTrainer(grpo_config)
    trainer.train(data)
    
    return grpo_config.output_dir


def run_evaluation(config, model_path, args):
    """Run evaluation phase."""
    from evaluation import GSM8KEvaluator, HumanEvalEvaluator, MATHEvaluator
    
    logger.info("=" * 50)
    logger.info("Phase 3: Evaluation")
    logger.info("=" * 50)
    
    results = {}
    evaluation_model_path, selection_metadata = resolve_evaluation_model_path(config, model_path)
    ttc_oracle_verify = bool(args.ttc_oracle_verify or config.training.evaluation.get("oracle_verify", False))
    
    if args.dataset == "gsm8k":
        evaluator = _construct_with_supported_kwargs(
            GSM8KEvaluator,
            model_name=evaluation_model_path,
            use_test_time_compute=args.use_ttc,
            ttc_samples=config.training.evaluation.num_paths,
            ttc_oracle_verify=ttc_oracle_verify,
        )
        result = evaluator.evaluate(subset_size=args.eval_subset_size)
        results["gsm8k"] = result
        logger.info(f"GSM8K Accuracy: {result.accuracy:.2%}")

    if args.dataset == "math":
        evaluator = _construct_with_supported_kwargs(
            MATHEvaluator,
            model_name=evaluation_model_path,
            use_test_time_compute=args.use_ttc,
            ttc_samples=config.training.evaluation.num_paths,
            ttc_oracle_verify=ttc_oracle_verify,
        )
        result = evaluator.evaluate(subset_size=args.eval_subset_size)
        results["math"] = result
        logger.info(f"MATH Accuracy: {result.accuracy:.2%}")
    
    if args.dataset in ["humaneval", "mbpp"]:
        evaluator = HumanEvalEvaluator(model_name=evaluation_model_path)
        result = evaluator.evaluate(subset_size=args.eval_subset_size)
        results["humaneval"] = result
        logger.info(f"HumanEval Pass@1: {result.accuracy:.2%}")
    
    # Save results
    for name, result in results.items():
        result.save(f"{config.general.output_dir}/{name}_results.json")
    with open(f"{config.general.output_dir}/evaluation_selection.json", "w") as handle:
        json.dump(selection_metadata, handle, indent=2)
    
    return results, selection_metadata


def main():
    parser = argparse.ArgumentParser(description="Run full reasoning loop pipeline")
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
        "--subset-size",
        type=int,
        default=100,
        help="Number of problems to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip data generation phase",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip SFT phase",
    )
    parser.add_argument(
        "--run-sft",
        action="store_true",
        help="Run SFT phase (overrides --skip-sft)",
    )
    parser.add_argument(
        "--skip-dpo",
        action="store_true",
        help="Skip DPO training phase",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation phase",
    )
    parser.add_argument(
        "--use-ttc",
        action="store_true",
        help="Use test-time compute for evaluation",
    )
    parser.add_argument(
        "--ttc-oracle-verify",
        action="store_true",
        help="Allow ground-truth verifier reranking during TTC evaluation (analysis only, not a valid benchmark setting)",
    )
    parser.add_argument(
        "--eval-subset-size",
        type=int,
        default=None,
        help="Number of problems for evaluation (None for all, overrides --subset-size for eval)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to existing synthetic data (skips generation)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to existing trained model (skips training)",
    )
    parser.add_argument(
        "--training-method",
        type=str,
        default=None,
        choices=["dpo", "grpo", "best"],
        help="Training method: dpo, grpo, or best (SFT -> DPO -> GRPO)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit run name for the pipeline artifact directory",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    track_policy = apply_track_policy(config)
    training_method = args.training_method or str(config.training.get("method", "best"))

    config_payload = OmegaConf.to_container(config, resolve=True)
    run_artifacts = RunArtifacts(
        root_output_dir=str(config.general.output_dir),
        dataset=args.dataset,
        training_method=training_method,
        config=config_payload,
        run_name=args.run_name,
    )
    config.general.output_dir = str(run_artifacts.run_dir)
    enabled_tracks = track_policy.enabled_optional_tracks()
    if enabled_tracks:
        run_artifacts.add_note(
            "Enabled optional tracks: " + ", ".join(enabled_tracks)
        )
    else:
        run_artifacts.add_note("Running core reasoning loop without optional tracks.")

    if config.training.wandb.enabled:
        config_payload = OmegaConf.to_container(config, resolve=True)
        ensure_wandb_run(
            project=config.training.wandb.project,
            name=f"pipeline-{args.dataset}-{run_artifacts.manifest.run_id}",
            mode=config.training.wandb.mode,
            config=config_payload,
            tags=["pipeline", args.dataset],
        )
        wandb = get_wandb()
        if wandb is not None and wandb.run is not None:
            wandb.config.update(config_payload, allow_val_change=True)
    
    # Create output directory
    Path(config.general.output_dir).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    logger.info("Starting Distributed Reasoning Loop Pipeline")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Run directory: {run_artifacts.run_dir}")
    
    # Phase 1: Data Generation
    if not args.skip_generation and not args.data_path:
        samples, pairs = run_data_generation(config, args)
        data_path = f"{config.general.output_dir}/synthetic_data/dpo_pairs.jsonl"
        correct_data_path = f"{config.general.output_dir}/synthetic_data/correct_samples.jsonl"
        run_artifacts.record_artifact(
            stage="generation",
            name="synthetic_data_dir",
            path=f"{config.general.output_dir}/synthetic_data",
            metadata={
                "sample_count": len(samples),
                "pair_count": len(pairs),
            },
        )
    else:
        data_path = args.data_path or f"{config.general.output_dir}/synthetic_data/dpo_pairs.jsonl"
        correct_data_path = args.data_path.replace("dpo_pairs", "correct_samples") if args.data_path else None
        run_artifacts.record_artifact(
            stage="generation",
            name="reused_data_path",
            path=data_path,
            metadata={"reused": True},
        )
    
    # Phase 2a: SFT (optional, disabled by default)
    sft_model = None
    run_sft = (args.run_sft or training_method == "best") and not args.skip_sft
    if run_sft and correct_data_path and Path(correct_data_path).exists():
        sft_model = run_sft_training(config, correct_data_path, args.dataset)
        run_artifacts.record_artifact(
            stage="training",
            name="sft_model",
            path=sft_model,
        )
    
    # Phase 2b: DPO/GRPO Training
    if not args.skip_dpo and not args.model_path:
        if training_method == "grpo":
            model_path = run_grpo_training(config, data_path, args.dataset, sft_model)
        elif training_method == "best":
            dpo_base_model = sft_model or config.data_generator.student_model
            dpo_model = run_dpo_training(config, data_path, args.dataset, dpo_base_model)
            run_artifacts.record_artifact(
                stage="training",
                name="dpo_model",
                path=dpo_model,
            )
            model_path = run_grpo_training(config, data_path, args.dataset, dpo_model)
        else:
            model_path = run_dpo_training(config, data_path, args.dataset, sft_model)
    else:
        model_path = args.model_path or config.data_generator.student_model
    run_artifacts.record_artifact(
        stage="training",
        name="selected_model",
        path=model_path,
        metadata={"training_method": training_method},
    )
    
    # Phase 3: Evaluation
    results = {}
    if not args.skip_eval:
        results, selection_metadata = run_evaluation(config, model_path, args)
        for benchmark_name, result in results.items():
            run_artifacts.record_metric(f"{benchmark_name}_accuracy", result.accuracy)
            run_artifacts.record_metric(f"{benchmark_name}_errors", result.errors)
            result_path = Path(config.general.output_dir) / f"{benchmark_name}_results.json"
            run_artifacts.record_artifact(
                stage="evaluation",
                name=f"{benchmark_name}_results",
                path=result_path,
                metadata=result.to_dict(),
            )
        run_artifacts.record_artifact(
            stage="evaluation",
            name="selected_eval_model",
            path=selection_metadata["selected_model_path"],
            metadata=selection_metadata,
        )
    
    elapsed = time.time() - start_time
    summary = {
        "dataset": args.dataset,
        "training_method": training_method,
        "model_path": model_path,
        "evaluation_model_path": (
            selection_metadata["selected_model_path"]
            if not args.skip_eval
            else model_path
        ),
        "elapsed_minutes": elapsed / 60.0,
        "evaluation": {
            name: result.to_dict()
            for name, result in results.items()
        } if not args.skip_eval else {},
    }
    with open(Path(config.general.output_dir) / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    run_artifacts.record_artifact(
        stage="pipeline",
        name="pipeline_summary",
        path=Path(config.general.output_dir) / "pipeline_summary.json",
    )
    run_artifacts.finalize("completed", summary=summary)
    logger.info("=" * 50)
    logger.info(f"Pipeline complete! Total time: {elapsed/60:.1f} minutes")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
