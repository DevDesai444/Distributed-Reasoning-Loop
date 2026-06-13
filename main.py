#!/usr/bin/env python3
"""
Distributed Reasoning Loop - Main Entry Point

A comprehensive pipeline for training reasoning models using:
- Synthetic data generation with verification
- Distributed orchestration with Kafka + Ray
- DPO/Rejection Sampling training
- Test-time compute evaluation
- Speculative decoding inference

Usage:
    python main.py generate --dataset gsm8k --num-paths 10
    python main.py train --data-path ./synthetic_data/dpo_pairs.jsonl
    python main.py evaluate --model ./dpo_output --benchmark gsm8k
    python main.py pipeline --dataset gsm8k --subset-size 100
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_generate(args):
    """Generate synthetic reasoning data."""
    from scripts.generate_synthetic_data import main as generate_main
    sys.argv = ['generate_synthetic_data.py'] + args.remaining
    generate_main()


def cmd_train(args):
    """Train model with DPO."""
    from scripts.train_dpo import main as train_main
    sys.argv = ['train_dpo.py'] + args.remaining
    train_main()


def cmd_train_grpo(args):
    """Train model with GRPO."""
    from src.training.grpo_trainer import (
        maybe_launch_grpo_distributed,
        train_grpo_from_synthetic_data,
    )
    import argparse

    def _build_distributed_args(parsed_args):
        forwarded = [
            "--data-path", parsed_args.data_path,
            "--output-dir", parsed_args.output_dir,
            "--model-name", parsed_args.model,
            "--learning-rate", str(parsed_args.learning_rate),
            "--num-epochs", str(parsed_args.epochs),
            "--batch-size", str(parsed_args.batch_size),
            "--group-size", str(parsed_args.group_size),
            "--max-length", str(parsed_args.max_length),
            "--max-prompt-length", str(parsed_args.max_prompt_length),
            "--prompt-problem-type", parsed_args.prompt_problem_type,
            "--verifier-timeout", str(parsed_args.verifier_timeout),
            "--code-docker-image", parsed_args.code_docker_image,
            "--code-memory-limit", parsed_args.code_memory_limit,
            "--kl-threshold", str(parsed_args.kl_threshold),
            "--eval-interval-steps", str(parsed_args.eval_interval_steps),
            "--online-max-new-tokens", str(parsed_args.online_max_new_tokens),
            "--online-temperature", str(parsed_args.online_temperature),
            "--online-top-p", str(parsed_args.online_top_p),
            "--online-resample-attempts", str(parsed_args.online_resample_attempts),
            "--online-min-reward-std", str(parsed_args.online_min_reward_std),
            "--ray-verifier-workers", str(parsed_args.ray_verifier_workers),
            "--heldout-dataset", parsed_args.heldout_dataset,
            "--heldout-split", parsed_args.heldout_split,
            "--heldout-eval-size", str(parsed_args.heldout_eval_size),
            "--eval-max-new-tokens", str(parsed_args.eval_max_new_tokens),
            "--best-checkpoint-metric", parsed_args.best_checkpoint_metric,
            "--min-eval-improvement", str(parsed_args.min_eval_improvement),
            "--early-stop-patience", str(parsed_args.early_stop_patience),
            "--wandb-project", parsed_args.wandb_project,
            "--wandb-mode", parsed_args.wandb_mode,
            "--distributed-timeout-minutes", str(parsed_args.distributed_timeout_minutes),
            "--num-gpus", str(parsed_args.num_gpus),
        ]
        if parsed_args.enable_ray_verification and not parsed_args.disable_ray_verification:
            forwarded.append("--enable-ray-verification")
        if parsed_args.disable_ray_verification:
            forwarded.append("--disable-ray-verification")
        if parsed_args.save_best_checkpoint and not parsed_args.disable_save_best_checkpoint:
            forwarded.append("--save-best-checkpoint")
        if parsed_args.disable_save_best_checkpoint:
            forwarded.append("--disable-save-best-checkpoint")
        if parsed_args.bf16 and not parsed_args.no_bf16:
            forwarded.append("--bf16")
        if parsed_args.no_bf16:
            forwarded.append("--no-bf16")
        if parsed_args.gradient_checkpointing and not parsed_args.no_gradient_checkpointing:
            forwarded.append("--gradient-checkpointing")
        if parsed_args.no_gradient_checkpointing:
            forwarded.append("--no-gradient-checkpointing")
        return forwarded
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./grpo_output')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--max-prompt-length', type=int, default=256)
    parser.add_argument('--prompt-problem-type', type=str, default='math')
    parser.add_argument('--verifier-timeout', type=int, default=10)
    parser.add_argument('--code-docker-image', type=str, default='distributed-reasoning-loop-sandbox:latest')
    parser.add_argument('--code-memory-limit', type=str, default='512m')
    parser.add_argument('--kl-threshold', type=float, default=0.1)
    parser.add_argument('--eval-interval-steps', type=int, default=50)
    parser.add_argument('--online-max-new-tokens', type=int, default=256)
    parser.add_argument('--online-temperature', type=float, default=0.8)
    parser.add_argument('--online-top-p', type=float, default=0.95)
    parser.add_argument('--online-resample-attempts', type=int, default=2)
    parser.add_argument('--online-min-reward-std', type=float, default=1e-6)
    parser.add_argument('--enable-ray-verification', action='store_true')
    parser.add_argument('--disable-ray-verification', action='store_true')
    parser.add_argument('--ray-verifier-workers', type=int, default=4)
    parser.add_argument('--heldout-dataset', type=str, default='gsm8k')
    parser.add_argument('--heldout-split', type=str, default='test')
    parser.add_argument('--heldout-eval-size', type=int, default=20)
    parser.add_argument('--eval-max-new-tokens', type=int, default=256)
    parser.add_argument('--save-best-checkpoint', action='store_true')
    parser.add_argument('--disable-save-best-checkpoint', action='store_true')
    parser.add_argument('--best-checkpoint-metric', type=str, default='pass_at_1')
    parser.add_argument('--min-eval-improvement', type=float, default=1e-4)
    parser.add_argument('--early-stop-patience', type=int, default=0)
    parser.add_argument('--wandb-project', type=str, default='distributed-reasoning-loop')
    parser.add_argument('--wandb-mode', type=str, default='offline')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no-bf16', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--no-gradient-checkpointing', action='store_true')
    parser.add_argument('--distributed-timeout-minutes', type=int, default=0)
    parser.add_argument('--num-gpus', type=str, default='auto')
    
    grpo_args = parser.parse_args(args.remaining)

    distributed_args = _build_distributed_args(grpo_args)

    if maybe_launch_grpo_distributed(distributed_args, requested_num_gpus=grpo_args.num_gpus):
        return
    
    train_grpo_from_synthetic_data(
        data_path=grpo_args.data_path,
        output_dir=grpo_args.output_dir,
        model_name=grpo_args.model,
        learning_rate=grpo_args.learning_rate,
        num_epochs=grpo_args.epochs,
        batch_size=grpo_args.batch_size,
        group_size=grpo_args.group_size,
        max_length=grpo_args.max_length,
        max_prompt_length=grpo_args.max_prompt_length,
        prompt_problem_type=grpo_args.prompt_problem_type,
        verifier_timeout=grpo_args.verifier_timeout,
        code_docker_image=grpo_args.code_docker_image,
        code_memory_limit=grpo_args.code_memory_limit,
        kl_threshold=grpo_args.kl_threshold,
        eval_interval_steps=grpo_args.eval_interval_steps,
        online_max_new_tokens=grpo_args.online_max_new_tokens,
        online_temperature=grpo_args.online_temperature,
        online_top_p=grpo_args.online_top_p,
        online_resample_attempts=grpo_args.online_resample_attempts,
        online_min_reward_std=grpo_args.online_min_reward_std,
        enable_ray_verification=(
            False
            if grpo_args.disable_ray_verification
            else True
        ),
        ray_verifier_workers=grpo_args.ray_verifier_workers,
        heldout_dataset=grpo_args.heldout_dataset,
        heldout_split=grpo_args.heldout_split,
        heldout_eval_size=grpo_args.heldout_eval_size,
        eval_max_new_tokens=grpo_args.eval_max_new_tokens,
        save_best_checkpoint=(
            False
            if grpo_args.disable_save_best_checkpoint
            else True
        ),
        best_checkpoint_metric=grpo_args.best_checkpoint_metric,
        min_eval_improvement=grpo_args.min_eval_improvement,
        early_stop_patience=grpo_args.early_stop_patience,
        wandb_project=grpo_args.wandb_project,
        wandb_mode=grpo_args.wandb_mode,
        bf16=False if grpo_args.no_bf16 else True,
        gradient_checkpointing=False if grpo_args.no_gradient_checkpointing else True,
        distributed_timeout_minutes=grpo_args.distributed_timeout_minutes,
    )


def cmd_evaluate(args):
    """Evaluate model on benchmarks."""
    from scripts.evaluate import main as evaluate_main
    sys.argv = ['evaluate.py'] + args.remaining
    evaluate_main()


def cmd_eval(args):
    """Eval-suite dispatcher."""
    remaining = args.remaining
    if not remaining:
        print("usage: python main.py eval <subcommand> [...]\n"
              "  subcommands:\n"
              "    agreement       three-way verifier/RM/judge agreement run\n"
              "    robust          perturbation-based robustness retention\n"
              "    contamination   n-gram overlap report between benchmark splits", file=sys.stderr)
        sys.exit(2)
    sub = remaining[0]
    rest = remaining[1:]
    if sub == "agreement":
        from scripts.agreement_eval import main as agreement_main
        raise SystemExit(agreement_main(rest))
    if sub == "robust":
        from scripts.robustness_eval import main as robust_main
        raise SystemExit(robust_main(rest))
    if sub == "contamination":
        from scripts.contamination_eval import main as contamination_main
        raise SystemExit(contamination_main(rest))
    print(f"unknown eval subcommand: {sub!r}", file=sys.stderr)
    sys.exit(2)


def cmd_pipeline(args):
    """Run full pipeline."""
    from scripts.run_pipeline import main as pipeline_main
    sys.argv = ['run_pipeline.py'] + args.remaining
    pipeline_main()


def cmd_serve(args):
    """Start inference server."""
    try:
        from vllm.entrypoints.openai.api_server import run_server
        logger.info("Starting vLLM inference server...")
        # Parse remaining args for server
        model = args.model or "Qwen/Qwen2.5-7B-Instruct"
        port = args.port or 8000
        
        sys.argv = [
            'api_server',
            '--model', model,
            '--port', str(port),
        ]
        run_server()
    except ImportError:
        logger.error("vLLM not installed. Install with: pip install vllm")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Distributed Reasoning Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate synthetic data:
    python main.py generate --dataset gsm8k --num-paths 10
    
  Train with DPO:
    python main.py train --data-path ./synthetic_data/dpo_pairs.jsonl --model Qwen/Qwen2.5-7B
    
  Evaluate model:
    python main.py evaluate --model ./dpo_output --benchmark gsm8k --use-ttc

  Run full pipeline:
    python main.py pipeline --dataset gsm8k --subset-size 100 --training-method best
    
  Start inference server:
    python main.py serve --model ./dpo_output --port 8000
        """
    )
    
    # Only parse the first argument (subcommand)
    parser.add_argument('command', nargs='?',
                        choices=['generate', 'train', 'train-grpo', 'evaluate', 'eval', 'pipeline', 'serve'],
                        help='Command to run')
    
    # Parse only the first argument
    args, remaining = parser.parse_known_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Map commands to functions
    cmd_map = {
        'generate': cmd_generate,
        'train': cmd_train,
        'train-grpo': cmd_train_grpo,
        'evaluate': cmd_evaluate,
        'eval': cmd_eval,
        'pipeline': cmd_pipeline,
        'serve': cmd_serve,
    }
    
    # Create args object with remaining arguments
    class Args:
        pass
    
    cmd_args = Args()
    cmd_args.remaining = remaining
    cmd_args.model = None
    cmd_args.port = 8000
    
    # For serve command, extract --model and --port
    if args.command == 'serve':
        serve_parser = argparse.ArgumentParser()
        serve_parser.add_argument('--model', type=str)
        serve_parser.add_argument('--port', type=int, default=8000)
        serve_args, extra = serve_parser.parse_known_args(remaining)
        cmd_args.model = serve_args.model
        cmd_args.port = serve_args.port
        cmd_args.remaining = extra
    
    cmd_map[args.command](cmd_args)


if __name__ == "__main__":
    main()
