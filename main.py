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
            "--num-epochs", str(parsed_args.epochs),
            "--batch-size", str(parsed_args.batch_size),
            "--group-size", str(parsed_args.group_size),
            "--online-max-new-tokens", str(parsed_args.online_max_new_tokens),
            "--online-temperature", str(parsed_args.online_temperature),
            "--online-top-p", str(parsed_args.online_top_p),
            "--online-resample-attempts", str(parsed_args.online_resample_attempts),
            "--ray-verifier-workers", str(parsed_args.ray_verifier_workers),
            "--num-gpus", str(parsed_args.num_gpus),
        ]
        if parsed_args.enable_ray_verification and not parsed_args.disable_ray_verification:
            forwarded.append("--enable-ray-verification")
        if parsed_args.disable_ray_verification:
            forwarded.append("--disable-ray-verification")
        return forwarded
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./grpo_output')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--online-max-new-tokens', type=int, default=256)
    parser.add_argument('--online-temperature', type=float, default=0.8)
    parser.add_argument('--online-top-p', type=float, default=0.95)
    parser.add_argument('--online-resample-attempts', type=int, default=2)
    parser.add_argument('--enable-ray-verification', action='store_true')
    parser.add_argument('--disable-ray-verification', action='store_true')
    parser.add_argument('--ray-verifier-workers', type=int, default=4)
    parser.add_argument('--num-gpus', type=str, default='auto')
    
    grpo_args = parser.parse_args(args.remaining)

    distributed_args = _build_distributed_args(grpo_args)

    if maybe_launch_grpo_distributed(distributed_args, requested_num_gpus=grpo_args.num_gpus):
        return
    
    train_grpo_from_synthetic_data(
        data_path=grpo_args.data_path,
        output_dir=grpo_args.output_dir,
        model_name=grpo_args.model,
        num_epochs=grpo_args.epochs,
        batch_size=grpo_args.batch_size,
        group_size=grpo_args.group_size,
        online_max_new_tokens=grpo_args.online_max_new_tokens,
        online_temperature=grpo_args.online_temperature,
        online_top_p=grpo_args.online_top_p,
        online_resample_attempts=grpo_args.online_resample_attempts,
        enable_ray_verification=(
            False
            if grpo_args.disable_ray_verification
            else True
        ),
        ray_verifier_workers=grpo_args.ray_verifier_workers,
    )


def cmd_evaluate(args):
    """Evaluate model on benchmarks."""
    from scripts.evaluate import main as evaluate_main
    sys.argv = ['evaluate.py'] + args.remaining
    evaluate_main()


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
    parser.add_argument('command', nargs='?', choices=['generate', 'train', 'train-grpo', 'evaluate', 'pipeline', 'serve'],
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
