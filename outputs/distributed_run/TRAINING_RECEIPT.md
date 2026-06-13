# Distributed Training Run

This file is a template. The fields below get filled in by an actual multi-GPU
training run; the unfilled template ships with the repo so the receipt shape
is documented and reviewable before the run happens.

| Field | Value |
|-------|-------|
| Date | _YYYY-MM-DD_ |
| GPUs | _N x [model]_ |
| Backend | NCCL |
| Stages | SFT -> DPO -> GRPO |
| Total wall clock | _HH:MM_ |
| Single-GPU baseline throughput | _ samples/sec_ |
| Multi-GPU throughput | _ samples/sec_ |
| Speedup | _ x_ |
| Efficiency | _ %_ |
| Final eval (mean over 3 seeds, GSM8K pass@1) | _ +/- _ |

## How the receipt is produced

1. Run the single-GPU baseline:
   ```bash
   torchrun --nproc-per-node=1 scripts/launch_distributed.py \
       --stage all --dataset gsm8k \
       --model Qwen/Qwen2.5-1.5B-Instruct \
       --output-dir ./outputs/distributed_run/baseline \
       --log-dir ./logs/distributed_run/baseline
   ```
   Record the `samples_per_sec` per stage from
   `outputs/distributed_run/baseline/scaling_summary.json`. Use these as the
   `--single-gpu-throughput` baseline for the multi-GPU run.

2. Run the multi-GPU run with the recorded baseline:
   ```bash
   torchrun --nproc-per-node=$NGPUS scripts/launch_distributed.py \
       --stage all --dataset gsm8k \
       --model Qwen/Qwen2.5-1.5B-Instruct \
       --output-dir ./outputs/distributed_run \
       --log-dir ./logs/distributed_run \
       --single-gpu-throughput <baseline-from-step-1>
   ```

3. Run the seed-replicated eval against the final checkpoint:
   ```bash
   python scripts/eval_replicated.py \
       --model ./outputs/distributed_run/grpo \
       --benchmark gsm8k \
       --seeds 0 1 2 \
       --n-problems 100
   ```

4. Fill in the table above with the numbers, attach the file paths under
   "Log files", and commit.

## Log files

- `logs/distributed_run/rank_*.log`
- `logs/distributed_run/training_steps.jsonl`
- `outputs/distributed_run/scaling_summary.json`
- `outputs/distributed_run/eval_replicated.json`

## Commit

_git ref of the commit that fills in this template_

## Status

**Awaiting hardware.** The launcher, DDP-wrapped trainers, scaling utility,
seed-replicated eval driver, and CPU smoke artifact all live in this repo
and run today. The multi-GPU receipt above is intentionally unfilled: the
project does not have multi-GPU access in the environment that produced this
commit, so claiming numbers here without running them would be dishonest.

When the actual run happens (Kaggle 2xT4, Colab Pro+, or a lab cluster), the
launcher requires no code changes — only the table above does.
