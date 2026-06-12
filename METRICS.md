# Metrics

This file tracks the most important benchmark checkpoints for the project and
documents how to interpret evaluation artifacts.

## Current Checkpoints

| Checkpoint | Git Ref | Benchmark | Subset | Correct | Incorrect | Errors | Accuracy | Avg Time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `checkpoint-1` | first tagged eval | GSM8K | 100 | 61 | 39 | 0 | 61.00% | n/a |
| `checkpoint-2` | `68c4bb6` | GSM8K | 100 | 67 | 33 | 0 | 67.00% | 4.59s |
| `checkpoint-3` | `ae25e49` | GSM8K | 100 | 71 | 29 | 0 | 71.00% | 6.83s |

## Evaluation Health Rules

Benchmark JSON files are only safe to quote when they are fully valid.

- `valid_run: true` means the run completed with `errors == 0`
- `status: partial` means some benchmark items failed, so the accuracy is incomplete
- `status: failed` means every item errored, so the artifact is not a real score

In other words, this is valid:

```json
{
  "correct": 71,
  "incorrect": 29,
  "errors": 0,
  "accuracy": 0.71,
  "valid_run": true
}
```

And this is not a benchmark score:

```json
{
  "correct": 0,
  "incorrect": 0,
  "errors": 100,
  "accuracy": 0.0,
  "valid_run": false,
  "status": "failed"
}
```

## Reproducibility

Every evaluation run should now emit:

- benchmark results JSON
- `run_manifest.json`

The manifest captures the timestamp, model, benchmark, TTC settings, git
commit, and git branch so a number can be traced back to the exact repo state
that produced it.

For strict automation, use:

```bash
python main.py evaluate --model ./outputs/grpo_model --benchmark gsm8k --fail-on-errors
```

That makes incomplete runs exit non-zero instead of quietly looking like usable
results.

## Agreement-Run Manifest

`python main.py eval agreement ...` emits an `agreement_results.json` plus a
`run_manifest.json` describing inputs and outputs. The manifest format:

```json
{
  "run_id": "agreement_<UTC_TIMESTAMP>",
  "kind": "agreement",
  "timestamp_utc": "<iso8601>",
  "model": "<policy model id or path>",
  "judge_model": "<judge model id>",
  "judge_prompt_version": "MATH_RUBRIC_V1",
  "benchmark": "gsm8k",
  "split": "test",
  "n_problems": 100,
  "n_traces": 400,
  "rm_threshold": 0.5,
  "artifacts": [
    "agreement_results.json",
    "report.md",
    "confusion_matrices/",
    "shortcuts.jsonl",
    "shortcuts_summary.md",
    "shortcuts_summary.json",
    "per_trace.jsonl"
  ]
}
```

Interpretation rules:

- `agreement_results.json` contains one `PairReport` per evaluator pair
  ((verifier, judge), (verifier, reward_model), (reward_model, judge) when
  the reward model is wired). Each `PairReport` carries overall kappa with a
  95% bootstrap CI, agreement rate, confusion-matrix counts, and per-bin
  metrics for `low` / `mid` / `high` difficulty buckets.
- `shortcuts.jsonl` lists verifier-accepted traces the judge rejected, keyed
  by reason code (`INCOMPLETE_REASONING`, `WRONG_METHOD`, `LUCKY_GUESS`,
  `MISSING_JUSTIFICATION`, `INCONSISTENT_STEPS`, `OTHER`).
- The `judge_prompt_version` field is the contract for longitudinal
  comparison — bump the rubric version in `src/evaluation/judges/prompts.py`
  whenever the wording could shift verdicts, never edit a published version.

The smoke artifact at `samples/agreement_smoke/agreement_smoke.json` is a
pipeline-only sample produced from synthetic problems and a rule-based
stub judge; do not treat it as a benchmark score.

## Robustness-Run Manifest

`python main.py eval robust ...` emits `robustness_results.json`, `report.md`,
and a `run_manifest.json`. The manifest format:

```json
{
  "run_id": "robustness_<UTC_TIMESTAMP>",
  "kind": "robustness",
  "timestamp_utc": "<iso8601>",
  "model": "<policy model id or path>",
  "benchmark": "gsm8k",
  "split": "test",
  "n_problems": 200,
  "n_samples_per_problem": 4,
  "seed": 0,
  "overall_robustness": 0.83,
  "artifacts": [
    "robustness_results.json",
    "report.md"
  ]
}
```

Interpretation rules:

- `robustness_results.json` carries one entry per perturbation in
  `per_perturbation`, each with `n_applicable`, `applicability_rate`,
  `matched_baseline_pass_at_1`, `perturbed_pass_at_1`, `retention`,
  `retention_ci`, and `rewrites_label`.
- `overall_robustness` is the geometric mean of per-perturbation retentions;
  it returns `0.0` if any perturbation's retention is undefined or zero —
  read the per-perturbation table before quoting it.
- Retention is computed on the matched applicable subset, not on all
  problems. Compare retentions across runs only when applicability rates
  are similar.

The smoke artifact at `samples/robustness_smoke/` is a pipeline-only sample
produced with a deterministic stub generator; do not treat it as a benchmark
score.

## Synthetic-Run Manifest

The synthetic data pipeline writes a `stats.json` per run in the output
directory. With the v1 emission path the manifest is:

```json
{
  "schema_version": "1",
  "total_problems_attempted": 7473,
  "total_generations": 89124,
  "verifier_accepts": 44612,
  "verifier_rejects": 44512,
  "verifier_accept_rate": 0.5,
  "pair_candidates": 38400,
  "unique_pairs_after_dedup": 30245,
  "dedup_ratio": 0.788,
  "per_backend": {"vllm": {"generations": 89124, "accepts": 44612}},
  "quality_score_distribution": {"p50": 0.78, "p90": 0.91, "p99": 0.97},
  "completed_at": "<iso8601>"
}
```

Interpretation rules:

- `verifier_accept_rate` is the fraction of generations the verifier
  accepted. Sustained drops (below ~0.3 on GSM8K train) usually mean a
  prompt or sampling regression in the generator.
- `dedup_ratio` is `unique_pairs_after_dedup / pair_candidates`. A
  ratio approaching 1.0 on a long run is suspicious — combinatorial
  expansion at `max_pairs_per_problem=8` should produce some collisions.
- `quality_score_distribution` is computed only over pairs that made it
  past dedup. The components are documented in
  `src/data_generator/quality_score.py`.

Checkpoints under `<output_dir>/checkpoints/pairs_<N>.jsonl` are
append-safe: a crash leaves a valid prefix and resume rebuilds the seen
pair-id set from disk. See `scripts/migrate_pairs_to_v1.py` for the
legacy-to-v1 conversion path.

The smoke artifact at `samples/synthetic_smoke/` is a tiny CPU-only set
exercising the v1 envelope and checkpointer without invoking the LLM; do
not treat it as a real generation run.

## Distributed-Training Receipt

`scripts/launch_distributed.py` runs the SFT -> DPO -> GRPO stack under
`torchrun`. Every run writes:

- `<log-dir>/rank_<RANK>.log` — per-rank init / heartbeat trace.
- `<log-dir>/training_steps.jsonl` — per-step metrics from rank 0.
- `<output-dir>/scaling_summary.json` — aggregated throughput per stage and,
  when a single-GPU baseline is supplied, speedup and efficiency.

The fields in `scaling_summary.json` follow the contract in
`src/orchestration/scaling.py`:

```json
{
  "stages": {
    "sft": {
      "metric": "samples_per_sec",
      "world_size": 4.0,
      "n_steps": 100.0,
      "total_wall_clock_s": 0.0,
      "tokens_per_sec": 0.0,
      "samples_per_sec": 0.0,
      "mean_step_ms": 0.0,
      "p95_step_ms": 0.0,
      "speedup": 0.0,
      "efficiency_pct": 0.0,
      "ideal_speedup": 4.0,
      "single_gpu_throughput": 0.0,
      "multi_gpu_throughput": 0.0
    }
  },
  "world_size": 4,
  "backend": "nccl",
  "metric": "samples_per_sec",
  "smoke": false,
  "single_gpu_throughput": 0.0
}
```

Interpretation rules:

- `speedup = multi_gpu_throughput / single_gpu_throughput`. Both terms must be
  in the same unit; the `metric` field carries which one.
- `efficiency_pct = 100 * speedup / world_size`. Linear scaling is 100%.
- `smoke: true` runs are CPU smoke artifacts, not benchmarks. The receipt
  template explicitly distinguishes them.

The filled-in human receipt lives at
`outputs/distributed_run/TRAINING_RECEIPT.md`. The template is committed
unfilled; an actual multi-GPU run fills the table and commits.

The Ray scaling number quoted as `3.25x speedup, 81.2% efficiency` in
`throughput_results.json` is **verification-stage** scaling produced by the
`RayMathVerificationPool` in `src/training/grpo_trainer.py`. It is not a
training-stage receipt and should not be conflated with one — the training
receipt above tracks training-stage scaling separately.

The CPU smoke artifact at `samples/distributed_smoke/` is produced by
`scripts/distributed_smoke.sh` running the launcher with `--smoke`. It
exercises the JSONL writer, the per-rank logger, and the scaling math on a
tiny 32x32 matmul; it is not a benchmark and the numbers there speak only to
the harness, not to training throughput.
