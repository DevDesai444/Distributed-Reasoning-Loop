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
