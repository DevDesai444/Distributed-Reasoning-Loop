# Distributed launcher smoke artifact

This directory is the receipt that `scripts/launch_distributed.py` runs end to
end on a CPU-only machine. It is **not a benchmark**. The numbers under
`outputs/scaling_summary.json` come from a 5-step loop over a 32x32 matmul, so
nothing here speaks to model quality or real training throughput.

Layout:

- `outputs/scaling_summary.json` — aggregated per-stage throughput, world size,
  and (when a baseline is supplied) speedup / efficiency.
- `logs/rank_0.log` — per-rank init / heartbeat trace. On a multi-rank run there
  is one of these per rank.
- `logs/training_steps.jsonl` — per-step record emitted from rank 0.

The point of this artifact is to keep the launcher harness honest: if the
JSONL shape changes, this file changes, and the receipt template under
`outputs/distributed_run/TRAINING_RECEIPT.md` has to be updated to match.
