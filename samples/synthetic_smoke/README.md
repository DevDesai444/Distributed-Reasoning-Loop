# synthetic_smoke

This directory is **not a real training run**. It is a tiny, CPU-only
smoke artifact set used to verify the v1 provenance envelope, the
quality-score components, and the resumable checkpointer end to end
without invoking the LLM.

Files:

- `dpo_pairs_v1.jsonl` — flat list of v1 pair records.
- `checkpoints/pairs_*.jsonl` — sharded checkpoint files.
- `stats.json` — run-level counters and quality-score percentiles.

To produce real data, see the `Synthetic Data Builder` section of
the top-level README.
