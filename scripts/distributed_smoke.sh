#!/usr/bin/env bash
# CPU smoke test for the distributed launcher.
#
# Backend is Gloo because NCCL requires CUDA and this script is meant to run on
# any machine — laptops, CI workers, anything with PyTorch and `torchrun`. The
# point is to prove the launcher, per-rank log format, scaling utility, and
# JSONL step writer all work end to end. It is NOT a benchmark and does not
# train a model.
#
# Output goes under samples/distributed_smoke/, which lives under version
# control as the receipt that the harness ran cleanly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE_DIR="${REPO_ROOT}/samples/distributed_smoke"
# Use `run_outputs/` rather than `outputs/` so the artifact is not swallowed
# by the top-level outputs/ gitignore rule.
OUTPUT_DIR="${SAMPLE_DIR}/run_outputs"
LOG_DIR="${SAMPLE_DIR}/logs"

rm -rf "${SAMPLE_DIR}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# world_size=1 keeps the smoke runnable everywhere. The launcher still uses the
# same code path as a multi-rank run; init_distributed sees WORLD_SIZE=1 and
# skips the process-group init but everything else (per-rank log, JSONL writer,
# scaling math) exercises the real path.
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/launch_distributed.py" \
    --stage sft \
    --dataset gsm8k \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir "${OUTPUT_DIR}" \
    --log-dir "${LOG_DIR}" \
    --backend gloo \
    --smoke \
    --n-steps 5 \
    --single-gpu-throughput 1.0 \
    --metric samples_per_sec

cat > "${SAMPLE_DIR}/README.md" <<'EOF'
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
EOF

echo
echo "Smoke artifact written under ${SAMPLE_DIR}"
echo "Scaling summary: ${OUTPUT_DIR}/scaling_summary.json"
echo "Step log:        ${LOG_DIR}/training_steps.jsonl"
echo "Rank log:        ${LOG_DIR}/rank_0.log"
