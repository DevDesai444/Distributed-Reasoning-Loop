# Distributed Reasoning Loop

**Author:** Dev Desai  
**Focus:** verifiable reasoning, synthetic supervision, post-training, test-time compute, and distributed ML systems

Distributed Reasoning Loop is an end-to-end research engineering system for improving reasoning models with objective feedback. It generates multiple candidate solutions, verifies them with math or code checkers, converts correctness into training data, fine-tunes the policy, and evaluates the result with benchmark and systems metrics.

The project is built around a simple thesis:

> In domains where answers can be checked automatically, correctness can become scalable supervision.

## Results Snapshot

| Benchmark | Task Type | Result |
|---|---|---:|
| GSM8K | grade-school math | **71.0% accuracy** |
| MATH | competition math | **36.0% accuracy** |
| HumanEval | code generation | **42.0% pass@1** |
| MBPP | Python programming problems | **57.0% pass@1** |
| GSM8K pass@k | math reasoning with multiple samples | **35.0% pass@1 -> 70.0% pass@8** |
| GSM8K fine-tuning comparison | base vs GRPO-tuned model | **+20.0 pass@1 points** |

### Fine-Tuning Lift

From `comparison_results.json`:

| Metric | Base Qwen2.5-1.5B-Instruct | Fine-Tuned `./outputs/grpo_model` | Gain |
|---|---:|---:|---:|
| Pass@1 | 35.0% | **55.0%** | **+20.0 pts** |
| Pass@4 | 65.0% | **75.0%** | **+10.0 pts** |
| Pass@8 | 80.0% | **90.0%** | **+10.0 pts** |
| Correct@1 | 7 / 20 | **11 / 20** | +4 |

### Systems Benchmarks

From `throughput_results.json`:

| System Metric | Result |
|---|---:|
| End-to-end pipeline throughput | **49.09 samples/sec** |
| Prefix-cache throughput | **49.50 samples/sec** |
| No-prefix-cache throughput | 19.89 samples/sec |
| Prefix-cache speedup | **~2.5x** |
| Math verifier throughput | **35,526 verifications/sec** |
| Ray scaling, 4 workers | **3.25x speedup**, 81.2% efficiency |

## What It Does

Most reasoning projects focus on one layer: prompting, fine-tuning, evaluation, or infrastructure. This repository connects the full loop:

1. Load benchmark problems from GSM8K, MATH, HumanEval, or MBPP.
2. Generate multiple reasoning paths with `transformers`, `vLLM`, or `SGLang`.
3. Verify outputs with symbolic math checks or Docker-isolated code execution.
4. Build synthetic artifacts such as `correct_samples.jsonl` and `dpo_pairs.jsonl`.
5. Train with SFT, DPO, GRPO, outcome reward models, or process reward models.
6. Evaluate with pass@k, consensus-style test-time compute, and benchmark manifests.
7. Scale expensive stages with optional Ray workers, Kafka streaming, and prefix caching.

## Architecture

```mermaid
flowchart LR
    A["Benchmark Loaders<br/>GSM8K / MATH / HumanEval / MBPP"] --> B["Reasoning Generation<br/>Transformers / vLLM / SGLang"]
    B --> C["Verification<br/>Math equivalence / Docker code execution"]
    C --> D["Synthetic Data Builder<br/>filter / dedup / pair / checkpoint"]
    D --> E["Training Stack<br/>SFT / DPO / GRPO / ORM / PRM"]
    E --> F["Evaluation<br/>pass@k / TTC / manifests"]
    E --> G["Inference Improvements<br/>reranking / self-consistency / cache reuse"]
    B --> H["Systems Layer<br/>Ray / Kafka / KV cache"]
    H --> C
    H --> D
```

The core design is deliberately modular: generation, verification, training, and evaluation can be run independently, but the strongest path is the closed loop.

```mermaid
sequenceDiagram
    participant L as Loader
    participant G as Generator
    participant V as Verifier
    participant P as Pair Builder
    participant T as Trainer
    participant E as Evaluator

    L->>G: normalized benchmark problems
    G->>V: k candidate reasoning paths
    V->>P: correctness labels + confidence
    P->>T: SFT samples + preference pairs
    T->>E: trained adapter / policy checkpoint
    E->>G: benchmark metrics and failure modes
```

## Core Components

### Dataset Layer

`src/data_generator/dataset_loader.py` normalizes benchmark examples into a common problem interface. Supported sources include:

- `GSM8KLoader`
- `MATHLoader`
- `HumanEvalLoader`
- `MBPPLoader`

This lets math and code tasks share the same generation/training pipeline while preserving domain-specific metadata such as expected answers, tests, and entry points.

### Generation Layer

`src/data_generator/cot_generator.py` produces multiple candidate trajectories per prompt. It supports:

- Hugging Face `transformers` for local generation
- `vLLM` for high-throughput batched generation
- `SGLang` for structured generation and prefix-aware inference

Each sample is stored with problem metadata, final-answer extraction, correctness status, and stable hashes for reproducible deduplication.

### Verification Layer

Verification is the main differentiator.

- `src/verifier/math_verifier.py` checks numeric and symbolic equivalence for math answers.
- `src/verifier/execution_verifier.py` and `src/verifier/code_verifier.py` execute generated code in constrained Docker sandboxes.
- GSM8K and HumanEval have specialized verifier paths for their answer/test formats.

This turns model outputs into objective labels instead of relying on manual preference annotation.

### Synthetic Data Builder

`src/data_generator/synthetic_data_pipeline.py` converts raw generations into reusable artifacts:

- `all_samples.jsonl`
- `filtered_samples.jsonl`
- `correct_samples.jsonl`
- `incorrect_samples.jsonl`
- `dpo_pairs.jsonl`
- `full_pairs.jsonl`
- `stats.json`

The preprocessing step filters repetitive traces, removes near duplicates, prioritizes high-confidence positives, and builds diverse correct/incorrect preference pairs.

### Training Stack

The training code lives in `src/training/`.

| Method | Purpose |
|---|---|
| SFT | Warm-start on verified correct reasoning traces |
| DPO | Learn from verifier-derived chosen/rejected pairs |
| GRPO | Optimize grouped candidate outcomes with relative rewards |
| Outcome Reward Model | Score full responses for reranking |
| Process Reward Model | Score intermediate reasoning steps |

The strongest built-in pipeline is:

```text
SFT on correct traces -> DPO on preference pairs -> GRPO refinement -> benchmark evaluation
```

Recent Kaggle/T4 support includes explicit `4bit` quantized LoRA loading and CUDA memory guards so the full `best` pipeline can run without skipping SFT.

### Evaluation and Test-Time Compute

`src/evaluation/` implements:

- GSM8K, MATH, and HumanEval evaluators
- pass@k evaluation
- majority vote and consensus-style selection
- best-of-n sampling
- optional reward-model reranking
- run manifests with git commit, branch, model path, validity status, and error counts

The project treats benchmark health seriously: `valid_run: true` means the run completed with `errors == 0`; partial or failed runs are not quoted as scores.

### Distributed Systems Layer

`src/orchestration/` includes:

- Ray workers for verification and tokenization parallelism
- Kafka streaming hooks for decoupled pipeline stages
- KV-cache management for serving-oriented inference experiments

This matters because verified reasoning gets expensive quickly: every prompt may create many candidate paths, and every path needs verification, filtering, and possibly training-time reuse.

## Repository Map

```text
.
├── config/                  # Model, training, verifier, and orchestration config
├── docs/                    # Kaggle and workflow notes
├── scripts/                 # Pipeline, evaluation, training, and benchmark CLIs
├── src/
│   ├── data_generator/      # Dataset loaders and synthetic data pipeline
│   ├── evaluation/          # Benchmark and test-time-compute logic
│   ├── inference/           # vLLM, SGLang, speculative decoding
│   ├── orchestration/       # Ray, Kafka, KV-cache utilities
│   ├── training/            # SFT, DPO, GRPO, ORM, PRM
│   └── verifier/            # Math and code correctness checkers
├── tests/                   # Unit and integration coverage
├── METRICS.md               # Valid benchmark checkpoint tracking
├── comparison_results.json  # Base vs fine-tuned comparison
├── pass_at_k_results.json   # pass@k artifact
└── throughput_results.json  # Systems benchmark artifact
```

## How To Run

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Generate synthetic data:

```bash
python main.py generate \
  --dataset gsm8k \
  --num-paths 8 \
  --subset-size 600 \
  --backend vllm \
  --output-dir ./outputs/synthetic_data
```

Run the full training loop:

```bash
python main.py pipeline \
  --dataset gsm8k \
  --subset-size 600 \
  --training-method best
```

Evaluate:

```bash
python main.py evaluate \
  --model ./outputs/grpo_model \
  --benchmark gsm8k \
  --fail-on-errors
```

Use test-time compute:

```bash
python main.py evaluate \
  --model ./outputs/grpo_model \
  --benchmark gsm8k \
  --use-ttc \
  --ttc-samples 16 \
  --fail-on-errors
```

### Three-way agreement

`python main.py eval agreement ...` runs the same set of traces through the
math/code verifier, the learned reward model, and an open-weights LLM judge
(default Qwen2.5-7B-Instruct), then writes Cohen's kappa with 95% bootstrap
CIs, confusion matrices per pair, and a shortcut bucket of verifier-accepted
traces the judge flagged as bad reasoning. Difficulty bins for GSM8K use the
`<<step>>` count in the gold solution.

```bash
python main.py eval agreement \
  --model ./outputs/grpo_model \
  --benchmark gsm8k \
  --n-problems 100 \
  --n-samples 4 \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --reward-model ./outputs/reward_model \
  --output-dir ./outputs/agreement_run
```

A pipeline-smoke artifact (`samples/agreement_smoke/agreement_smoke.json`,
produced by `scripts/agreement_smoke.py`) shows the format. It is **not** a
benchmark claim — it runs against five synthetic problems with a rule-based
fake judge so the harness works on machines with no GPU and no network.

### Robustness (perturbation retention)

`python main.py eval robust ...` measures how pass@1 holds up under six
deterministic GSM8K perturbations:

| Perturbation | Label | What it does |
|---|---|---|
| number_swap | rewrites gold | Swap one operand in the final multiplicative step; scale gold accordingly |
| unit_change | rewrites gold | Rewrite a recognised unit (dollars↔cents, hours↔minutes, kg↔g); scale gold |
| irrelevant_context | preserves | Prepend an unrelated sentence |
| distractor_sentence | preserves | Insert a sentence with a plausible-looking but irrelevant number |
| paraphrase | preserves | Rule-based clause/word rewrites (no LLM) |
| reordering | preserves | Swap the first two sentences when there is no anaphora |

For each perturbation the runner reports applicability rate, per-perturbation
pass@1 retention with a 95% bootstrap CI on the same subset of problems
(controlling for difficulty), and an overall robustness score equal to the
geometric mean of per-perturbation retentions.

```bash
python main.py eval robust \
  --model ./outputs/grpo_model \
  --benchmark gsm8k \
  --n-problems 200 \
  --n-samples 4 \
  --seed 0 \
  --output-dir ./outputs/robustness_run
```

A pipeline-smoke artifact (`samples/robustness_smoke/`, produced by
`scripts/robustness_smoke.py`) exercises the same path with hand-written
predictions against six inline problems. It is **not** a benchmark claim.

Run on Kaggle T4:

- Use `drlll2_training_ready.ipynb`.
- Keep `--training-method best`.
- Do not add `--skip-sft`.
- Put precomputed synthetic data at `outputs/synthetic_data/dpo_pairs.jsonl` and `outputs/synthetic_data/correct_samples.jsonl` if reusing a previous generation run.

## Why This Is Portfolio-Relevant

This project demonstrates more than model fine-tuning. It shows the ability to build an ML system across the full research-to-infrastructure stack:

- objective verification instead of subjective labeling
- synthetic data construction with quality filters
- multiple post-training algorithms
- benchmark validity checks and reproducibility manifests
- GPU-aware training paths for constrained runtimes
- distributed pipeline components for scale
- honest measurement of both model quality and systems throughput

The result is a project that a reviewer can understand as a complete engineering system, not just a notebook experiment.

## Roadmap

1. Scale all four benchmark runs to larger held-out slices.
2. Compare verifier reranking, outcome reward reranking, and process reward reranking on the same evaluation sets.
3. Add richer error taxonomy dashboards for failed reasoning paths.
4. Expand Kaggle and local reproducibility docs with cost/runtime estimates.
5. Extend the loop to harder symbolic, tool-use, and multi-step coding tasks.

## Attribution

Created by **Dev Desai** as a research-engineering project on scalable supervision for reasoning models.
