# Agreement Report — `agreement_smoke`

- Model under test: `(synthetic)`
- Benchmark: `gsm8k-synthetic`
- Problems: 5, traces: 5
- Judge model: `stub-judge` (prompt MATH_RUBRIC_V1)

## Pairwise agreement

| Pair | N | Agreement | Kappa | Kappa 95% CI |
|---|---:|---:|---:|---|
| verifier vs judge | 4 | 0.750 | 0.500 | [0.000, 1.000] |

### verifier vs judge — confusion

| | B accept | B reject |
|---|---:|---:|
| A accept | 2 | 1 |
| A reject | 0 | 1 |
(dropped uncertain pairs: 1)

#### Per-difficulty bin — verifier vs judge

| Bin | N | Agreement | Kappa |
|---|---:|---:|---:|
| low | 5 | 0.750 | 0.500 |

## Metadata

```json
{
  "timestamp_utc": "2026-06-12T04:36:40.579386+00:00",
  "note": "pipeline smoke artifact \u2014 not a benchmark claim",
  "judge_backend": "stub"
}
```