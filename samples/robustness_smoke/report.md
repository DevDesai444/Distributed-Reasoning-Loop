# Robustness report — gsm8k-synthetic

- run id: `robustness_smoke`
- model: `smoke-stub`
- problems: 6, samples per problem: 1
- seed: 0
- overall robustness (geometric mean of retentions): **0.000**

| perturbation | applicable | applic.% | baseline p@1 | perturbed p@1 | retention | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| number_swap | 4/6 | 66.7% | 1.000 | 0.000 | 0.000 | [0.000, 0.000] |
| unit_change | 3/6 | 50.0% | 1.000 | 0.000 | 0.000 | [0.000, 0.000] |
| irrelevant_context | 6/6 | 100.0% | 1.000 | 1.000 | 1.000 | [1.000, 1.000] |
| distractor_sentence | 6/6 | 100.0% | 1.000 | 1.000 | 1.000 | [1.000, 1.000] |
| paraphrase | 2/6 | 33.3% | 1.000 | 0.500 | 0.500 | [0.000, 1.000] |
| reordering | 4/6 | 66.7% | 1.000 | 1.000 | 1.000 | [1.000, 1.000] |
