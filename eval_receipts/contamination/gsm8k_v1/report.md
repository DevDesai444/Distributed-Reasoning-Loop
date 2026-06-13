# Contamination Report

- benchmark: `gsm8k`
- data source: `loader`
- n-gram size: `13`
- high-overlap threshold: `>= 3 matching n-grams`
- generated: `2026-06-13T01:22:07.518992+00:00`

## Summary

| Direction | Haystack size | Needle size | Any overlap | High overlap |
|---|---:|---:|---:|---:|
| gsm8k/test in gsm8k/train | 7473 | 1319 | 3 (0.23%) | 3 (0.23%) |
| gsm8k/train in gsm8k/test | 1319 | 7473 | 4 (0.05%) | 4 (0.05%) |
| gsm8k/test in synthetic_pairs:dpo_pairs.jsonl | 1688 | 1319 | 0 (0.00%) | 0 (0.00%) |

## Top matches

### gsm8k/test in gsm8k/train

- `gsm8k_test_632` — 13 matching n-grams
  - text: Max bought stamps at the post office. Some of the stamps had a snowflake design, some had a truck design, and some had a rose design. Max bought 16 snowflake stamps. He bought 3 more truck stamps than snowflake stamps, a...
  - example n-gram: `bought stamps at the post office some of the stamps had a snowflake`
- `gsm8k_test_602` — 7 matching n-grams
  - text: A plane travels 1200 miles in 3 hours. At the same rate, how many additional hours would it take to travel an additional 2000 miles?
  - example n-gram: `miles in 3 hours at the same rate how many additional hours would`
- `gsm8k_test_581` — 3 matching n-grams
  - text: Max plans to watch two movies this weekend. The first movie is 1 hour and 30 minutes long while the second movie is 2 hours and 5 minutes long. How many minutes will it take Max to watch the two movies?
  - example n-gram: `the first movie is 1 hour and 30 minutes long while the second`

### gsm8k/train in gsm8k/test

- `gsm8k_train_20` — 13 matching n-grams
  - text: Bella bought stamps at the post office. Some of the stamps had a snowflake design, some had a truck design, and some had a rose design. Bella bought 11 snowflake stamps. She bought 9 more truck stamps than snowflake stam...
  - example n-gram: `bought stamps at the post office some of the stamps had a snowflake`
- `gsm8k_train_1314` — 7 matching n-grams
  - text: A train travels 270 miles in 3 hours. At the same rate, how many additional hours would it take to travel an additional 180 miles?
  - example n-gram: `miles in 3 hours at the same rate how many additional hours would`
- `gsm8k_train_5162` — 7 matching n-grams
  - text: A train travels 360 miles in 3 hours. At the same rate, how many additional hours would it take to travel an additional 240 miles?
  - example n-gram: `miles in 3 hours at the same rate how many additional hours would`
- `gsm8k_train_406` — 3 matching n-grams
  - text: Joseph and his friends watched two movies in his house. The first movie is 1 hour and 30 minutes long while the second movie is 30 minutes longer than the first. Before the movies, they spent 10 minutes making popcorn an...
  - example n-gram: `the first movie is 1 hour and 30 minutes long while the second`

### gsm8k/test in synthetic_pairs:dpo_pairs.jsonl

_no overlapping needles_
