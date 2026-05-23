#!/usr/bin/env python3
"""
Head-to-head evaluation of PRM reranking vs outcome reward model reranking.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_gsm8k_problems(num_problems: int):
    from data_generator.dataset_loader import GSM8KLoader

    loader = GSM8KLoader(split="test", subset_size=num_problems)
    return [{"prompt": problem.problem, "answer": problem.answer} for problem in loader.load()]


def evaluate_mode(model_name: str, problems, reranker_type: str, reward_model_path: str | None, process_reward_model_path: str | None, num_samples: int) -> float:
    from evaluation.test_time_compute import TestTimeCompute, TestTimeComputeConfig
    from verifier import GSM8KVerifier

    config = TestTimeComputeConfig(
        num_samples=num_samples,
        use_reward_model=reward_model_path is not None,
        reward_model_path=reward_model_path,
        process_reward_model_path=process_reward_model_path,
        reranker_type=reranker_type,
    )
    ttc = TestTimeCompute(model_name, config, verifier_type="math")
    ttc.setup()
    verifier = GSM8KVerifier()

    solved = 0
    for problem in problems:
        best, _ = ttc.solve(problem["prompt"])
        predicted = best.final_answer or ""
        result = verifier.verify(predicted, problem["answer"])
        if result.status.value == "correct":
            solved += 1

    return solved / len(problems) if problems else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--reward-model-path", type=str, required=True)
    parser.add_argument("--process-reward-model-path", type=str, required=True)
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    problems = load_gsm8k_problems(args.num_problems)
    logger.info("Loaded %s GSM8K test problems", len(problems))

    prm_pass_at_1 = evaluate_mode(
        model_name=args.model,
        problems=problems,
        reranker_type="prm",
        reward_model_path=args.reward_model_path,
        process_reward_model_path=args.process_reward_model_path,
        num_samples=args.num_samples,
    )
    orm_pass_at_1 = evaluate_mode(
        model_name=args.model,
        problems=problems,
        reranker_type="orm",
        reward_model_path=args.reward_model_path,
        process_reward_model_path=args.process_reward_model_path,
        num_samples=args.num_samples,
    )

    print("\nPRM vs ORM Reranking")
    print("====================")
    print(f"{'Mode':<10} {'Pass@1':>8}")
    print("-" * 20)
    print(f"{'PRM':<10} {prm_pass_at_1 * 100:>7.1f}%")
    print(f"{'ORM':<10} {orm_pass_at_1 * 100:>7.1f}%")
    print()


if __name__ == "__main__":
    main()
