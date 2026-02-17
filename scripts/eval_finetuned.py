#!/usr/bin/env python3
"""
Evaluate a base model against a GRPO fine-tuned checkpoint on held-out GSM8K.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_torch():
    import torch
    return torch


@dataclass
class ModelMetrics:
    model_name: str
    pass_at_1: float
    pass_at_4: float
    pass_at_8: float
    mean_reward: float
    avg_response_length: float
    avg_reasoning_steps: float
    inference_time_per_problem: float
    total_problems: int


@dataclass
class ComparisonResults:
    base: ModelMetrics
    finetuned: ModelMetrics
    improvements: Dict[str, float]
    held_out_problems: int
    timestamp: str


class ModelEvaluator:
    """Evaluate a model by sampling multiple solutions per GSM8K problem."""

    def __init__(self, model_path: str, batch_size: int = 4):
        self.model_path = model_path
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.verifier = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from verifier import GSM8KVerifier

        torch = get_torch()
        device_map = "auto" if torch.cuda.is_available() else None
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        self.verifier = GSM8KVerifier()
        logger.info("Loaded model: %s", self.model_path)

    def generate(
        self,
        prompt: str,
        n_samples: int,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        batch_prompts = [prompt_text] * n_samples
        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=2048,
        )

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        torch = get_torch()
        responses: List[str] = []
        do_sample = temperature > 0

        with torch.no_grad():
            for offset in range(0, n_samples, self.batch_size):
                sub_batch = {key: value[offset : offset + self.batch_size] for key, value in inputs.items()}
                generation_kwargs = dict(
                    **sub_batch,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if do_sample:
                    generation_kwargs["temperature"] = temperature
                    generation_kwargs["top_p"] = 0.95
                outputs = self.model.generate(**generation_kwargs)
                prompt_len = sub_batch["input_ids"].shape[1]
                decoded = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
                responses.extend(decoded)

        return responses

    @staticmethod
    def count_reasoning_steps(response: str) -> int:
        return max(len([line for line in response.splitlines() if line.strip()]), 1)

    def reward_response(self, response: str, answer: str) -> float:
        result = self.verifier.verify_reasoning_path(response, answer)
        return 1.0 if result.status.value == "correct" else 0.0

    def evaluate(
        self,
        problems: List[Dict],
        k_values: List[int] | None = None,
        temperature: float = 0.7,
    ) -> ModelMetrics:
        k_values = k_values or [1, 4, 8]
        max_k = max(k_values)

        all_correctness: List[List[bool]] = []
        all_rewards: List[float] = []
        all_lengths: List[int] = []
        all_steps: List[int] = []
        total_time = 0.0

        for problem in tqdm(problems, desc=f"Evaluating {Path(self.model_path).name}"):
            prompt = problem["prompt"]
            answer = problem["answer"]

            start_time = time.time()
            responses = self.generate(prompt, n_samples=max_k, temperature=temperature)
            total_time += time.time() - start_time

            rewards = [self.reward_response(response, answer) for response in responses]
            correctness = [reward >= 1.0 for reward in rewards]
            all_correctness.append(correctness)
            all_rewards.extend(rewards)
            all_lengths.extend(len(response) for response in responses)
            all_steps.extend(self.count_reasoning_steps(response) for response in responses)

        pass_at_k = {}
        for k in k_values:
            solved = sum(1 for correctness in all_correctness if any(correctness[:k]))
            pass_at_k[k] = solved / len(problems) if problems else 0.0

        return ModelMetrics(
            model_name=self.model_path,
            pass_at_1=pass_at_k.get(1, 0.0),
            pass_at_4=pass_at_k.get(4, 0.0),
            pass_at_8=pass_at_k.get(8, 0.0),
            mean_reward=sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            avg_response_length=sum(all_lengths) / len(all_lengths) if all_lengths else 0.0,
            avg_reasoning_steps=sum(all_steps) / len(all_steps) if all_steps else 0.0,
            inference_time_per_problem=total_time / len(problems) if problems else 0.0,
            total_problems=len(problems),
        )


def load_held_out_problems(num_problems: int = 50) -> List[Dict]:
    from data_generator.dataset_loader import GSM8KLoader

    loader = GSM8KLoader(split="test", subset_size=num_problems)
    problems = loader.load()
    return [{"prompt": problem.problem, "answer": problem.answer} for problem in problems]


def format_row(name: str, metrics: ModelMetrics) -> str:
    return (
        f"{name:<12} "
        f"{metrics.pass_at_1 * 100:>8.1f}% "
        f"{metrics.pass_at_4 * 100:>8.1f}% "
        f"{metrics.pass_at_8 * 100:>8.1f}% "
        f"{metrics.mean_reward:>10.3f} "
        f"{metrics.avg_response_length:>11.1f}"
    )


def compare_models(
    base_model: str,
    finetuned_model: str,
    num_problems: int = 50,
    output_path: str = "./outputs/training_improvement.json",
    temperature: float = 0.7,
) -> ComparisonResults:
    problems = load_held_out_problems(num_problems)

    logger.info("Evaluating base model on %s held-out GSM8K problems", len(problems))
    base_evaluator = ModelEvaluator(base_model)
    base_evaluator.load()
    base_metrics = base_evaluator.evaluate(problems, temperature=temperature)

    torch = get_torch()
    del base_evaluator.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Evaluating fine-tuned model on %s held-out GSM8K problems", len(problems))
    finetuned_evaluator = ModelEvaluator(finetuned_model)
    finetuned_evaluator.load()
    finetuned_metrics = finetuned_evaluator.evaluate(problems, temperature=temperature)

    from datetime import datetime

    improvements = {
        "pass_at_1": (finetuned_metrics.pass_at_1 - base_metrics.pass_at_1) * 100,
        "pass_at_4": (finetuned_metrics.pass_at_4 - base_metrics.pass_at_4) * 100,
        "pass_at_8": (finetuned_metrics.pass_at_8 - base_metrics.pass_at_8) * 100,
        "mean_reward": finetuned_metrics.mean_reward - base_metrics.mean_reward,
        "avg_response_length": finetuned_metrics.avg_response_length - base_metrics.avg_response_length,
    }

    results = ComparisonResults(
        base=base_metrics,
        finetuned=finetuned_metrics,
        improvements=improvements,
        held_out_problems=len(problems),
        timestamp=datetime.now().isoformat(),
    )

    output = {
        "base": asdict(base_metrics),
        "finetuned": asdict(finetuned_metrics),
        "improvements": improvements,
        "held_out_problems": len(problems),
        "timestamp": results.timestamp,
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as handle:
        json.dump(output, handle, indent=2)

    return results


def print_comparison_report(results: ComparisonResults):
    print("\n" + "=" * 72)
    print("TRAINING IMPROVEMENT REPORT")
    print("=" * 72)
    print(f"{'Model':<12} {'Pass@1':>8} {'Pass@4':>8} {'Pass@8':>8} {'MeanReward':>10} {'AvgLen':>11}")
    print("-" * 72)
    print(format_row("Base", results.base))
    print(format_row("GRPO", results.finetuned))
    print("-" * 72)
    print(
        f"{'Delta':<12} "
        f"{results.improvements['pass_at_1']:>7.1f}% "
        f"{results.improvements['pass_at_4']:>7.1f}% "
        f"{results.improvements['pass_at_8']:>7.1f}% "
        f"{results.improvements['mean_reward']:>10.3f} "
        f"{results.improvements['avg_response_length']:>11.1f}"
    )
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--finetuned-model", type=str, default="./outputs/grpo_model")
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--output", type=str, default="./outputs/training_improvement.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    results = compare_models(
        base_model=args.base_model,
        finetuned_model=args.finetuned_model,
        num_problems=args.num_problems,
        output_path=args.output,
        temperature=args.temperature,
    )
    print_comparison_report(results)


if __name__ == "__main__":
    main()
