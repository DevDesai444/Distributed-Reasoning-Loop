"""
Data Preprocessor for reasoning data.
Handles deduplication, quality filtering, length filtering, and smart pair selection.
"""

import re
import logging
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random
import math

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""
    # Length filtering
    min_reasoning_tokens: int = 50
    max_reasoning_tokens: int = 2048
    min_response_length: int = 100  # characters
    max_response_length: int = 8000  # characters
    
    # Quality filtering
    min_step_count: int = 2  # Minimum reasoning steps
    require_final_answer: bool = True
    filter_repetitive: bool = True
    max_repetition_ratio: float = 0.3
    
    # Deduplication
    dedup_threshold: float = 0.85  # Jaccard similarity threshold
    use_semantic_dedup: bool = False
    
    # Pair selection
    min_pair_diversity: float = 0.2
    max_pairs_per_problem: int = 5
    prefer_high_confidence: bool = True
    require_answer_disagreement_for_pairs: bool = True
    prioritize_step_aligned_pairs: bool = True
    max_pairs_per_error_type: int = 2
    min_chosen_confidence: float = 0.5
    min_negative_step_ratio: float = 0.5
    min_pair_quality_score: float = 0.05
    prefer_near_miss_negatives: bool = True
    
    # Normalization
    normalize_whitespace: bool = True
    normalize_math: bool = True
    strip_system_artifacts: bool = True


class DataPreprocessor:
    """
    Preprocesses reasoning data for RL training.
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.stats = defaultdict(int)
        self._numeric_pattern = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")
    
    def preprocess(
        self,
        samples: List[Dict[str, Any]],
        create_pairs: bool = True,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Full preprocessing pipeline.
        
        Returns:
            (filtered_samples, dpo_pairs)
        """
        logger.info(f"Starting preprocessing of {len(samples)} samples")
        self.stats = defaultdict(int)
        self.stats["input_samples"] = len(samples)
        
        # Step 1: Normalize
        samples = [self._normalize(s) for s in samples]
        
        # Step 2: Quality filter
        samples = self._quality_filter(samples)
        logger.info(f"After quality filter: {len(samples)}")
        
        # Step 3: Length filter
        samples = self._length_filter(samples)
        logger.info(f"After length filter: {len(samples)}")
        
        # Step 4: Deduplicate
        samples = self._deduplicate(samples)
        logger.info(f"After deduplication: {len(samples)}")
        
        self.stats["output_samples"] = len(samples)
        
        # Step 5: Create pairs
        pairs = []
        if create_pairs:
            pairs = self._create_smart_pairs(samples)
            self.stats["pairs_created"] = len(pairs)
            logger.info(f"Created {len(pairs)} DPO pairs")
        
        logger.info(f"Preprocessing stats: {dict(self.stats)}")
        return samples, pairs
    
    def _normalize(self, sample: Dict) -> Dict:
        """Normalize a single sample."""
        reasoning = sample.get("reasoning", sample.get("response", ""))
        
        if self.config.normalize_whitespace:
            # Collapse multiple newlines
            reasoning = re.sub(r'\n{3,}', '\n\n', reasoning)
            # Collapse multiple spaces
            reasoning = re.sub(r' {2,}', ' ', reasoning)
            reasoning = reasoning.strip()
        
        if self.config.normalize_math:
            # Normalize common math formatting
            reasoning = re.sub(r'\$\$\s*', '$$', reasoning)
            reasoning = re.sub(r'\s*\$\$', '$$', reasoning)
        
        if self.config.strip_system_artifacts:
            # Remove common artifacts
            reasoning = re.sub(r'^(Assistant:|AI:|Response:)\s*', '', reasoning, flags=re.IGNORECASE)
            reasoning = re.sub(r'<\|.*?\|>', '', reasoning)  # Remove special tokens
            reasoning = re.sub(r'\[INST\].*?\[/INST\]', '', reasoning, flags=re.DOTALL)
        
        sample["reasoning"] = reasoning
        sample["reasoning_step_count"] = self._estimate_step_count(reasoning)
        sample["has_final_answer"] = self._has_final_answer(reasoning)
        sample["answer_key"] = self._extract_answer_key(
            sample.get("final_answer")
            or sample.get("predicted")
            or reasoning
        )
        sample["expected_answer_key"] = self._extract_answer_key(sample.get("expected_answer", ""))
        return sample

    def _estimate_step_count(self, reasoning: str) -> int:
        step_patterns = [
            r'step\s*\d+',
            r'\d+\)',
            r'\d+\.',
            r'first|second|third|then|next|finally|therefore',
            r'let\'s|we can|we need|we have',
        ]
        return sum(
            len(re.findall(pattern, reasoning, re.IGNORECASE))
            for pattern in step_patterns
        )

    def _has_final_answer(self, reasoning: str) -> bool:
        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)',
            r'(?:therefore|thus|so|hence)',
            r'=\s*[\d.]+\s*$',
            r'\\boxed\{',
            r'####',
        ]
        return any(
            re.search(pattern, reasoning, re.IGNORECASE | re.MULTILINE)
            for pattern in answer_patterns
        )

    def _extract_answer_key(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        matches = self._numeric_pattern.findall(text.replace(",", ""))
        if matches:
            try:
                numeric = matches[-1]
                if "/" in numeric:
                    numerator, denominator = numeric.split("/", 1)
                    return f"{float(numerator) / float(denominator):.8f}".rstrip("0").rstrip(".")
                return f"{float(numeric):.8f}".rstrip("0").rstrip(".")
            except Exception:
                return matches[-1]
        normalized = re.sub(r"\s+", " ", text.lower()).strip(" .,!?:;`'\"")
        return normalized or None

    def _classify_failure_mode(self, sample: Dict[str, Any]) -> str:
        reasoning = sample.get("reasoning", "")
        answer_key = sample.get("answer_key")
        expected_key = sample.get("expected_answer_key")

        if not reasoning or len(reasoning.split()) < max(8, self.config.min_reasoning_tokens // 4):
            return "trivial_short"
        if not sample.get("has_final_answer", False):
            return "missing_final_answer"
        if self._is_repetitive(reasoning):
            return "repetitive"
        if answer_key and expected_key and answer_key != expected_key:
            return "answer_mismatch"
        if answer_key is None:
            return "parse_like_failure"
        return "reasoning_gap"
    
    def _quality_filter(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples based on quality criteria."""
        filtered = []
        
        for sample in samples:
            reasoning = sample.get("reasoning", "")
            
            # Check minimum steps
            if self.config.min_step_count > 0:
                step_count = sample.get("reasoning_step_count", self._estimate_step_count(reasoning))
                if step_count < self.config.min_step_count:
                    self.stats["filtered_no_steps"] += 1
                    continue
            
            # Check for final answer
            if self.config.require_final_answer:
                has_answer = sample.get("has_final_answer", self._has_final_answer(reasoning))
                if not has_answer:
                    self.stats["filtered_no_answer"] += 1
                    continue
            
            # Check for repetition
            if self.config.filter_repetitive:
                if self._is_repetitive(reasoning):
                    self.stats["filtered_repetitive"] += 1
                    continue
            
            filtered.append(sample)
        
        return filtered
    
    def _is_repetitive(self, text: str) -> bool:
        """Check if text has too much repetition."""
        sentences = re.split(r'[.!?\n]', text)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 3:
            return False
        
        # Check for duplicate sentences
        unique = set(sentences)
        if len(unique) / len(sentences) < (1 - self.config.max_repetition_ratio):
            return True
        
        # Check for repeated phrases
        words = text.lower().split()
        if len(words) < 20:
            return False
        
        # N-gram repetition check
        ngram_size = 5
        ngrams = [' '.join(words[i:i+ngram_size]) for i in range(len(words) - ngram_size)]
        unique_ngrams = set(ngrams)
        
        if len(ngrams) > 0 and len(unique_ngrams) / len(ngrams) < 0.5:
            return True
        
        return False
    
    def _length_filter(self, samples: List[Dict]) -> List[Dict]:
        """Filter by length."""
        filtered = []
        
        for sample in samples:
            reasoning = sample.get("reasoning", "")
            
            # Character length
            if len(reasoning) < self.config.min_response_length:
                self.stats["filtered_too_short"] += 1
                continue
            
            if len(reasoning) > self.config.max_response_length:
                self.stats["filtered_too_long"] += 1
                continue
            
            # Token count (approximate)
            token_count = len(reasoning.split())
            if token_count < self.config.min_reasoning_tokens:
                self.stats["filtered_too_few_tokens"] += 1
                continue
            
            if token_count > self.config.max_reasoning_tokens:
                self.stats["filtered_too_many_tokens"] += 1
                continue
            
            filtered.append(sample)
        
        return filtered
    
    def _deduplicate(self, samples: List[Dict]) -> List[Dict]:
        """Remove near-duplicate samples."""
        if not samples:
            return samples
        
        # Group by problem
        by_problem = defaultdict(list)
        for sample in samples:
            problem_id = sample.get("problem_id", sample.get("id", ""))
            by_problem[problem_id].append(sample)
        
        deduplicated = []
        
        for problem_id, problem_samples in by_problem.items():
            # Keep track of unique samples for this problem
            unique_samples = []
            
            for sample in problem_samples:
                reasoning = sample.get("reasoning", "")
                is_duplicate = False
                
                for existing in unique_samples:
                    existing_reasoning = existing.get("reasoning", "")
                    similarity = self._jaccard_similarity(reasoning, existing_reasoning)
                    
                    if similarity > self.config.dedup_threshold:
                        is_duplicate = True
                        self.stats["dedup_removed"] += 1
                        # Keep the one with higher confidence if available
                        if sample.get("verification_confidence", 0) > existing.get("verification_confidence", 0):
                            unique_samples.remove(existing)
                            unique_samples.append(sample)
                        break
                
                if not is_duplicate:
                    unique_samples.append(sample)
            
            deduplicated.extend(unique_samples)
        
        return deduplicated
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

    def _pair_quality_score(self, pos: Dict[str, Any], neg: Dict[str, Any], diversity: float) -> float:
        chosen_conf = float(pos.get("verification_confidence", 0.0))
        rejected_conf = float(neg.get("verification_confidence", 0.0))
        confidence_gap = max(0.0, chosen_conf - rejected_conf)

        neg_step_count = float(neg.get("reasoning_step_count", 0))
        pos_step_count = float(pos.get("reasoning_step_count", 0))
        step_alignment = min(neg_step_count, pos_step_count) / max(pos_step_count, 1.0)

        neg_error_type = self._classify_failure_mode(neg)
        error_type_bonus = {
            "answer_mismatch": 0.18,
            "reasoning_gap": 0.12,
            "parse_like_failure": -0.05,
            "missing_final_answer": -0.10,
            "trivial_short": -0.15,
            "repetitive": -0.12,
        }.get(neg_error_type, 0.0)

        answer_disagreement = (
            1.0
            if neg.get("answer_key")
            and pos.get("answer_key")
            and neg.get("answer_key") != pos.get("answer_key")
            else 0.0
        )

        final_answer_bonus = 0.08 if neg.get("has_final_answer", False) else -0.08
        near_miss_bonus = 0.0
        if self.config.prefer_near_miss_negatives:
            near_miss_bonus = min(rejected_conf, 0.75) * 0.12

        score = (
            0.38 * diversity
            + 0.22 * confidence_gap
            + 0.14 * step_alignment
            + 0.10 * answer_disagreement
            + final_answer_bonus
            + error_type_bonus
            + near_miss_bonus
        )
        return float(score)
    
    def _create_smart_pairs(self, samples: List[Dict]) -> List[Dict]:
        """Create DPO pairs with smart selection."""
        # Group by problem
        by_problem = defaultdict(lambda: {"correct": [], "incorrect": []})
        
        for sample in samples:
            problem_id = sample.get("problem_id", sample.get("id", ""))
            is_correct = sample.get("is_correct", False)
            
            if is_correct:
                by_problem[problem_id]["correct"].append(sample)
            else:
                by_problem[problem_id]["incorrect"].append(sample)
        
        pairs = []
        
        for problem_id, groups in by_problem.items():
            correct = groups["correct"]
            incorrect = groups["incorrect"]
            
            if not correct or not incorrect:
                continue
            
            # Sort by confidence if available
            if self.config.prefer_high_confidence:
                correct.sort(key=lambda x: x.get("verification_confidence", 0), reverse=True)
                incorrect.sort(key=lambda x: x.get("verification_confidence", 0), reverse=True)
            
            # Create pairs with diversity filtering
            problem_pairs = []
            
            for pos in correct:
                if float(pos.get("verification_confidence", 0.0)) < self.config.min_chosen_confidence:
                    self.stats["pairs_rejected_low_chosen_confidence"] += 1
                    continue
                for neg in incorrect:
                    pos_reasoning = pos.get("reasoning", "")
                    neg_reasoning = neg.get("reasoning", "")
                    
                    diversity = 1 - self._jaccard_similarity(pos_reasoning, neg_reasoning)

                    if diversity < self.config.min_pair_diversity:
                        continue

                    neg_error_type = self._classify_failure_mode(neg)
                    if (
                        self.config.require_answer_disagreement_for_pairs
                        and pos.get("answer_key")
                        and neg.get("answer_key")
                        and pos.get("answer_key") == neg.get("answer_key")
                    ):
                        self.stats["pairs_rejected_same_answer"] += 1
                        continue

                    if (
                        self.config.prioritize_step_aligned_pairs
                        and pos.get("reasoning_step_count", 0) > 0
                    ):
                        neg_step_ratio = (
                            float(neg.get("reasoning_step_count", 0))
                            / max(float(pos.get("reasoning_step_count", 0)), 1.0)
                        )
                        if (
                            neg_step_ratio < self.config.min_negative_step_ratio
                            and neg_error_type in {"trivial_short", "missing_final_answer", "repetitive"}
                        ):
                            self.stats["pairs_rejected_unstructured_negative"] += 1
                            continue

                    pair_quality = self._pair_quality_score(pos, neg, diversity)
                    if pair_quality < self.config.min_pair_quality_score:
                        self.stats["pairs_rejected_low_quality"] += 1
                        continue
                    pair = {
                        "prompt": pos.get("problem", ""),
                        "chosen": pos_reasoning,
                        "rejected": neg_reasoning,
                        "problem_id": problem_id,
                        "diversity_score": diversity,
                        "pair_quality_score": pair_quality,
                        "chosen_confidence": pos.get("verification_confidence", 0),
                        "rejected_confidence": neg.get("verification_confidence", 0),
                        "chosen_answer": pos.get("final_answer"),
                        "rejected_answer": neg.get("final_answer"),
                        "expected_answer": pos.get("expected_answer"),
                        "chosen_step_count": pos.get("reasoning_step_count", 0),
                        "rejected_step_count": neg.get("reasoning_step_count", 0),
                        "rejected_error_type": neg_error_type,
                    }
                    problem_pairs.append(pair)

            by_error_type = defaultdict(list)
            for pair in problem_pairs:
                by_error_type[pair["rejected_error_type"]].append(pair)

            selected_pairs: List[Dict[str, Any]] = []
            for error_type, bucket in by_error_type.items():
                bucket.sort(
                    key=lambda pair: (pair["pair_quality_score"], pair["diversity_score"]),
                    reverse=True,
                )
                selected_pairs.extend(bucket[: self.config.max_pairs_per_error_type])

            selected_pairs.sort(
                key=lambda pair: (pair["pair_quality_score"], pair["diversity_score"]),
                reverse=True,
            )
            pairs.extend(selected_pairs[: self.config.max_pairs_per_problem])
        
        return pairs
    
    def get_stats(self) -> Dict[str, int]:
        """Return preprocessing statistics."""
        return dict(self.stats)

    def summarize_pairs(self, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize the shape and quality of generated preference pairs."""
        if not pairs:
            return {
                "pair_count": 0,
                "avg_diversity_score": 0.0,
                "avg_pair_quality_score": 0.0,
                "error_type_counts": {},
            }

        error_type_counts = defaultdict(int)
        diversity_scores: List[float] = []
        pair_quality_scores: List[float] = []

        for pair in pairs:
            error_type_counts[pair.get("rejected_error_type", "unknown")] += 1
            diversity_scores.append(float(pair.get("diversity_score", 0.0)))
            pair_quality_scores.append(float(pair.get("pair_quality_score", 0.0)))

        return {
            "pair_count": len(pairs),
            "avg_diversity_score": sum(diversity_scores) / len(diversity_scores),
            "avg_pair_quality_score": sum(pair_quality_scores) / len(pair_quality_scores),
            "error_type_counts": dict(error_type_counts),
        }

    def summarize_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize filtered samples by correctness and failure mode."""
        if not samples:
            return {
                "sample_count": 0,
                "correct_count": 0,
                "incorrect_count": 0,
                "failure_mode_counts": {},
            }

        failure_mode_counts = defaultdict(int)
        correct_count = 0
        incorrect_count = 0

        for sample in samples:
            if sample.get("is_correct", False):
                correct_count += 1
                continue
            incorrect_count += 1
            failure_mode_counts[self._classify_failure_mode(sample)] += 1

        return {
            "sample_count": len(samples),
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "failure_mode_counts": dict(failure_mode_counts),
        }


def preprocess_jsonl(
    input_path: str,
    output_path: str,
    pairs_output_path: Optional[str] = None,
    config: Optional[PreprocessConfig] = None,
) -> Dict[str, int]:
    """
    Preprocess a JSONL file.
    
    Args:
        input_path: Path to input JSONL
        output_path: Path for filtered samples output
        pairs_output_path: Path for DPO pairs output (optional)
        config: Preprocessing configuration
    
    Returns:
        Statistics dictionary
    """
    import json
    
    # Load samples
    samples = []
    with open(input_path) as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    filtered_samples, pairs = preprocessor.preprocess(
        samples, 
        create_pairs=pairs_output_path is not None
    )
    
    # Save filtered samples
    with open(output_path, 'w') as f:
        for sample in filtered_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Save pairs if requested
    if pairs_output_path and pairs:
        with open(pairs_output_path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')
    
    return preprocessor.get_stats()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess reasoning data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--pairs-output", type=str, help="Output path for DPO pairs")
    parser.add_argument("--min-length", type=int, default=100, help="Min response length")
    parser.add_argument("--max-length", type=int, default=8000, help="Max response length")
    parser.add_argument("--dedup-threshold", type=float, default=0.85, help="Dedup threshold")
    
    args = parser.parse_args()
    
    config = PreprocessConfig(
        min_response_length=args.min_length,
        max_response_length=args.max_length,
        dedup_threshold=args.dedup_threshold,
    )
    
    logging.basicConfig(level=logging.INFO)
    stats = preprocess_jsonl(args.input, args.output, args.pairs_output, config)
    print(f"Stats: {stats}")
