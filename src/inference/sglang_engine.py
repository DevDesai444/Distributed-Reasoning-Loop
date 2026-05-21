"""
SGLang inference engine wrapper.
Provides RadixAttention-optimized inference for reasoning models.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable, Tuple

try:
    from orchestration.kv_cache_manager import KVCacheManager
except ImportError:  # pragma: no cover - import path fallback for script usage
    from src.orchestration.kv_cache_manager import KVCacheManager

logger = logging.getLogger(__name__)


@dataclass
class SGLangConfig:
    """Configuration for SGLang engine."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # RadixAttention settings
    enable_radix_cache: bool = True
    max_radix_cache_size: int = 16 * 1024 * 1024 * 1024  # 16GB
    
    # Sampling
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 2048
    
    # Parallel sampling
    parallel_sample_num: int = 1

    # Local prefix index for cache-aware scheduling + hit accounting.
    # KV tensors are still owned by SGLang runtime.
    enable_local_prefix_index: bool = True
    local_prefix_index_memory_bytes: int = 2 * 1024 * 1024 * 1024  # 2GB
    enable_cache_aware_scheduling: bool = True


class SGLangEngine:
    """
    SGLang inference engine with RadixAttention support.
    Optimized for reasoning tasks with prefix caching.
    """
    
    def __init__(self, config: SGLangConfig):
        self.config = config
        self.runtime = None
        self._initialized = False
        self.tokenizer = None
        self.prefix_index: Optional[KVCacheManager] = None
        self._prefix_stats: Dict[str, float] = {
            "queries": 0,
            "hits": 0,
            "matched_prefix_tokens": 0,
        }

    def _init_prefix_index(self):
        if self.prefix_index is None and self.config.enable_local_prefix_index:
            self.prefix_index = KVCacheManager(
                max_memory_bytes=self.config.local_prefix_index_memory_bytes,
                eviction_policy="lru",
            )

    def _init_tokenizer(self):
        if not self.config.enable_local_prefix_index or self.tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize tokenizer for local prefix index; "
                "cache-aware scheduling disabled for SGLangEngine: %s",
                exc,
            )
            self.config.enable_local_prefix_index = False
            self.config.enable_cache_aware_scheduling = False

    def _tokenize_prompt(self, prompt: str) -> List[int]:
        if not self.tokenizer:
            return []
        try:
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        except Exception:
            return []

    def _estimate_and_record_prefix(self, prompt: str):
        if not self.prefix_index:
            return
        token_ids = self._tokenize_prompt(prompt)
        if not token_ids:
            return
        matched = self.prefix_index.get_prefix_match_length(token_ids)
        self._prefix_stats["queries"] += 1
        if matched > 0:
            self._prefix_stats["hits"] += 1
            self._prefix_stats["matched_prefix_tokens"] += matched
        self.prefix_index.record_prefix(token_ids, metadata={"engine": "sglang"})

    def _schedule_prompts_for_prefix_reuse(self, prompts: List[str]) -> Tuple[List[str], List[int]]:
        if not prompts:
            return [], []
        if not self.config.enable_cache_aware_scheduling or not self.prefix_index:
            return prompts, list(range(len(prompts)))

        indexed: List[Tuple[int, List[int], str]] = []
        for idx, prompt in enumerate(prompts):
            token_ids = self._tokenize_prompt(prompt)
            indexed.append((idx, token_ids, prompt))

        # Lexicographic token ordering clusters prompts with shared prefixes.
        indexed.sort(key=lambda item: item[1])
        ordered_prompts = [item[2] for item in indexed]
        original_indices = [item[0] for item in indexed]
        return ordered_prompts, original_indices
    
    def initialize(self):
        """Initialize SGLang runtime."""
        if self._initialized:
            return
        
        try:
            import sglang as sgl
            
            self.sgl = sgl
            self._init_prefix_index()
            self._init_tokenizer()
            
            # Set default model
            sgl.set_default_backend(sgl.RuntimeEndpoint(self.config.model_name))
            
            self._initialized = True
            logger.info(f"SGLang engine initialized with {self.config.model_name}")
            
        except ImportError:
            raise ImportError("SGLang not installed. Install with: pip install sglang")
    
    def create_cot_program(self, problem_type: str = "math"):
        """Create SGLang program for Chain-of-Thought generation."""
        self.initialize()
        sgl = self.sgl
        
        if problem_type == "math":
            @sgl.function
            def math_cot(s, problem):
                s += sgl.system("You are a helpful math tutor. Solve problems step by step.")
                s += sgl.user(f"Solve this problem:\n{problem}")
                s += sgl.assistant(sgl.gen("reasoning", max_tokens=self.config.max_tokens))
            
            return math_cot
        else:
            @sgl.function
            def code_cot(s, problem):
                s += sgl.system("You are an expert programmer. Solve coding problems step by step.")
                s += sgl.user(f"Solve this problem:\n{problem}")
                s += sgl.assistant(sgl.gen("solution", max_tokens=self.config.max_tokens))
            
            return code_cot
    
    def generate(
        self,
        problem: str,
        problem_type: str = "math",
    ) -> str:
        """
        Generate reasoning path for a single problem.
        Uses RadixAttention for efficient prefix caching.
        """
        program = self.create_cot_program(problem_type)
        
        state = program.run(
            problem=problem,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        self._estimate_and_record_prefix(problem)
        
        if problem_type == "math":
            return state["reasoning"]
        return state["solution"]
    
    def generate_multiple(
        self,
        problem: str,
        num_paths: int = 4,
        problem_type: str = "math",
    ) -> List[str]:
        """
        Generate multiple reasoning paths.
        RadixAttention caches the shared prefix automatically.
        """
        program = self.create_cot_program(problem_type)
        
        paths = []
        for _ in range(num_paths):
            state = program.run(
                problem=problem,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            
            if problem_type == "math":
                paths.append(state["reasoning"])
            else:
                paths.append(state["solution"])

        self._estimate_and_record_prefix(problem)
        
        return paths
    
    def batch_generate(
        self,
        problems: List[str],
        problem_type: str = "math",
    ) -> List[str]:
        """
        Generate for multiple problems in batch.
        """
        program = self.create_cot_program(problem_type)
        
        ordered_problems, original_indices = self._schedule_prompts_for_prefix_reuse(problems)

        # Run in parallel
        states = program.run_batch(
            [{"problem": p} for p in ordered_problems],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        key = "reasoning" if problem_type == "math" else "solution"
        ordered_outputs = [s[key] for s in states]

        # Restore original ordering.
        outputs = ["" for _ in problems]
        for ordered_idx, original_idx in enumerate(original_indices):
            outputs[original_idx] = ordered_outputs[ordered_idx]

        for prompt in problems:
            self._estimate_and_record_prefix(prompt)

        return outputs

    def generate_batch(
        self,
        problems: List[str],
        problem_type: str = "math",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Compatibility wrapper used by benchmarking scripts.
        Temporarily overrides sampling settings for a batched request.
        """
        old_temperature = self.config.temperature
        old_top_p = self.config.top_p
        old_max_tokens = self.config.max_tokens

        if temperature is not None:
            self.config.temperature = temperature
        if top_p is not None:
            self.config.top_p = top_p
        if max_tokens is not None:
            self.config.max_tokens = max_tokens

        try:
            return self.batch_generate(problems, problem_type=problem_type)
        finally:
            self.config.temperature = old_temperature
            self.config.top_p = old_top_p
            self.config.max_tokens = old_max_tokens
    
    def create_custom_program(
        self,
        program_fn: Callable,
    ) -> Callable:
        """
        Create a custom SGLang program.
        
        Args:
            program_fn: Function decorated with @sgl.function
            
        Returns:
            Compiled SGLang program
        """
        self.initialize()
        return self.sgl.function(program_fn)

    def get_prefix_cache_stats(self) -> Dict[str, Any]:
        """Return local prefix index stats and derived hit rates."""
        queries = int(self._prefix_stats["queries"])
        hits = int(self._prefix_stats["hits"])
        matched_tokens = int(self._prefix_stats["matched_prefix_tokens"])
        local_hit_rate = hits / queries if queries > 0 else 0.0

        payload: Dict[str, Any] = {
            "enabled_local_prefix_index": bool(self.prefix_index is not None),
            "queries": queries,
            "hits": hits,
            "local_hit_rate": local_hit_rate,
            "matched_prefix_tokens": matched_tokens,
            "note": "Local index tracks prefix reuse opportunities; runtime KV tensors are managed by SGLang.",
        }
        if self.prefix_index is not None:
            stats = self.prefix_index.get_stats()
            payload.update(
                {
                    "index_total_entries": stats.total_entries,
                    "index_memory_used_bytes": stats.memory_used_bytes,
                    "index_evictions": stats.evictions,
                }
            )
        return payload

    def reset_prefix_cache_stats(self):
        """Reset local prefix accounting and clear the local index."""
        self._prefix_stats = {
            "queries": 0,
            "hits": 0,
            "matched_prefix_tokens": 0,
        }
        if self.prefix_index is not None:
            self.prefix_index.clear()


class SGLangReasoningChain:
    """
    Multi-step reasoning chain using SGLang.
    Each step can branch and backtrack.
    """
    
    def __init__(self, config: SGLangConfig):
        self.config = config
        self.engine = SGLangEngine(config)
    
    def solve_with_verification(
        self,
        problem: str,
        verifier: Callable[[str, str], bool],
        max_attempts: int = 5,
    ) -> Optional[str]:
        """
        Generate solutions and verify until correct.
        
        Args:
            problem: The problem to solve
            verifier: Function that takes (solution, problem) and returns bool
            max_attempts: Maximum generation attempts
            
        Returns:
            Verified solution or None
        """
        for _ in range(max_attempts):
            solution = self.engine.generate(problem)
            
            if verifier(solution, problem):
                return solution
        
        return None
    
    def multi_turn_reasoning(
        self,
        problem: str,
        num_turns: int = 3,
    ) -> str:
        """
        Multi-turn reasoning where model refines its solution.
        """
        self.engine.initialize()
        sgl = self.engine.sgl
        
        @sgl.function
        def multi_turn(s, problem):
            s += sgl.system("You are a helpful assistant. Think step by step.")
            s += sgl.user(f"Problem: {problem}\n\nFirst, understand the problem:")
            s += sgl.assistant(sgl.gen("understanding", max_tokens=256))
            
            s += sgl.user("Now, outline your approach:")
            s += sgl.assistant(sgl.gen("approach", max_tokens=256))
            
            s += sgl.user("Finally, solve the problem and provide the answer:")
            s += sgl.assistant(sgl.gen("solution", max_tokens=512))
        
        state = multi_turn.run(
            problem=problem,
            temperature=self.config.temperature,
        )
        
        return f"""Understanding: {state['understanding']}

Approach: {state['approach']}

Solution: {state['solution']}"""
