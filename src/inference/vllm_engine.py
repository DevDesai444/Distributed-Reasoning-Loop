"""
vLLM inference engine wrapper.
Provides high-throughput inference for reasoning models.
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

try:
    from orchestration.kv_cache_manager import KVCacheManager
except ImportError:  # pragma: no cover - import path fallback for script usage
    from src.orchestration.kv_cache_manager import KVCacheManager

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer environment variable %s=%r", name, value)
        return None


@dataclass
class VLLMConfig:
    """Configuration for vLLM engine."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = False
    
    # Sampling
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 2048
    
    # Batching
    max_num_seqs: int = 256

    # vLLM native prefix caching.
    enable_prefix_caching: bool = True

    # Local prefix index for cache-aware scheduling + hit accounting.
    # KV tensors remain owned by vLLM.
    enable_local_prefix_index: bool = True
    local_prefix_index_memory_bytes: int = 2 * 1024 * 1024 * 1024  # 2GB
    enable_cache_aware_scheduling: bool = True


class VLLMEngine:
    """
    vLLM inference engine for high-throughput generation.
    """
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.llm = None
        self.sampling_params = None
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
                "cache-aware scheduling disabled for VLLMEngine: %s",
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
        self.prefix_index.record_prefix(token_ids, metadata={"engine": "vllm"})

    def _schedule_prompts_for_prefix_reuse(self, prompts: List[str]) -> Tuple[List[str], List[int]]:
        if not prompts:
            return [], []
        if not self.config.enable_cache_aware_scheduling or not self.prefix_index:
            return prompts, list(range(len(prompts)))

        indexed: List[Tuple[int, List[int], str]] = []
        for idx, prompt in enumerate(prompts):
            token_ids = self._tokenize_prompt(prompt)
            indexed.append((idx, token_ids, prompt))

        indexed.sort(key=lambda item: item[1])
        ordered_prompts = [item[2] for item in indexed]
        original_indices = [item[0] for item in indexed]
        return ordered_prompts, original_indices
    
    def initialize(self):
        """Initialize vLLM engine."""
        if self._initialized:
            return
        
        try:
            from vllm import LLM, SamplingParams

            llm_kwargs = {
                "model": self.config.model_name,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_model_len": _env_int("DRL_VLLM_MAX_MODEL_LEN") or self.config.max_model_len,
                "enable_prefix_caching": self.config.enable_prefix_caching,
                "trust_remote_code": True,
            }
            if self.config.enforce_eager or _env_flag("DRL_VLLM_ENFORCE_EAGER"):
                llm_kwargs["enforce_eager"] = True
            if self.config.disable_custom_all_reduce or _env_flag("DRL_VLLM_DISABLE_CUSTOM_ALL_REDUCE"):
                llm_kwargs["disable_custom_all_reduce"] = True

            self.llm = LLM(**llm_kwargs)
            
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_tokens=self.config.max_tokens,
            )
            
            self._initialized = True
            self._init_prefix_index()
            self._init_tokenizer()
            logger.info(
                "vLLM engine initialized with %s (tp=%s, eager=%s, max_model_len=%s)",
                self.config.model_name,
                self.config.tensor_parallel_size,
                bool(llm_kwargs.get("enforce_eager", False)),
                llm_kwargs.get("max_model_len"),
            )
            
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate completions for multiple prompts.
        
        Args:
            prompts: List of prompts
            sampling_params: Optional override sampling parameters
            
        Returns:
            List of generated texts
        """
        self.initialize()
        
        from vllm import SamplingParams
        
        if sampling_params:
            params = SamplingParams(**sampling_params)
        else:
            params = self.sampling_params
        
        ordered_prompts, original_indices = self._schedule_prompts_for_prefix_reuse(prompts)
        outputs = self.llm.generate(ordered_prompts, params)

        ordered_results = []
        for output in outputs:
            ordered_results.append(output.outputs[0].text)

        results = [""] * len(prompts)
        for ordered_idx, original_idx in enumerate(original_indices):
            results[original_idx] = ordered_results[ordered_idx]

        for prompt in prompts:
            self._estimate_and_record_prefix(prompt)

        return results
    
    def generate_with_logprobs(
        self,
        prompts: List[str],
        num_logprobs: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate with log probabilities."""
        self.initialize()
        
        from vllm import SamplingParams
        
        params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            logprobs=num_logprobs,
        )
        
        outputs = self.llm.generate(prompts, params)
        
        results = []
        for output in outputs:
            result = {
                "text": output.outputs[0].text,
                "logprobs": [],
            }
            
            if output.outputs[0].logprobs:
                for token_logprobs in output.outputs[0].logprobs:
                    result["logprobs"].append({
                        token: logprob
                        for token, logprob in token_logprobs.items()
                    })
            
            results.append(result)
        
        return results
    
    def batch_generate_cot(
        self,
        problems: List[str],
        problem_type: str = "math",
        num_paths_per_problem: int = 1,
    ) -> List[List[str]]:
        """
        Generate Chain-of-Thought solutions for multiple problems.
        
        Args:
            problems: List of problems
            problem_type: 'math' or 'code'
            num_paths_per_problem: Paths to generate per problem
            
        Returns:
            List of lists of reasoning paths
        """
        self.initialize()
        
        # Format prompts
        if problem_type == "math":
            template = """Solve this math problem step by step. Show your work and end with #### followed by the answer.

Problem: {problem}

Solution:"""
        else:
            template = """Solve this coding problem. Explain your approach and provide the solution.

Problem: {problem}

Solution:"""
        
        # Create all prompts
        all_prompts = []
        prompt_mapping = []
        
        for i, problem in enumerate(problems):
            for _ in range(num_paths_per_problem):
                all_prompts.append(template.format(problem=problem))
                prompt_mapping.append(i)
        
        # Generate
        outputs = self.generate(all_prompts)
        
        # Organize results
        results = [[] for _ in problems]
        for output, problem_idx in zip(outputs, prompt_mapping):
            results[problem_idx].append(output)
        
        return results

    def get_prefix_cache_stats(self) -> Dict[str, Any]:
        """Return local prefix index stats and derived hit rates."""
        queries = int(self._prefix_stats["queries"])
        hits = int(self._prefix_stats["hits"])
        matched_tokens = int(self._prefix_stats["matched_prefix_tokens"])
        local_hit_rate = hits / queries if queries > 0 else 0.0

        payload: Dict[str, Any] = {
            "enabled_native_prefix_caching": self.config.enable_prefix_caching,
            "enabled_local_prefix_index": bool(self.prefix_index is not None),
            "queries": queries,
            "hits": hits,
            "local_hit_rate": local_hit_rate,
            "matched_prefix_tokens": matched_tokens,
            "note": "Local index tracks prefix reuse opportunities; runtime KV tensors are managed by vLLM.",
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


class AsyncVLLMEngine:
    """
    Async vLLM engine for streaming and async generation.
    """
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.engine = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize async engine."""
        if self._initialized:
            return
        
        try:
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs
            
            engine_args = AsyncEngineArgs(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._initialized = True
            
        except ImportError:
            raise ImportError("vLLM async engine not available")
    
    async def generate_streaming(
        self,
        prompt: str,
        request_id: str,
    ):
        """
        Generate with streaming output.
        
        Yields tokens as they are generated.
        """
        await self.initialize()
        
        from vllm import SamplingParams
        
        params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        async for output in self.engine.generate(prompt, params, request_id):
            yield output.outputs[0].text
