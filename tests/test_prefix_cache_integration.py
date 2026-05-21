"""
Tests for local prefix-index integration in inference engines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestration.kv_cache_manager import KVCacheManager
from inference.sglang_engine import SGLangEngine, SGLangConfig
from inference.vllm_engine import VLLMEngine, VLLMConfig


class _DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in text]


def test_kv_cache_record_prefix_and_match_length():
    cache = KVCacheManager(max_memory_bytes=1024)
    cache.record_prefix([1, 2, 3], metadata={"source": "test"})
    cache.record_prefix([1, 2, 3], metadata={"source": "test2"})

    matched = cache.get_prefix_match_length([1, 2, 3, 4])
    assert matched == 3

    stats = cache.get_stats()
    assert stats.hits == 1
    assert stats.total_entries == 1


def test_sglang_local_prefix_index_stats():
    engine = SGLangEngine(SGLangConfig(enable_local_prefix_index=True))
    engine.tokenizer = _DummyTokenizer()
    engine.prefix_index = KVCacheManager(max_memory_bytes=1024)

    engine._estimate_and_record_prefix("abc")
    engine._estimate_and_record_prefix("abc")

    stats = engine.get_prefix_cache_stats()
    assert stats["queries"] == 2
    assert stats["hits"] == 1
    assert stats["local_hit_rate"] == 0.5


def test_vllm_cache_aware_prompt_scheduling():
    engine = VLLMEngine(VLLMConfig(enable_local_prefix_index=True, enable_cache_aware_scheduling=True))
    engine.tokenizer = _DummyTokenizer()
    engine.prefix_index = KVCacheManager(max_memory_bytes=1024)

    prompts = ["bbb", "aaa", "aac"]
    ordered_prompts, original_indices = engine._schedule_prompts_for_prefix_reuse(prompts)

    assert ordered_prompts == ["aaa", "aac", "bbb"]
    assert original_indices == [1, 2, 0]
