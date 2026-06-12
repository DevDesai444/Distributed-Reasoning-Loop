"""Tests for the scaling-efficiency helpers used by the distributed launcher."""

from __future__ import annotations

import math

import pytest

from orchestration.scaling import (
    StepRecord,
    build_scaling_summary,
    measure_throughput,
    scaling_efficiency,
    summarize_records,
)


def test_summarize_records_empty():
    summary = summarize_records([])
    assert summary.n_steps == 0
    assert summary.tokens_per_sec == 0.0
    assert summary.samples_per_sec == 0.0
    assert summary.mean_step_ms == 0.0
    assert summary.p95_step_ms == 0.0


def test_summarize_records_token_and_sample_throughput():
    records = [
        StepRecord(step_index=i, wall_clock_s=0.1, tokens=10, samples=2)
        for i in range(5)
    ]
    summary = summarize_records(records)
    assert summary.n_steps == 5
    # 50 tokens / 0.5 s = 100 tokens/sec; 10 samples / 0.5 s = 20 samples/sec
    assert math.isclose(summary.tokens_per_sec, 100.0, rel_tol=1e-6)
    assert math.isclose(summary.samples_per_sec, 20.0, rel_tol=1e-6)
    assert math.isclose(summary.mean_step_ms, 100.0, rel_tol=1e-6)


def test_summarize_records_p95_is_monotone():
    # p95 should sit above mean and below the max for non-uniform steps.
    records = [
        StepRecord(step_index=0, wall_clock_s=0.05, tokens=1, samples=1),
        StepRecord(step_index=1, wall_clock_s=0.05, tokens=1, samples=1),
        StepRecord(step_index=2, wall_clock_s=0.05, tokens=1, samples=1),
        StepRecord(step_index=3, wall_clock_s=0.05, tokens=1, samples=1),
        StepRecord(step_index=4, wall_clock_s=1.00, tokens=1, samples=1),
    ]
    summary = summarize_records(records)
    assert summary.mean_step_ms < summary.p95_step_ms <= 1000.0


def test_scaling_efficiency_linear_speedup_is_100pct():
    # 4 GPUs, perfect scaling: speedup=4, efficiency=100%
    result = scaling_efficiency(
        single_gpu_throughput=10.0,
        multi_gpu_throughput=40.0,
        world_size=4,
    )
    assert math.isclose(result["speedup"], 4.0, rel_tol=1e-9)
    assert math.isclose(result["efficiency_pct"], 100.0, rel_tol=1e-9)
    assert math.isclose(result["ideal_speedup"], 4.0, rel_tol=1e-9)


def test_scaling_efficiency_partial_scaling():
    # 3.25x on 4 GPUs is 81.25% efficient
    result = scaling_efficiency(
        single_gpu_throughput=10.0,
        multi_gpu_throughput=32.5,
        world_size=4,
    )
    assert math.isclose(result["speedup"], 3.25, rel_tol=1e-9)
    assert math.isclose(result["efficiency_pct"], 81.25, rel_tol=1e-9)


def test_scaling_efficiency_rejects_nonpositive_world_size():
    with pytest.raises(ValueError):
        scaling_efficiency(1.0, 1.0, 0)


def test_scaling_efficiency_handles_zero_baseline():
    result = scaling_efficiency(0.0, 5.0, 2)
    assert result["speedup"] == 0.0
    assert result["efficiency_pct"] == 0.0
    assert result["ideal_speedup"] == 2.0


def test_measure_throughput_invokes_callable_n_times():
    calls = []

    def step(idx: int):
        calls.append(idx)
        return {"tokens": 5, "samples": 1}

    summary = measure_throughput(step, n_steps=4)
    assert calls == [0, 1, 2, 3]
    assert summary.n_steps == 4
    assert summary.tokens_per_sec > 0.0


def test_measure_throughput_rejects_zero_steps():
    with pytest.raises(ValueError):
        measure_throughput(lambda _: None, n_steps=0)


def test_build_scaling_summary_with_baseline_includes_speedup():
    records = [
        StepRecord(step_index=i, wall_clock_s=0.1, tokens=10, samples=2)
        for i in range(5)
    ]
    summary = summarize_records(records)
    out = build_scaling_summary(
        summary,
        world_size=2,
        single_gpu_throughput=10.0,
        metric="samples_per_sec",
    )
    # samples_per_sec=20 from the summary, baseline=10, world=2 -> speedup=2, eff=100%
    assert math.isclose(out["speedup"], 2.0, rel_tol=1e-6)
    assert math.isclose(out["efficiency_pct"], 100.0, rel_tol=1e-6)


def test_build_scaling_summary_unknown_metric():
    summary = summarize_records([])
    with pytest.raises(ValueError):
        build_scaling_summary(summary, world_size=1, metric="latency")
