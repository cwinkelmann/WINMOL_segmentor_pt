import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from benchmark_cpu_latency import latency_stats


def test_latency_stats_computes_median_p90_min_mean():
    # 10 samples 1..10 ms
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    s = latency_stats(samples)
    assert s["min_ms"] == 1.0
    assert s["mean_ms"] == 5.5
    assert s["median_ms"] == 5.5          # average of 5th/6th
    assert s["p90_ms"] == 9.1             # linear-interp 90th percentile of 1..10
    assert s["runs"] == 10
