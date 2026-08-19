"""The aggregation is the claim — pin its arithmetic, not just that it runs.

A wrong sign or an unpaired mean here would produce a confident wrong conclusion about
scale robustness, which is exactly the failure mode `docs/process.md` exists to stop.
"""
import json
import math

import pytest

from scripts.plot_scale_sweep import drop_from_peak, load, paired, paired_spread

CROPS = [394, 512, 666]


def _write(d, arm, seed, f1s, crops=CROPS):
    rows = [{"crop_px": c, "f1": f, "precision": f, "recall": f, "tiles": 10,
             "gsd_cm": 2.9297 * c / 512, "ratio": c / 512}
            for c, f in zip(crops, f1s)]
    (d / f"sweep-{arm}-s{seed}.json").write_text(
        json.dumps({"label": f"{arm}-s{seed}", "model": "x.onnx", "rows": rows}))


def test_load_groups_by_arm_and_seed(tmp_path):
    _write(tmp_path, "fixed", 1, [0.70, 0.80, 0.72])
    _write(tmp_path, "jitter", 1, [0.75, 0.79, 0.77])
    data, crops = load(str(tmp_path), ["fixed", "jitter"])
    assert crops == CROPS
    assert sorted(data["fixed"]) == [1] and sorted(data["jitter"]) == [1]
    assert data["fixed"][1][512]["f1"] == pytest.approx(0.80)


def test_diff_is_second_arm_minus_first(tmp_path):
    """Baseline first: a positive diff must mean the second arm won."""
    _write(tmp_path, "fixed", 1, [0.70, 0.80, 0.72])
    _write(tmp_path, "jitter", 1, [0.75, 0.79, 0.77])
    data, crops = load(str(tmp_path), ["fixed", "jitter"])
    rows = {r["crop_px"]: r for r in paired(data, ["fixed", "jitter"], crops)}
    assert rows[394]["mean_diff"] == pytest.approx(0.05)    # jitter better at the flank
    assert rows[512]["mean_diff"] == pytest.approx(-0.01)   # fixed better at native


def test_pairs_only_seeds_present_in_both_arms(tmp_path):
    """An unmatched seed must be dropped, not averaged in — that would unpair the test."""
    _write(tmp_path, "fixed", 1, [0.70, 0.80, 0.72])
    _write(tmp_path, "fixed", 2, [0.10, 0.10, 0.10])       # no jitter counterpart
    _write(tmp_path, "jitter", 1, [0.75, 0.79, 0.77])
    data, crops = load(str(tmp_path), ["fixed", "jitter"])
    r = paired(data, ["fixed", "jitter"], crops)[0]
    assert r["seeds"] == [1] and r["n"] == 1
    assert r["mean_diff"] == pytest.approx(0.05)


def test_t_statistic_uses_the_paired_differences(tmp_path):
    for s, (f, j) in enumerate([(0.70, 0.75), (0.60, 0.66), (0.80, 0.84)], start=1):
        _write(tmp_path, "fixed", s, [f, f, f])
        _write(tmp_path, "jitter", s, [j, j, j])
    data, crops = load(str(tmp_path), ["fixed", "jitter"])
    r = paired(data, ["fixed", "jitter"], crops)[0]
    diffs = [0.05, 0.06, 0.04]
    mean = sum(diffs) / 3
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / 2)
    assert r["mean_diff"] == pytest.approx(mean)
    assert r["t"] == pytest.approx(mean / (sd / math.sqrt(3)))
    # Arms are 10-20 F1 apart between seeds but every pair moves the same way: the
    # within-arm spread would call this noise, the paired test must not.
    assert r["consistent"] and r["strong"]


def test_drop_from_peak_is_per_seed_worst_to_best(tmp_path):
    _write(tmp_path, "fixed", 1, [0.70, 0.80, 0.72])
    data, crops = load(str(tmp_path), ["fixed", "jitter"])
    p = drop_from_peak(data, "fixed", crops)[1]
    assert p["peak"] == pytest.approx(0.80) and p["peak_crop"] == 512
    assert p["spread"] == pytest.approx(0.10)


def test_mismatched_crop_axis_fails_loudly(tmp_path):
    _write(tmp_path, "fixed", 1, [0.70, 0.80, 0.72])
    _write(tmp_path, "jitter", 1, [0.75, 0.79], crops=[394, 512])
    with pytest.raises(SystemExit, match="different crop axis"):
        load(str(tmp_path), ["fixed", "jitter"])


def test_paired_spread_flags_the_flatter_arm(tmp_path):
    """Negative mean = arm B flatter. 3/3 consistent must not be called noise."""
    for s, (f, j) in enumerate([([0.70, 0.80, 0.70], [0.76, 0.79, 0.76]),
                                ([0.60, 0.74, 0.60], [0.70, 0.75, 0.70]),
                                ([0.75, 0.84, 0.75], [0.80, 0.84, 0.80])], start=1):
        _write(tmp_path, "fixed", s, f)
        _write(tmp_path, "jitter", s, j)
    data, crops = load(str(tmp_path), ["fixed", "jitter"])
    peaks = {a: drop_from_peak(data, a, crops) for a in ("fixed", "jitter")}
    ps = paired_spread(peaks, ["fixed", "jitter"])
    assert ps["mean_diff"] < 0 and ps["flatter"] == 3
    assert ps["consistent"] and ps["strong"]
