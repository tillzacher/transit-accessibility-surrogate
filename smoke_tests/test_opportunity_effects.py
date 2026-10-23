import numpy as np
import pytest
from modules.opportunity_effects import (
    score_increment,
    score_attenuation,
    summarize_increment,
)


def run(a, b, n, floor=0.01, eligible=None):
    return score_increment([a], [b], [n], [floor], eligible=eligible)


def test_unchanged_background_does_not_dilute_contrast():
    a, b = (run(100, 90, 90), run(10, 0, 0))
    assert a["pass"][0] and b["pass"][0]
    assert a["score"][0] == b["score"][0] == 1
    assert 100 / 190 < 0.75


@pytest.mark.parametrize(
    "a,b,n,expected",
    [
        (100, 100, 90, "insufficient_attenuation"),
        (100, 89, 90, "negative_probe"),
        (89, 90, 90, "negative_reference"),
        (90, 90, 90, "low_reference_increment"),
        (3, 1, 0, "insufficient_attenuation"),
        (3, 0.999, 0, "pass"),
        (1, -1, 0, "invalid_input"),
        (np.nan, 1, 0, "invalid_input"),
        (np.inf, 1, 0, "invalid_input"),
    ],
)
def test_cases(a, b, n, expected):
    r = run(a, b, n)
    assert r["outcome"][0] == expected
    assert bool(r["pass"][0]) == (expected == "pass")


def test_negative_probe_is_retained_not_clamped():
    r = run(100, 89, 90)
    assert r["focal_probe"][0] == -1
    assert r["probe_reference_ratio"][0] == -0.1
    assert np.isnan(r["score"][0])
    assert r["supported"][0] and (not r["pass"][0])


def test_eligibility_and_support_are_different():
    r = score_increment(
        [1, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0.1] * 3,
        eligible=np.array([True, True, False]),
    )
    s = summarize_increment(r)
    assert s["conditional_pass_rate"] == 1
    assert s["support_coverage"] == s["pass_fraction_all_eligible"] == 0.5
    assert r["outcome"][2] == "outside_scope"


def test_zero_model_does_not_report_success():
    s = summarize_increment(run(0, 0, 0))
    assert s["conditional_pass_rate"] is None
    assert s["pass_fraction_all_eligible"] == s["support_coverage"] == 0


def test_empty_scope_does_not_report_success():
    s = summarize_increment(run(1, 0, 0, eligible=np.array([False])))
    assert s["conditional_pass_rate"] is None
    assert s["pass_fraction_all_eligible"] is None


def test_support_boundary_and_units():
    assert run(0.01, 0, 0)["supported"][0]
    assert not run(0.009, 0, 0)["supported"][0]
    a, b = (run(10, 2, 0, 0.1), run(100, 20, 0, 1))
    assert a["score"][0] == b["score"][0]


@pytest.mark.parametrize("floor", [0, -1, np.nan, np.inf])
def test_bad_floor_rejected(floor):
    with pytest.raises(ValueError):
        run(1, 0, 0, floor)


def test_shapes_and_scope_rejected():
    with pytest.raises(ValueError):
        score_increment([1, 2], [0], [0], [0.1])
    with pytest.raises(ValueError):
        run(1, 0, 0, eligible=np.array([1]))


def test_any_negative_probe_fails_strict_primary_rule():
    r = run(1, 0.1 - 1e-12, 0.1)
    assert r["negative_probe"][0] and (not r["pass"][0])


def test_attenuation_penalizes_both_signs_equally():
    pos = score_attenuation([100], [92], [90], [0.01])
    neg = score_attenuation([100], [88], [90], [0.01])
    assert pos["score"][0] == neg["score"][0] < 1
    assert pos["pass"][0] and neg["pass"][0]
    assert neg["negative_probe"][0] and (not neg["strict_joint_pass"][0])


def test_large_negative_residual_does_not_pass_attenuation():
    r = score_attenuation([100], [80], [90], [0.01])
    assert r["score"][0] == 0.5
    assert not r["pass"][0]


def test_attenuation_support_boundary_and_empty_support():
    assert not score_attenuation([3], [1], [0], [0.01])["pass"][0]
    assert score_attenuation([3], [0.99], [0], [0.01])["pass"][0]
    assert (
        summarize_increment(score_attenuation([0], [0], [0], [0.01]))[
            "conditional_pass_rate"
        ]
        is None
    )
