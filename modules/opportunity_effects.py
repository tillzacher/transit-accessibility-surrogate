"""Background-controlled opportunity effects and signed diagnostics."""

from __future__ import annotations
import numpy as np

VERSION = "opportunity-effects-1"
SENSITIVITY_GATE = 0.75


def score_increment(gain_a, gain_b, gain_n, floor, *, eligible=None):
    """Return signed diagnostics, support and pass flags for one channel vector."""
    a, b, n, f = [
        np.asarray(x, dtype=np.float64) for x in (gain_a, gain_b, gain_n, floor)
    ]
    if a.ndim != 1 or not a.shape == b.shape == n.shape == f.shape:
        raise ValueError("gains and floor must be matching one-dimensional vectors")
    if not np.all(np.isfinite(f) & (f > 0)):
        raise ValueError("support floors must be finite and strictly positive")
    scope = np.ones(a.shape, dtype=bool) if eligible is None else np.asarray(eligible)
    if scope.shape != a.shape or scope.dtype != np.bool_:
        raise ValueError("eligible must be a matching boolean vector")
    with np.errstate(over="ignore", invalid="ignore"):
        fa, fb = (a - n, b - n)
    finite = (
        np.isfinite(a)
        & np.isfinite(b)
        & np.isfinite(n)
        & np.isfinite(fa)
        & np.isfinite(fb)
        & (a >= 0)
        & (b >= 0)
        & (n >= 0)
    )
    supported = scope & finite & (fa >= f)
    negative_a = scope & finite & (fa < 0)
    negative_b = scope & finite & (fb < 0)
    ratio = np.full(a.shape, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(fb, fa, out=ratio, where=scope & finite & (fa > 0))
    sign_ok = finite & (fa >= 0) & (fb >= 0)
    contrast_pass = supported & (ratio < (1 - SENSITIVITY_GATE) / SENSITIVITY_GATE)
    passed = contrast_pass & sign_ok
    bounded_score = np.full(a.shape, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(1.0, 1.0 + ratio, out=bounded_score, where=supported & sign_ok)
    outcome = np.full(a.shape, "outside_scope", dtype="U24")
    outcome[scope & ~finite] = "invalid_input"
    outcome[scope & finite & ~supported] = "low_reference_increment"
    outcome[negative_a] = "negative_reference"
    outcome[supported & negative_b] = "negative_probe"
    outcome[supported & sign_ok & ~passed] = "insufficient_attenuation"
    outcome[passed] = "pass"
    return {
        "focal_reference": fa,
        "focal_probe": fb,
        "probe_reference_ratio": ratio,
        "score": bounded_score,
        "eligible": scope,
        "supported": supported,
        "negative_reference": negative_a,
        "negative_probe": negative_b,
        "invalid": scope & ~finite,
        "pass": passed,
        "outcome": outcome,
    }


def summarize_increment(result):
    """Keep exclusions visible; an empty denominator returns None, never 100%."""
    counts = {
        name: int(np.count_nonzero(result[name]))
        for name in (
            "eligible",
            "supported",
            "negative_reference",
            "negative_probe",
            "invalid",
            "pass",
        )
    }
    n, s, p = (counts["eligible"], counts["supported"], counts["pass"])
    return {
        **{f"n_{k}": v for k, v in counts.items()},
        "support_coverage": s / n if n else None,
        "conditional_pass_rate": p / s if s else None,
        "pass_fraction_all_eligible": p / n if n else None,
    }


def score_attenuation(gain_a, gain_b, gain_n, floor, *, eligible=None):
    """Test attenuation magnitude and audit sign separately."""
    result = score_increment(gain_a, gain_b, gain_n, floor, eligible=eligible)
    result["strict_joint_pass"] = result["pass"].copy()
    magnitude_ratio = np.abs(result["probe_reference_ratio"])
    passed = result["supported"] & (
        magnitude_ratio < (1 - SENSITIVITY_GATE) / SENSITIVITY_GATE
    )
    score = np.full(magnitude_ratio.shape, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(1.0, 1.0 + magnitude_ratio, out=score, where=result["supported"])
    result["score"] = score
    result["pass"] = passed
    result["magnitude_ratio"] = magnitude_ratio
    result["outcome"][result["supported"]] = "insufficient_attenuation"
    result["outcome"][passed] = "pass"
    return result
