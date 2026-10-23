"""Dependency-light smoke test for the paper's relation thresholds and channel scoring."""

from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.behavioral_thresholds import passes
from modules.generated_suite_scoring import _bounded, _gain


def main() -> int:
    prediction = np.array([[[1.0, -1.0], [3.0, 5.0]]], dtype=np.float32)
    mask = np.array([[True, False], [True, False]])
    gain = _gain(prediction, mask)
    assert np.allclose(gain, [2.0])
    good = np.array([3.0], dtype=np.float32)
    bad = np.array([1.0], dtype=np.float32)
    leakage = _bounded(good, bad, sensitivity=False)
    sensitivity = _bounded(good, bad, sensitivity=True)
    assert np.allclose(leakage, [0.25])
    assert np.allclose(sensitivity, [0.75])
    assert passes("isolated_stop_leakage", 0.24)
    assert not passes("isolated_stop_leakage", 0.25)
    assert passes("rail_connection_sensitivity", 0.76)
    assert not passes("rail_connection_sensitivity", 0.75)
    assert passes("bus_intensity_saturation", 0.5)
    print("relation-scoring smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
