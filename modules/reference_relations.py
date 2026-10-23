"""Unchanged relational score functions used by reference checks."""

import numpy as np
from modules.behavioral_thresholds import SCORE_KIND, passes

EPS = 1e-12


def score_relation(
    relation: str, reference_gain: np.ndarray, probe_gain: np.ndarray, floor: np.ndarray
) -> dict[str, np.ndarray]:
    reference_gain = np.asarray(reference_gain, dtype=np.float32)
    probe_gain = np.asarray(probe_gain, dtype=np.float32)
    floor = np.asarray(floor, dtype=np.float32)
    if not reference_gain.shape == probe_gain.shape == floor.shape:
        raise ValueError(
            "reference, probe, and floor vectors must have identical shape"
        )
    if SCORE_KIND[relation] == "sensitivity":
        score = reference_gain / (reference_gain + probe_gain + EPS)
    else:
        score = probe_gain / (reference_gain + probe_gain + EPS)
    valid = reference_gain >= floor
    passed = np.asarray(
        [
            bool(passes(relation, float(value))) if ok else False
            for value, ok in zip(score, valid)
        ],
        dtype=bool,
    )
    return {
        "score": score.astype(np.float32),
        "reference_gain": reference_gain,
        "probe_gain": probe_gain,
        "valid": valid,
        "pass": passed,
    }
