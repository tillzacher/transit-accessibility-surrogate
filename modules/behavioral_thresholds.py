"""Behavioral acceptance thresholds fixed before model evaluation."""

from __future__ import annotations
from typing import Dict, Optional

SCORE_KIND: Dict[str, str] = {
    "isolated_stop_leakage": "leakage",
    "stub_edge_leakage": "leakage",
    "spatial_delta_localization": "leakage",
    "weak_activation_leakage": "leakage",
    "poi_reachability_sensitivity": "sensitivity",
    "rail_connection_sensitivity": "sensitivity",
    "stop_density_monotonicity": "sensitivity",
    "overlap_diminishing_returns": "saturation",
    "bus_intensity_saturation": "saturation",
}
ALIASES = {
    "isl": "isolated_stop_leakage",
    "sel": "stub_edge_leakage",
    "sdl": "spatial_delta_localization",
    "wal": "weak_activation_leakage",
    "prs": "poi_reachability_sensitivity",
    "poi": "poi_reachability_sensitivity",
    "rcs": "rail_connection_sensitivity",
    "rail": "rail_connection_sensitivity",
    "sdm": "stop_density_monotonicity",
    "odr": "overlap_diminishing_returns",
    "bis": "bus_intensity_saturation",
}
LEAKAGE_MAX = 0.25
SENSITIVITY_MIN = 0.75
SATURATION_HALFWIDTH = 0.15
DEPRECATED_SCORES: Dict[str, str] = {
    "stop_density_monotonicity": "Excluded before the reported analysis. The relation asserts that densifying stops on a corridor must raise accessibility (kind='sensitivity', ideal 1). The simulator that generates the training targets says otherwise: measured on kappa tile 3, going from 15 to 30 stops on the same corridor lowered aggregate accessibility by 18%, because each stop adds dwell time and through-riders lose more than newly-served riders gain. That is the classical stop-spacing trade-off, so the premise is wrong rather than the reference calculation. Two further problems: (a) the scenario is a raster with no timetable, so it asks 'more stop blobs -> more gain?' while its training data comes from a dwell-aware simulator; (b) it is in direct tension with odr/bis, which penalise exactly the 'more bus mass -> more gain' response that sdm rewards."
}
REPORTED_SCORES = tuple((n for n in SCORE_KIND if n not in DEPRECATED_SCORES))


def _canon(name: str) -> str:
    return ALIASES.get(name, name)


def threshold(name: str) -> Dict[str, float]:
    """Return the pass rule for a score as a dict describing the bound(s)."""
    k = SCORE_KIND[_canon(name)]
    if k == "leakage":
        return {"kind": k, "op": "<", "bound": LEAKAGE_MAX}
    if k == "sensitivity":
        return {"kind": k, "op": ">", "bound": SENSITIVITY_MIN}
    return {"kind": k, "op": "within", "center": 0.5, "halfwidth": SATURATION_HALFWIDTH}


def passes(name: str, score: Optional[float]) -> Optional[bool]:
    """True/False if the score meets its pre-registered threshold; None if score is invalid."""
    if score is None:
        return None
    k = SCORE_KIND[_canon(name)]
    if k == "leakage":
        return score < LEAKAGE_MAX
    if k == "sensitivity":
        return score > SENSITIVITY_MIN
    return abs(score - 0.5) <= SATURATION_HALFWIDTH


