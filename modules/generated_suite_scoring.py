"""Channelwise scoring for the generated transit accessibility metamorphic suite."""

from __future__ import annotations
import copy
from pathlib import Path
from typing import Callable
import numpy as np
from modules.behavioral_thresholds import REPORTED_SCORES, passes
from modules.scoring_scenarios import ScenarioTile, load_scenario

PredictionFunction = Callable[[ScenarioTile], np.ndarray]


def score_evaluation_set(tiles, predict, *, dead_reference_floor):
    """Score eight relations using any callable returning a raw (C,H,W) field.

    No model, training pipeline, database, or routing engine is required. The
    opportunity control is mandatory; unsupported observations are not passes.
    """
    if "opportunity_background" not in tiles:
        raise ValueError(
            "Missing opportunity background control; regenerate the evaluation corpus."
        )
    background = tiles["opportunity_background"]
    eligible = np.asarray(background.metadata["opportunity_eligible"], dtype=bool)
    return score_set_channels(
        tiles,
        predict,
        dead_reference_floor=dead_reference_floor,
        opportunity_background=background,
        opportunity_eligible=eligible,
    )


EPS = 1e-12


def load_generated_set(path: Path) -> dict[str, ScenarioTile]:
    return {p.stem: load_scenario(p) for p in sorted(path.glob("*.npz"))}


def _scaled(tile: ScenarioTile, factor: float) -> ScenarioTile:
    return ScenarioTile(
        bus_stops=tile.bus_stops * factor,
        bus_edges=tile.bus_edges * factor,
        rail_raw=tile.rail_raw,
        poi_raw=tile.poi_raw,
        eval_mask=tile.eval_mask,
        control_mask=tile.control_mask,
        metadata=copy.deepcopy(tile.metadata),
    )


def _gain(prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if prediction.ndim != 3 or prediction.shape[1:] != mask.shape:
        raise ValueError(
            f"prediction/mask mismatch: {prediction.shape} vs {mask.shape}"
        )
    return np.maximum(prediction[:, mask], 0).mean(axis=1)


def _bounded(good: np.ndarray, bad: np.ndarray, *, sensitivity: bool) -> np.ndarray:
    if sensitivity:
        return good / (good + bad + EPS)
    return bad / (good + bad + EPS)


def score_set_channels(
    tiles: dict[str, ScenarioTile],
    predict: PredictionFunction,
    *,
    dead_reference_floor: float | np.ndarray = 1e-06,
    opportunity_background: ScenarioTile | None = None,
    opportunity_eligible: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Score eight reported relations before any aggregation over raw channels."""
    cache: dict[str, np.ndarray] = {}

    def pred(name: str) -> np.ndarray:
        if name not in cache:
            cache[name] = np.asarray(predict(tiles[name]), dtype=np.float32)
        return cache[name]

    pairs: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
    pairs["isolated_stop_leakage"] = (
        _gain(
            pred("connected_stop_with_poi"), tiles["connected_stop_with_poi"].eval_mask
        ),
        _gain(
            pred("isolated_stop_with_poi"), tiles["isolated_stop_with_poi"].eval_mask
        ),
        False,
    )
    pairs["poi_reachability_sensitivity"] = (
        _gain(pred("route_poi_reachable"), tiles["route_poi_reachable"].eval_mask),
        _gain(pred("route_poi_unreachable"), tiles["route_poi_unreachable"].eval_mask),
        True,
    )
    pairs["rail_connection_sensitivity"] = (
        _gain(
            pred("stop_near_rail_connected"),
            tiles["stop_near_rail_connected"].eval_mask,
        ),
        _gain(
            pred("stop_near_rail_isolated"), tiles["stop_near_rail_isolated"].eval_mask
        ),
        True,
    )
    full = tiles["scaled_bus_full"]
    full_gain = _gain(pred("scaled_bus_full"), full.eval_mask)
    weak_gain = _gain(np.asarray(predict(_scaled(full, 0.25))), full.eval_mask)
    pairs["weak_activation_leakage"] = (full_gain, weak_gain, False)
    pairs["overlap_diminishing_returns"] = (
        _gain(pred("route_single_line"), tiles["route_single_line"].eval_mask),
        _gain(
            pred("route_redundant_overlap"), tiles["route_redundant_overlap"].eval_mask
        ),
        False,
    )
    loc = tiles["localized_route"]
    if loc.control_mask is None:
        raise ValueError("localized_route has no control_mask")
    loc_pred = pred("localized_route")
    pairs["spatial_delta_localization"] = (
        _gain(loc_pred, loc.eval_mask),
        _gain(loc_pred, loc.control_mask),
        False,
    )
    strong_gain = _gain(np.asarray(predict(_scaled(full, 4.0))), full.eval_mask)
    pairs["bus_intensity_saturation"] = (full_gain, strong_gain, False)
    pairs["stub_edge_leakage"] = (
        _gain(pred("stub_edge_full_route"), tiles["stub_edge_full_route"].eval_mask),
        _gain(pred("stub_edge_origin_only"), tiles["stub_edge_origin_only"].eval_mask),
        False,
    )
    result: dict[str, dict[str, np.ndarray]] = {}
    for name in REPORTED_SCORES:
        good, bad, sensitivity = pairs[name]
        if name in {"overlap_diminishing_returns", "bus_intensity_saturation"}:
            score = bad / (good + bad + EPS)
        else:
            score = _bounded(good, bad, sensitivity=sensitivity)
        floor = np.asarray(dead_reference_floor, dtype=np.float32)
        if floor.ndim and floor.shape != good.shape:
            raise ValueError(
                f"dead-reference floor shape {floor.shape} != channel shape {good.shape}"
            )
        valid = good >= floor
        verdict = np.asarray(
            [passes(name, float(v)) if ok else False for v, ok in zip(score, valid)]
        )
        result[name] = {
            "score": score.astype(np.float32),
            "reference_gain": good.astype(np.float32),
            "probe_gain": bad.astype(np.float32),
            "valid": valid,
            "pass": verdict,
        }
    if opportunity_background is not None:
        from modules.opportunity_attenuation import score_attenuation

        ref = tiles["route_poi_reachable"]
        probe = tiles["route_poi_unreachable"]
        for tile in (probe, opportunity_background):
            for key in ("bus_stops", "bus_edges", "rail_raw", "eval_mask"):
                if not np.array_equal(getattr(ref, key), getattr(tile, key)):
                    raise ValueError(f"Opportunity control changes {key}")
        name = "poi_reachability_sensitivity"
        old = result[name]
        null_gain = _gain(np.asarray(predict(opportunity_background)), ref.eval_mask)
        floors = np.broadcast_to(np.asarray(dead_reference_floor), null_gain.shape)
        new = score_attenuation(
            old["reference_gain"],
            old["probe_gain"],
            null_gain,
            floors,
            eligible=opportunity_eligible,
        )
        result[name] = {
            **new,
            "valid": new["supported"],
            "reference_gain": old["reference_gain"],
            "probe_gain": old["probe_gain"],
            "null_gain": null_gain,
        }
    return result


__all__ = ["load_generated_set", "score_evaluation_set", "score_set_channels"]
