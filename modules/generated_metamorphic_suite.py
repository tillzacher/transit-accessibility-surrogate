"""Deterministic, full-channel metamorphic scenarios for transit accessibility."""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom
from modules.scoring_scenarios import ScenarioTile, save_scenario

FORMAT = "surrogate-generated-metamorphic-suite-v2"
GENERATOR_VERSION = "2.1.1"
TILE_SIZE = 200
STOP_SIGMA_PX = 1.0
EDGE_SIGMA_PX = 0.65
POI_SIGMA_PX = 1.0
SOFT_CAP_MAX = 1.0
SOFT_CAP_BETA = 10.0
MARGIN = 10
FAMILY_NAMES = (
    "isolated_stop_leakage",
    "poi_reachability_sensitivity",
    "rail_connection_sensitivity",
    "amplitude_response",
    "spatial_delta_localization",
    "stub_edge_leakage",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _soft_cap(x: np.ndarray) -> np.ndarray:
    cap = SOFT_CAP_MAX
    beta = SOFT_CAP_BETA
    softplus = lambda a: np.logaddexp(0.0, beta * a) / beta
    return (x - softplus(x - cap) + softplus(-cap)).astype(np.float32)


def gaussian_splat(
    shape: tuple[int, int], row: float, col: float, sigma: float, peak: float = 1.0
) -> np.ndarray:
    """Normalized Gaussian point splat with three-sigma support."""
    h, w = shape
    radius = int(np.ceil(3.0 * sigma))
    c0, c1 = (max(0, int(col) - radius), min(w - 1, int(col) + radius))
    r0, r1 = (max(0, int(row) - radius), min(h - 1, int(row) + radius))
    out = np.zeros(shape, dtype=np.float32)
    if c0 > c1 or r0 > r1:
        return out
    cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
    values = np.exp(-0.5 * ((cc - col) ** 2 + (rr - row) ** 2) / sigma**2)
    if values.size and values.max() > 0:
        values = peak * values / values.max()
    out[r0 : r1 + 1, c0 : c1 + 1] = values.astype(np.float32)
    return out


def line_splat(
    shape: tuple[int, int],
    p0: tuple[float, float],
    p1: tuple[float, float],
    sigma: float,
) -> np.ndarray:
    """Gaussian line rasterization with two samples per pixel."""
    h, w = shape
    r0, c0 = p0
    r1, c1 = p1
    length = float(np.hypot(c1 - c0, r1 - r0))
    n = max(2, int(np.ceil(length * 2)))
    radius = int(np.ceil(3.0 * sigma))
    accum = np.zeros(shape, dtype=np.float64)
    for t in np.linspace(0.0, 1.0, n):
        r, c = (r0 + t * (r1 - r0), c0 + t * (c1 - c0))
        rows = np.arange(
            max(0, int(np.floor(r)) - radius), min(h - 1, int(np.floor(r)) + radius) + 1
        )
        cols = np.arange(
            max(0, int(np.floor(c)) - radius), min(w - 1, int(np.floor(c)) + radius) + 1
        )
        cc, rr = np.meshgrid(cols, rows)
        values = np.exp(-0.5 * ((cc - c) ** 2 + (rr - r) ** 2) / sigma**2)
        accum[np.ix_(rows, cols)] += values
    accum /= n
    if accum.max() > 0:
        accum /= accum.max()
    return accum.astype(np.float32)


def polyline_splat(
    shape: tuple[int, int], points: Iterable[tuple[float, float]]
) -> np.ndarray:
    points = list(points)
    out = np.zeros(shape, dtype=np.float32)
    for a, b in zip(points[:-1], points[1:]):
        out += line_splat(shape, a, b, EDGE_SIGMA_PX)
    return _soft_cap(out)


def point_splats(
    shape: tuple[int, int],
    points: Iterable[tuple[float, float]],
    sigma: float = STOP_SIGMA_PX,
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    for row, col in points:
        out += gaussian_splat(shape, row, col, sigma)
    return _soft_cap(out)


def _path(
    rng: np.random.Generator, length: float | None = None
) -> list[tuple[float, float]]:
    length = float(length if length is not None else rng.uniform(70, 120))
    for _ in range(500):
        theta = rng.uniform(0, 2 * np.pi)
        center = rng.uniform(58, 142, size=2)
        delta = 0.5 * length * np.array([np.sin(theta), np.cos(theta)])
        normal = np.array([-np.cos(theta), np.sin(theta)])
        bend = rng.uniform(-0.18, 0.18) * length
        pts = [
            tuple(center - delta),
            tuple(center + bend * normal),
            tuple(center + delta),
        ]
        safe = MARGIN + int(np.ceil(3 * max(STOP_SIGMA_PX, EDGE_SIGMA_PX))) + 1
        if all((safe <= x <= TILE_SIZE - 1 - safe for p in pts for x in p)):
            return pts
    raise RuntimeError("could not place an interior route")


def _sample_path(
    points: list[tuple[float, float]], n: int
) -> list[tuple[float, float]]:
    seg = np.asarray(points, dtype=float)
    lens = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lens)]
    targets = np.linspace(0, cumulative[-1], n)
    out = []
    for x in targets:
        j = min(int(np.searchsorted(cumulative, x, side="right") - 1), len(lens) - 1)
        f = (x - cumulative[j]) / max(lens[j], 1e-09)
        out.append(tuple(seg[j] + f * (seg[j + 1] - seg[j])))
    return out


def _far_anchor(
    rng: np.random.Generator, avoid: Iterable[tuple[float, float]]
) -> tuple[float, float]:
    avoid_array = np.asarray(list(avoid), dtype=float)
    safe = MARGIN + int(np.ceil(3 * POI_SIGMA_PX)) + 2
    for _ in range(500):
        point = rng.uniform(safe, TILE_SIZE - 1 - safe, size=2)
        if np.min(np.linalg.norm(avoid_array - point, axis=1)) >= 50:
            return tuple(point)
    raise RuntimeError("could not place a focal POI far from the route")


def _disk_mask(points: Iterable[tuple[float, float]], radius: float = 15) -> np.ndarray:
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    for p in points:
        mask |= gaussian_splat(mask.shape, *p, sigma=max(radius / 3, 1)) > np.exp(-4.5)
    return mask


def _route_masks(edge: np.ndarray, width: int = 10) -> tuple[np.ndarray, np.ndarray]:
    near = distance_transform_edt(edge <= 0) <= width
    distance = distance_transform_edt(~near)
    candidates = np.flatnonzero(~near)
    if len(candidates) < int(near.sum()):
        raise RuntimeError("insufficient area for an equal-area localization control")
    interior = distance.copy()
    interior[:30] = 0
    interior[-30:] = 0
    interior[:, :30] = 0
    interior[:, -30:] = 0
    center = np.asarray(np.unravel_index(np.argmax(interior), interior.shape))
    coords = np.column_stack(np.unravel_index(candidates, near.shape))
    radial = np.sum((coords - center) ** 2, axis=1)
    order = candidates[np.argsort(radial, kind="stable")]
    control = np.zeros_like(near)
    control.ravel()[order[: int(near.sum())]] = True
    return (near, control)


def _coverage_mask(
    cal: dict[str, Any], rng: np.random.Generator, mode: str
) -> np.ndarray:
    c = len(cal["channels"])
    if mode == "all_channels":
        return np.ones(c, dtype=bool)
    summary = cal["poi"]["active_channels_per_tile"]
    q05, q25, q50, q75, q95 = (
        float(summary[name]) for name in ("q05", "q25", "q50", "q75", "q95")
    )
    probabilities = np.asarray((0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0))
    counts = np.asarray(
        (
            max(1.0, q05 - (q25 - q05)),
            q05,
            q25,
            q50,
            q75,
            q95,
            min(float(c), q95 + (q95 - q75)),
        )
    )
    target = float(np.clip(np.interp(rng.random(), probabilities, counts), 1, c))
    freq = np.clip(
        np.asarray(cal["poi"]["active_frequency"], dtype=float), 1e-05, 1.0 - 1e-05
    )
    log_odds = np.log(freq / (1.0 - freq))
    lower, upper = (-20.0, 20.0)
    for _ in range(30):
        offset = 0.5 * (lower + upper)
        probabilities_by_channel = 1.0 / (1.0 + np.exp(-(log_odds + offset)))
        if probabilities_by_channel.sum() < target:
            lower = offset
        else:
            upper = offset
    offset = 0.5 * (lower + upper)
    probabilities_by_channel = 1.0 / (1.0 + np.exp(-(log_odds + offset)))
    active = rng.random(c) < probabilities_by_channel
    if not active.any():
        active[int(np.argmax(probabilities_by_channel))] = True
    return active


def _poi_background(
    cal: dict[str, Any],
    active: np.ndarray,
    rng: np.random.Generator,
    record: dict[int, list[tuple[float, float]]] | None = None,
) -> np.ndarray:
    """Scatter the nuisance opportunity background."""
    c = len(active)
    out = np.zeros((c, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    stats = cal["poi"]["nonzero_pixels_when_active"]
    for j in np.flatnonzero(active):
        estimated = max(1, int(round(float(stats[j]["q50"]) / 42.0)))
        n = min(120, int(rng.lognormal(np.log(estimated + 1), 0.25)))
        for row, col in rng.uniform(MARGIN, TILE_SIZE - 1 - MARGIN, size=(n, 2)):
            out[j] = np.maximum(
                out[j], gaussian_splat(out.shape[1:], row, col, POI_SIGMA_PX)
            )
            if record is not None:
                record.setdefault(int(j), []).append((float(row), float(col)))
    return out


def _add_channel_focals(
    poi: np.ndarray,
    active: np.ndarray,
    anchor: tuple[float, float],
    rng: np.random.Generator,
    record: dict[int, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Add one focal opportunity per active channel near ``anchor``."""
    out = poi.copy()
    for j in np.flatnonzero(active):
        row, col = np.asarray(anchor) + rng.normal(0, 3.0, size=2)
        out[j] = np.maximum(
            out[j], gaussian_splat(out.shape[1:], row, col, POI_SIGMA_PX)
        )
        if record is not None:
            record[int(j)] = (float(row), float(col))
    return out


def _rail_context(cal: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    c = len(cal["channels"])
    low = gaussian_filter(rng.random((20, 20)), sigma=2.2)
    common = zoom(low, TILE_SIZE / 20, order=3)[:TILE_SIZE, :TILE_SIZE]
    common -= common.min()
    common /= max(common.max(), 1e-08)
    out = np.empty((c, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    stats = cal["rail"]["positive_median_when_active"]
    for j in range(c):
        median = max(float(stats[j]["q50"]), 0.02)
        out[j] = (common * rng.uniform(1.0, 3.0) * median).astype(np.float32)
    return out


def _tile(
    name: str,
    group: str,
    role: str,
    stops: np.ndarray,
    edges: np.ndarray,
    poi: np.ndarray,
    rail: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, Any],
    control_mask: np.ndarray | None = None,
) -> ScenarioTile:
    meta = dict(metadata)
    meta.update({"scenario_name": name, "scenario_group": group, "role": role})
    return ScenarioTile(stops, edges, rail, poi, mask, control_mask, meta)


def _family_metadata(
    base: dict[str, Any], family: str, seed: int, changed: str
) -> dict[str, Any]:
    return {
        **base,
        "family": family,
        "family_seed": int(seed),
        "controlled_change": changed,
        "tile_size": TILE_SIZE,
        "raw_channels": len(base["channel_names"]),
        "raster_parameters": {
            "stop_sigma_px": STOP_SIGMA_PX,
            "edge_sigma_px": EDGE_SIGMA_PX,
            "poi_sigma_px": POI_SIGMA_PX,
            "soft_cap_max": SOFT_CAP_MAX,
            "soft_cap_beta": SOFT_CAP_BETA,
        },
    }


Point = tuple[float, float]


@dataclass(frozen=True)
class FamilyScene:
    """Latent geometry of one relation family within one generated set."""

    family: str
    corridor: list[Point]
    stop_points: list[Point]
    active_channels: list[int]
    poi_background: dict[int, list[Point]]
    focals: dict[str, dict[int, Point]]
    rail_anchor: Point | None = None
    overlap_corridor: list[Point] | None = None
    stub_corridor: list[Point] | None = None
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class SetScene:
    """Latent geometry of one complete generated set."""

    top_seed: int
    set_index: int
    set_seed: int
    coverage_mode: str
    channel_names: list[str]
    families: dict[str, FamilyScene]


FAMILY_CONDITIONS: dict[str, tuple[str, ...]] = {
    "isolated_stop_leakage": ("connected", "isolated:analytic_zero"),
    "poi_reachability_sensitivity": (
        "reachable",
        "displaced",
        "displaced_far",
        "null_background",
    ),
    "rail_connection_sensitivity": (
        "connected",
        "severed_edges:analytic_zero",
        "severed_rail_stop",
    ),
    "amplitude_response": ("single", "near_overlap"),
    "spatial_delta_localization": ("route",),
    "stub_edge_leakage": ("full_route", "stub:analytic_zero"),
}


def background_only_poi(fam: FamilyScene, n_channels: int = 105) -> np.ndarray:
    """Re-render a family's opportunity background with no focals."""
    out = np.zeros((n_channels, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    for ch, points in fam.poi_background.items():
        for row, col in points:
            out[ch] = np.maximum(
                out[ch], gaussian_splat(out.shape[1:], row, col, POI_SIGMA_PX)
            )
    return out


def build_set_with_scene(
    calibration: dict[str, Any], schema: dict[str, Any], top_seed: int, set_index: int
) -> tuple[dict[str, ScenarioTile], SetScene]:
    """Build one blocked replicate with independent deterministic family streams, plus its latent scene geometry."""
    if len(schema["poi"]) != 105 or schema["poi"] != calibration["channels"]:
        raise ValueError("calibration and frozen 105-channel schema do not align")
    set_seed = int(np.random.SeedSequence([top_seed, set_index]).generate_state(1)[0])
    coverage_mode = "all_channels" if set_index % 5 == 0 else "empirical_density"
    base_meta = {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "top_seed": int(top_seed),
        "set_index": int(set_index),
        "set_seed": set_seed,
        "coverage_mode": coverage_mode,
        "channel_names": list(schema["poi"]),
    }
    seed_seq = np.random.SeedSequence([top_seed, set_index, 659918])
    children = seed_seq.spawn(len(FAMILY_NAMES))
    family_seeds = [int(x.generate_state(1)[0]) for x in children]
    tiles: dict[str, ScenarioTile] = {}
    scenes: dict[str, FamilyScene] = {}

    def context(rng: np.random.Generator, bg_record: dict | None = None):
        active = _coverage_mask(calibration, rng, coverage_mode)
        return (
            active,
            _poi_background(calibration, active, rng, record=bg_record),
            _rail_context(calibration, rng),
        )

    def scene(family: str, **kw) -> None:
        scenes[family] = FamilyScene(
            family=family, conditions=FAMILY_CONDITIONS[family], **kw
        )

    rng = np.random.default_rng(family_seeds[0])
    path = _path(rng)
    endpoints = [path[0], path[-1]]
    bg: dict = {}
    foc: dict = {}
    active, poi_bg, rail = context(rng, bg)
    poi = _add_channel_focals(poi_bg, active, path[-1], rng, record=foc)
    stops = point_splats((TILE_SIZE, TILE_SIZE), endpoints)
    no_edges = np.zeros_like(stops)
    edges = polyline_splat(stops.shape, path)
    mask = _disk_mask(endpoints)
    meta = _family_metadata(
        base_meta, FAMILY_NAMES[0], family_seeds[0], "bus_edges only"
    )
    tiles["isolated_stop_no_poi"] = _tile(
        "isolated_stop_no_poi",
        FAMILY_NAMES[0],
        "null_context",
        stops,
        no_edges,
        poi_bg,
        rail,
        mask,
        meta,
    )
    tiles["isolated_stop_with_poi"] = _tile(
        "isolated_stop_with_poi",
        FAMILY_NAMES[0],
        "isolated",
        stops,
        no_edges,
        poi,
        rail,
        mask,
        meta,
    )
    tiles["connected_stop_with_poi"] = _tile(
        "connected_stop_with_poi",
        FAMILY_NAMES[0],
        "connected",
        stops,
        edges,
        poi,
        rail,
        mask,
        meta,
    )
    scene(
        FAMILY_NAMES[0],
        corridor=list(path),
        stop_points=list(endpoints),
        active_channels=[int(j) for j in np.flatnonzero(active)],
        poi_background=bg,
        focals={"connected": foc, "isolated": foc},
    )
    rng = np.random.default_rng(family_seeds[1])
    path = _path(rng)
    stops_pts = _sample_path(path, 4)
    bg = {}
    foc_near: dict = {}
    foc_far: dict = {}
    active, poi_bg, rail = context(rng, bg)
    far = _far_anchor(rng, path)
    reachable = _add_channel_focals(poi_bg, active, path[-1], rng, record=foc_near)
    state = rng.bit_generator.state
    unreachable = _add_channel_focals(poi_bg, active, far, rng, record=foc_far)
    rng.bit_generator.state = state
    stops = point_splats((TILE_SIZE, TILE_SIZE), stops_pts)
    edges = polyline_splat(stops.shape, path)
    mask = _disk_mask(stops_pts, 17)
    meta = _family_metadata(
        base_meta, FAMILY_NAMES[1], family_seeds[1], "focal POI locations only"
    )
    tiles["route_poi_reachable"] = _tile(
        "route_poi_reachable",
        FAMILY_NAMES[1],
        "reachable",
        stops,
        edges,
        reachable,
        rail,
        mask,
        meta,
    )
    tiles["route_poi_unreachable"] = _tile(
        "route_poi_unreachable",
        FAMILY_NAMES[1],
        "unreachable",
        stops,
        edges,
        unreachable,
        rail,
        mask,
        meta,
    )
    scene(
        FAMILY_NAMES[1],
        corridor=list(path),
        stop_points=list(stops_pts),
        active_channels=[int(j) for j in np.flatnonzero(active)],
        poi_background=bg,
        focals={"reachable": foc_near, "displaced": foc_far},
    )
    rng = np.random.default_rng(family_seeds[2])
    path = _path(rng, rng.uniform(55, 90))
    endpoints = [path[0], path[-1]]
    bg = {}
    active, poi, rail = context(rng, bg)
    transfer = path[-1]
    for j in range(len(active)):
        rail[j] += gaussian_splat(
            (TILE_SIZE, TILE_SIZE),
            *transfer,
            sigma=9.0,
            peak=max(float(rail[j].max()), 0.1),
        )
    stops = point_splats((TILE_SIZE, TILE_SIZE), endpoints)
    edges = polyline_splat(stops.shape, path)
    mask = _disk_mask(endpoints, 17)
    meta = _family_metadata(
        base_meta, FAMILY_NAMES[2], family_seeds[2], "bus_edges only"
    )
    tiles["stop_near_rail_connected"] = _tile(
        "stop_near_rail_connected",
        FAMILY_NAMES[2],
        "connected",
        stops,
        edges,
        poi,
        rail,
        mask,
        meta,
    )
    tiles["stop_near_rail_isolated"] = _tile(
        "stop_near_rail_isolated",
        FAMILY_NAMES[2],
        "isolated",
        stops,
        np.zeros_like(edges),
        poi,
        rail,
        mask,
        meta,
    )
    scene(
        FAMILY_NAMES[2],
        corridor=list(path),
        stop_points=list(endpoints),
        active_channels=[int(j) for j in np.flatnonzero(active)],
        poi_background=bg,
        focals={},
        rail_anchor=(float(transfer[0]), float(transfer[1])),
    )
    rng = np.random.default_rng(family_seeds[3])
    path = _path(rng)
    stops_pts = _sample_path(path, 5)
    bg = {}
    foc = {}
    active, poi_bg, rail = context(rng, bg)
    poi = _add_channel_focals(poi_bg, active, path[-1], rng, record=foc)
    stops = point_splats((TILE_SIZE, TILE_SIZE), stops_pts)
    edges = polyline_splat(stops.shape, path)
    mask = _disk_mask(stops_pts, 16)
    offset = [(r + 1.25, c - 1.25) for r, c in path]
    redundant_edges = _soft_cap(edges + polyline_splat(edges.shape, offset))
    meta = _family_metadata(
        base_meta,
        FAMILY_NAMES[3],
        family_seeds[3],
        "bus amplitude or near-overlap only",
    )
    tiles["scaled_bus_full"] = _tile(
        "scaled_bus_full",
        FAMILY_NAMES[3],
        "reference",
        stops,
        edges,
        poi,
        rail,
        mask,
        meta,
    )
    tiles["route_single_line"] = _tile(
        "route_single_line",
        FAMILY_NAMES[3],
        "single",
        stops,
        edges,
        poi,
        rail,
        mask,
        meta,
    )
    tiles["route_redundant_overlap"] = _tile(
        "route_redundant_overlap",
        FAMILY_NAMES[3],
        "near_overlap",
        stops,
        redundant_edges,
        poi,
        rail,
        mask,
        meta,
    )
    scene(
        FAMILY_NAMES[3],
        corridor=list(path),
        stop_points=list(stops_pts),
        active_channels=[int(j) for j in np.flatnonzero(active)],
        poi_background=bg,
        focals={"single": foc, "near_overlap": foc},
        overlap_corridor=list(offset),
    )
    rng = np.random.default_rng(family_seeds[4])
    path = _path(rng)
    stops_pts = _sample_path(path, 5)
    bg = {}
    foc = {}
    active, poi_bg, rail = context(rng, bg)
    poi = _add_channel_focals(poi_bg, active, path[-1], rng, record=foc)
    stops = point_splats((TILE_SIZE, TILE_SIZE), stops_pts)
    edges = polyline_splat(stops.shape, path)
    near, far_mask = _route_masks(edges)
    meta = _family_metadata(
        base_meta, FAMILY_NAMES[4], family_seeds[4], "evaluation region only"
    )
    tiles["localized_route"] = _tile(
        "localized_route",
        FAMILY_NAMES[4],
        "route_vs_far",
        stops,
        edges,
        poi,
        rail,
        near,
        meta,
        far_mask,
    )
    scene(
        FAMILY_NAMES[4],
        corridor=list(path),
        stop_points=list(stops_pts),
        active_channels=[int(j) for j in np.flatnonzero(active)],
        poi_background=bg,
        focals={"route": foc},
    )
    rng = np.random.default_rng(family_seeds[5])
    path = _path(rng)
    endpoints = [path[0], path[-1]]
    bg = {}
    foc = {}
    active, poi_bg, rail = context(rng, bg)
    poi = _add_channel_focals(poi_bg, active, path[-1], rng, record=foc)
    stops = point_splats((TILE_SIZE, TILE_SIZE), endpoints)
    full = polyline_splat(stops.shape, path)
    p = np.asarray(path)
    stub_end = tuple(p[0] + 0.13 * (p[1] - p[0]))
    stub = polyline_splat(stops.shape, [path[0], stub_end])
    mask = _disk_mask(endpoints, 17)
    meta = _family_metadata(
        base_meta, FAMILY_NAMES[5], family_seeds[5], "bus_edges only"
    )
    tiles["stub_edge_origin_only"] = _tile(
        "stub_edge_origin_only",
        FAMILY_NAMES[5],
        "stub",
        stops,
        stub,
        poi,
        rail,
        mask,
        meta,
    )
    tiles["stub_edge_full_route"] = _tile(
        "stub_edge_full_route",
        FAMILY_NAMES[5],
        "full_route",
        stops,
        full,
        poi,
        rail,
        mask,
        meta,
    )
    scene(
        FAMILY_NAMES[5],
        corridor=list(path),
        stop_points=list(endpoints),
        active_channels=[int(j) for j in np.flatnonzero(active)],
        poi_background=bg,
        focals={"full_route": foc, "stub": foc},
        stub_corridor=[tuple(path[0]), tuple(stub_end)],
    )
    return (
        tiles,
        SetScene(
            top_seed=int(top_seed),
            set_index=int(set_index),
            set_seed=set_seed,
            coverage_mode=coverage_mode,
            channel_names=list(schema["poi"]),
            families=scenes,
        ),
    )


def validate_set(tiles: dict[str, ScenarioTile], channels: int = 105) -> list[str]:
    errors: list[str] = []
    for name, tile in tiles.items():
        if tile.bus_stops.shape != (TILE_SIZE, TILE_SIZE):
            errors.append(f"{name}: bus shape")
        if tile.poi_raw.shape != (channels, TILE_SIZE, TILE_SIZE):
            errors.append(f"{name}: POI shape")
        if tile.rail_raw.shape != (channels, TILE_SIZE, TILE_SIZE):
            errors.append(f"{name}: rail shape")
        border = np.r_[
            tile.bus_stops[:MARGIN].ravel(),
            tile.bus_stops[-MARGIN:].ravel(),
            tile.bus_stops[:, :MARGIN].ravel(),
            tile.bus_stops[:, -MARGIN:].ravel(),
            tile.bus_edges[:MARGIN].ravel(),
            tile.bus_edges[-MARGIN:].ravel(),
            tile.bus_edges[:, :MARGIN].ravel(),
            tile.bus_edges[:, -MARGIN:].ravel(),
        ]
        if np.any(border > 1e-06):
            errors.append(f"{name}: bus feature touches safety margin")
        poi_border = np.concatenate(
            (
                tile.poi_raw[:, 0, :].ravel(),
                tile.poi_raw[:, -1, :].ravel(),
                tile.poi_raw[:, :, 0].ravel(),
                tile.poi_raw[:, :, -1].ravel(),
            )
        )
        if np.any(poi_border > 1e-06):
            errors.append(f"{name}: POI feature touches tile border")
    pairs = [
        ("isolated_stop_with_poi", "connected_stop_with_poi", ("bus_edges",)),
        ("route_poi_reachable", "route_poi_unreachable", ("poi_raw",)),
        ("stop_near_rail_connected", "stop_near_rail_isolated", ("bus_edges",)),
        ("stub_edge_origin_only", "stub_edge_full_route", ("bus_edges",)),
    ]
    fields = ("bus_stops", "bus_edges", "poi_raw", "rail_raw", "eval_mask")
    for a, b, changed in pairs:
        for field in fields:
            equal = np.array_equal(getattr(tiles[a], field), getattr(tiles[b], field))
            if field in changed and equal:
                errors.append(f"{a}/{b}: {field} did not change")
            if field not in changed and (not equal):
                errors.append(f"{a}/{b}: nuisance {field} changed")
    loc = tiles["localized_route"]
    if loc.control_mask is None or loc.eval_mask.sum() != loc.control_mask.sum():
        errors.append("localized_route: comparison masks are not equal-area")
    elif np.any(loc.eval_mask & loc.control_mask):
        errors.append("localized_route: comparison masks overlap")
    return errors


def build_evaluation_set(calibration, schema, top_seed, set_index):
    """Return the 13 paired scenes plus the background-only opportunity control."""
    tiles, scene = build_set_with_scene(calibration, schema, top_seed, set_index)
    family = scene.families["poi_reachability_sensitivity"]
    eligible = np.zeros(len(schema["poi"]), dtype=bool)
    eligible[list(family.focals["reachable"])] = True
    reference = tiles["route_poi_reachable"]
    tiles["opportunity_background"] = replace(
        reference,
        poi_raw=background_only_poi(family),
        metadata={**reference.metadata, "opportunity_eligible": eligible.tolist()},
    )
    return tiles


def write_set(tiles: dict[str, ScenarioTile], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, tile in sorted(tiles.items()):
        path = out_dir / f"{name}.npz"
        save_scenario(tile, path, compressed=True)
        files.append({"name": name, "file": path.name, "sha256": sha256_file(path)})
    return {"directory": out_dir.name, "n_scenarios": len(files), "files": files}


__all__ = [
    "FORMAT",
    "GENERATOR_VERSION",
    "TILE_SIZE",
    "build_evaluation_set",
    "build_set_with_scene",
    "FamilyScene",
    "SetScene",
    "background_only_poi",
    "validate_set",
    "write_set",
    "gaussian_splat",
    "line_splat",
    "polyline_splat",
    "sha256_file",
]
