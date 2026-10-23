"""Tile Graph Specification builder."""

from __future__ import annotations
import warnings
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


def _install_numpy_compat_shim() -> None:
    """Register numpy._core aliases so pickles created with NumPy ≥2.0 can be
    loaded in environments that still ship NumPy 1.x (e.g. PyTorch 24.01
    container).  Safe to call multiple times; is a no-op on NumPy 2.x."""
    if hasattr(np, "_core"):
        return
    import numpy.core as _nc

    _aliases = [
        ("numpy._core", _nc),
        ("numpy._core.multiarray", _nc.multiarray),
        ("numpy._core.numeric", _nc.numeric),
        ("numpy._core.fromnumeric", _nc.fromnumeric),
        ("numpy._core.arrayprint", _nc.arrayprint),
        ("numpy._core._methods", getattr(_nc, "_methods", _nc)),
        ("numpy._core.umath", _nc.umath),
        ("numpy._core.numerictypes", _nc.numerictypes),
        ("numpy._core.shape_base", _nc.shape_base),
    ]
    for mod_name, mod in _aliases:
        sys.modules.setdefault(mod_name, mod)


from scipy.spatial import KDTree
import networkx as nx

try:
    import geopandas as gpd
except ImportError:
    gpd = None
from modules.training_data import _gaussian_splat_pixels


@dataclass
class TileGraphSpec:
    """Precomputed candidate stop/edge graph and renderer contribution arrays for one tile."""

    stop_xy: np.ndarray
    stop_node_ids: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_length: np.ndarray
    edge_path_xy: Tuple[np.ndarray, ...]
    stop_pixel_idx: np.ndarray
    stop_stop_idx: np.ndarray
    stop_weight: np.ndarray
    edge_pixel_idx: np.ndarray
    edge_edge_idx: np.ndarray
    edge_weight: np.ndarray
    H: int
    W: int
    bounds: Tuple[float, float, float, float]
    crs: str
    candidate_edge_mode: str = "road_segments"
    routing_mode: str = "airline"

    def num_stops(self) -> int:
        return len(self.stop_xy)

    def num_edges(self) -> int:
        return len(self.edge_u)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "TileGraphSpec":
        _install_numpy_compat_shim()
        with open(path, "rb") as f:
            return pickle.load(f)


def _xy_to_pixel(
    x: np.ndarray, y: np.ndarray, bounds: tuple, H: int, W: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert projected (x, y) to fractional pixel (col_f, row_f)."""
    minx, miny, maxx, maxy = bounds
    pixel_w = (maxx - minx) / W
    pixel_h = (maxy - miny) / H
    col_f = (x - minx) / pixel_w
    row_f = (maxy - y) / pixel_h
    return (col_f, row_f)


def _line_splat_pixels(
    col0: float, row0: float, col1: float, row1: float, H: int, W: int, sigma_px: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel indices and accumulated weights for a Gaussian-blurred line segment."""
    length_px = np.sqrt((col1 - col0) ** 2 + (row1 - row0) ** 2)
    n_samples = max(2, int(np.ceil(length_px * 2)))
    t = np.linspace(0, 1, n_samples)
    sample_cols = col0 + t * (col1 - col0)
    sample_rows = row0 + t * (row1 - row0)
    radius_px = int(np.ceil(3.0 * sigma_px))
    dr_range = np.arange(-radius_px, radius_px + 1)
    dc_range = np.arange(-radius_px, radius_px + 1)
    dr, dc = np.meshgrid(dr_range, dc_range, indexing="ij")
    dr = dr.ravel()
    dc = dc.ravel()
    all_pix: list = []
    all_w: list = []
    for sc, sr in zip(sample_cols, sample_rows):
        pr = int(np.floor(sr)) + dr
        pc = int(np.floor(sc)) + dc
        dist_sq = (pc - sc) ** 2 + (pr - sr) ** 2
        w = np.exp(-0.5 * dist_sq / sigma_px**2).astype(np.float32)
        valid = (pr >= 0) & (pr < H) & (pc >= 0) & (pc < W) & (w > 1e-06)
        if np.any(valid):
            all_pix.append((pr[valid] * W + pc[valid]).astype(np.int64))
            all_w.append(w[valid])
    if not all_pix:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    flat_pix = np.concatenate(all_pix)
    flat_w = np.concatenate(all_w)
    unique_pix, inverse = np.unique(flat_pix, return_inverse=True)
    acc_w = np.bincount(
        inverse, weights=flat_w.astype(float), minlength=len(unique_pix)
    )
    out_w = (acc_w / n_samples).astype(np.float32)
    if out_w.size:
        w_max = float(out_w.max())
        if w_max > 0:
            out_w = (out_w / w_max).astype(np.float32)
    return (unique_pix.astype(np.int64), out_w)


def _multiline_splat_pixels(
    coords_list: list, H: int, W: int, sigma_px: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Splat multiple line segments (from a path) along actual road geometry."""
    if len(coords_list) < 2:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    all_pix: list = []
    all_w: list = []
    for i in range(len(coords_list) - 1):
        col0, row0 = coords_list[i]
        col1, row1 = coords_list[i + 1]
        pix_idx, w = _line_splat_pixels(col0, row0, col1, row1, H, W, sigma_px)
        if len(pix_idx) > 0:
            all_pix.append(pix_idx)
            all_w.append(w)
    if not all_pix:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    flat_pix = np.concatenate(all_pix)
    flat_w = np.concatenate(all_w)
    unique_pix, inverse = np.unique(flat_pix, return_inverse=True)
    acc_w = np.bincount(
        inverse, weights=flat_w.astype(float), minlength=len(unique_pix)
    )
    out_w = (acc_w / len(coords_list)).astype(np.float32)
    if out_w.size:
        w_max = float(out_w.max())
        if w_max > 0:
            out_w = (out_w / w_max).astype(np.float32)
    return (unique_pix.astype(np.int64), out_w)


def _geometry_splat_pixels(
    geom, bounds: tuple, H: int, W: int, sigma_px: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Splat a LineString/MultiLineString geometry in projected CRS."""
    if geom is None:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    if geom.geom_type == "LineString":
        geoms = [geom]
    elif geom.geom_type == "MultiLineString":
        geoms = list(geom.geoms)
    else:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    all_pix: list = []
    all_w: list = []
    for part in geoms:
        coords = list(part.coords)
        if len(coords) < 2:
            continue
        xs = np.array([c[0] for c in coords], dtype=np.float64)
        ys = np.array([c[1] for c in coords], dtype=np.float64)
        cols, rows = _xy_to_pixel(xs, ys, bounds, H, W)
        path_coords = [(float(c), float(r)) for c, r in zip(cols, rows)]
        pix_idx, w = _multiline_splat_pixels(path_coords, H, W, sigma_px)
        if len(pix_idx):
            all_pix.append(pix_idx)
            all_w.append(w)
    if not all_pix:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    flat_pix = np.concatenate(all_pix)
    flat_w = np.concatenate(all_w)
    unique_pix, inverse = np.unique(flat_pix, return_inverse=True)
    acc_w = np.bincount(
        inverse, weights=flat_w.astype(float), minlength=len(unique_pix)
    )
    out_w = acc_w.astype(np.float32)
    if out_w.size:
        w_max = float(out_w.max())
        if w_max > 0:
            out_w = (out_w / w_max).astype(np.float32)
    return (unique_pix.astype(np.int64), out_w)


def _geometry_to_xy_path(geom) -> np.ndarray:
    """Extract projected polyline coordinates from line geometry."""
    if geom is None:
        return np.zeros((0, 2), dtype=np.float32)
    if geom.geom_type == "LineString":
        coords = np.asarray(geom.coords, dtype=np.float32)
        return (
            coords
            if coords.ndim == 2 and coords.shape[1] == 2
            else np.zeros((0, 2), dtype=np.float32)
        )
    if geom.geom_type == "MultiLineString":
        parts = []
        for line in geom.geoms:
            arr = np.asarray(line.coords, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 2 and (len(arr) >= 2):
                if parts and np.allclose(parts[-1][-1], arr[0]):
                    parts.append(arr[1:])
                else:
                    parts.append(arr)
        if parts:
            return np.concatenate(parts, axis=0)
    return np.zeros((0, 2), dtype=np.float32)


def _build_road_xy_lookup(node_source) -> dict:
    road_xy_lookup = {}
    if node_source is None or node_source.empty:
        return road_xy_lookup
    id_col = "id" if "id" in node_source.columns else None
    if id_col is not None:
        for _, nrow in node_source[["id", "geometry"]].iterrows():
            road_xy_lookup[nrow["id"]] = (
                float(nrow.geometry.x),
                float(nrow.geometry.y),
            )
    else:
        for nid, nrow in node_source[["geometry"]].iterrows():
            road_xy_lookup[nid] = (float(nrow.geometry.x), float(nrow.geometry.y))
    return road_xy_lookup


def _map_gtfs_to_candidate_pairs(
    spec: TileGraphSpec,
    gtfs_nodes_proj: Optional[gpd.GeoDataFrame],
    gtfs_edges_proj: Optional[gpd.GeoDataFrame],
    *,
    snap_radius_m: float,
) -> Tuple[np.ndarray, dict, dict, dict]:
    """Map GTFS stops to candidate stops and count GTFS stop-pair traversals."""
    n_stops = spec.num_stops()
    z_real = np.zeros(n_stops, dtype=np.float32)
    pair_counts: dict = {}
    gtfs_id_to_cand: dict = {}
    stats = {"active_gtfs_stops": 0, "snapped_gtfs_stops": 0, "dropped_gtfs_stops": 0}
    if n_stops == 0 or gtfs_nodes_proj is None or gtfs_nodes_proj.empty:
        return (z_real, gtfs_id_to_cand, pair_counts, stats)
    stop_tree = KDTree(spec.stop_xy.astype(np.float64))
    gtfs_active_ids = None
    if gtfs_edges_proj is not None and (not gtfs_edges_proj.empty):
        if "u" in gtfs_edges_proj.columns and "v" in gtfs_edges_proj.columns:
            gtfs_active_ids = set(gtfs_edges_proj["u"].tolist()) | set(
                gtfs_edges_proj["v"].tolist()
            )
    gtfs_xy = np.column_stack(
        [gtfs_nodes_proj.geometry.x.values, gtfs_nodes_proj.geometry.y.values]
    )
    if "id" in gtfs_nodes_proj.columns:
        gtfs_ids = gtfs_nodes_proj["id"].values
    else:
        gtfs_ids = gtfs_nodes_proj.index.values
    if gtfs_active_ids is not None:
        keep_mask = np.array([gid in gtfs_active_ids for gid in gtfs_ids], dtype=bool)
        gtfs_xy = gtfs_xy[keep_mask]
        gtfs_ids = gtfs_ids[keep_mask]
    stats["active_gtfs_stops"] = int(len(gtfs_ids))
    dists, cand_idx = stop_tree.query(gtfs_xy, k=1)
    for gid, dist, cidx in zip(gtfs_ids, dists, cand_idx):
        if dist <= snap_radius_m:
            z_real[int(cidx)] = 1.0
            gtfs_id_to_cand[gid] = int(cidx)
        else:
            stats["dropped_gtfs_stops"] += 1
    stats["snapped_gtfs_stops"] = int(len(gtfs_id_to_cand))
    if gtfs_edges_proj is None or gtfs_edges_proj.empty:
        return (z_real, gtfs_id_to_cand, pair_counts, stats)
    for _, row in gtfs_edges_proj.iterrows():
        u_id = row.get("u")
        v_id = row.get("v")
        u_cand = gtfs_id_to_cand.get(u_id)
        v_cand = gtfs_id_to_cand.get(v_id)
        if u_cand is None or v_cand is None or u_cand == v_cand:
            continue
        key = (int(u_cand), int(v_cand))
        pair_counts[key] = pair_counts.get(key, 0.0) + 1.0
    return (z_real, gtfs_id_to_cand, pair_counts, stats)


def _resolve_path_weight(value: float, weight_mode: str) -> float:
    mode = str(weight_mode).strip().lower()
    if mode == "count":
        return float(value)
    if mode == "binary":
        return 1.0 if value > 0 else 0.0
    if mode == "global_binary":
        return 1.0 if value > 0 else 0.0
    if mode == "log1p":
        return float(np.log1p(value))
    raise ValueError(
        f"Unsupported BUS_PATH_RENDER_WEIGHT_MODE '{weight_mode}'. Expected 'count', 'binary', 'global_binary', or 'log1p'."
    )


def _normalize_bus_path_routing_direction(value: str | None) -> str:
    mode = str(value or "undirected").strip().lower()
    aliases = {
        "undirected": "undirected",
        "bidirectional": "undirected",
        "directed": "directed",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported BUS_PATH_ROUTING_DIRECTION '{value}'. Expected 'undirected' or 'directed'."
        )
    return aliases[mode]


def _resolve_bus_path_routing_graph(
    road_graph: nx.MultiDiGraph, routing_direction: str
) -> nx.MultiDiGraph | nx.MultiGraph:
    mode = _normalize_bus_path_routing_direction(routing_direction)
    if mode == "undirected":
        return road_graph.to_undirected(as_view=False)
    return road_graph


def _road_component_lookup(road_graph: nx.Graph) -> dict:
    if road_graph.is_directed():
        comps = nx.weakly_connected_components(road_graph)
    else:
        comps = nx.connected_components(road_graph)
    lookup = {}
    for comp_idx, nodes in enumerate(comps):
        for node in nodes:
            lookup[node] = comp_idx
    return lookup


def _failure_reason(
    road_graph: nx.Graph, component_lookup: dict | None, road_u_id: int, road_v_id: int
) -> str:
    if not road_graph.has_node(road_u_id) or not road_graph.has_node(road_v_id):
        return "missing_node"
    if component_lookup is None:
        return "no_path"
    comp_u = component_lookup.get(road_u_id)
    comp_v = component_lookup.get(road_v_id)
    if comp_u is None or comp_v is None:
        return "no_path"
    if comp_u != comp_v:
        return "different_weak_component"
    if road_graph.is_directed():
        return "same_component_no_directed_path"
    return "same_component_no_path"


def render_gtfs_paths_to_grid(
    spec: TileGraphSpec,
    gtfs_nodes_proj: Optional[gpd.GeoDataFrame],
    gtfs_edges_proj: Optional[gpd.GeoDataFrame],
    road_graph: Optional[nx.MultiDiGraph],
    road_nodes_lookup_proj: Optional[gpd.GeoDataFrame] = None,
    *,
    snap_radius_m: float = 300.0,
    sigma_px: float = 1.5,
    weight_mode: str = "count",
    routing_direction: str = "undirected",
) -> Tuple[np.ndarray, dict]:
    """Render merged GTFS stop-to-stop routed paths directly to a tile edge image."""
    edge_img = np.zeros((spec.H, spec.W), dtype=np.float32)
    routing_direction = _normalize_bus_path_routing_direction(routing_direction)
    stats = {
        "pair_count": 0,
        "routed_pairs": 0,
        "no_path_pairs": 0,
        "missing_node_pairs": 0,
        "short_path_pairs": 0,
        "active_pixels": 0,
        "weight_mode": str(weight_mode).strip().lower(),
        "routing_direction": routing_direction,
        "snap_stats": {},
        "failure_examples": [],
    }
    if (
        spec.num_stops() == 0
        or gtfs_edges_proj is None
        or gtfs_edges_proj.empty
        or (road_graph is None)
    ):
        return (edge_img, stats)
    _, _, pair_counts, snap_stats = _map_gtfs_to_candidate_pairs(
        spec, gtfs_nodes_proj, gtfs_edges_proj, snap_radius_m=snap_radius_m
    )
    stats["pair_count"] = len(pair_counts)
    stats["snap_stats"] = snap_stats
    if not pair_counts:
        return (edge_img, stats)
    road_xy_lookup = _build_road_xy_lookup(road_nodes_lookup_proj)
    if not road_xy_lookup:
        return (edge_img, stats)
    routing_graph = _resolve_bus_path_routing_graph(road_graph, routing_direction)
    flat = np.zeros(spec.H * spec.W, dtype=np.float32)
    component_lookup = None
    for (u_cand, v_cand), cnt in pair_counts.items():
        ru = int(spec.stop_node_ids[int(u_cand)])
        rv = int(spec.stop_node_ids[int(v_cand)])
        if not routing_graph.has_node(ru) or not routing_graph.has_node(rv):
            stats["missing_node_pairs"] += 1
            if len(stats["failure_examples"]) < 5:
                stats["failure_examples"].append(
                    {
                        "reason": "missing_node",
                        "count": float(cnt),
                        "candidate_u": int(u_cand),
                        "candidate_v": int(v_cand),
                        "road_u_id": ru,
                        "road_v_id": rv,
                        "straight_line_m": round(
                            float(
                                np.linalg.norm(
                                    spec.stop_xy[int(u_cand)]
                                    - spec.stop_xy[int(v_cand)]
                                )
                            ),
                            1,
                        ),
                    }
                )
            continue
        try:
            path = nx.shortest_path(routing_graph, ru, rv, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            stats["no_path_pairs"] += 1
            if component_lookup is None:
                component_lookup = _road_component_lookup(routing_graph)
            if len(stats["failure_examples"]) < 5:
                stats["failure_examples"].append(
                    {
                        "reason": _failure_reason(
                            routing_graph, component_lookup, ru, rv
                        ),
                        "count": float(cnt),
                        "candidate_u": int(u_cand),
                        "candidate_v": int(v_cand),
                        "road_u_id": ru,
                        "road_v_id": rv,
                        "component_u": component_lookup.get(ru),
                        "component_v": component_lookup.get(rv),
                        "straight_line_m": round(
                            float(
                                np.linalg.norm(
                                    spec.stop_xy[int(u_cand)]
                                    - spec.stop_xy[int(v_cand)]
                                )
                            ),
                            1,
                        ),
                    }
                )
            continue
        path_coords_proj = []
        for nid in path:
            xy = road_xy_lookup.get(nid)
            if xy is not None:
                path_coords_proj.append(xy)
        if len(path_coords_proj) < 2:
            stats["short_path_pairs"] += 1
            continue
        xs = np.asarray([x for x, _ in path_coords_proj], dtype=np.float64)
        ys = np.asarray([y for _, y in path_coords_proj], dtype=np.float64)
        cols, rows = _xy_to_pixel(xs, ys, spec.bounds, spec.H, spec.W)
        path_coords_pixel = [(float(col), float(row)) for col, row in zip(cols, rows)]
        pix_idx, w = _multiline_splat_pixels(
            path_coords_pixel, spec.H, spec.W, sigma_px
        )
        if len(pix_idx) == 0:
            stats["short_path_pairs"] += 1
            continue
        contrib = (_resolve_path_weight(float(cnt), weight_mode) * w).astype(np.float32)
        if stats["weight_mode"] == "global_binary":
            flat[pix_idx] = np.maximum(flat[pix_idx], contrib)
        else:
            flat[pix_idx] += contrib
        stats["routed_pairs"] += 1
    edge_img = flat.reshape(spec.H, spec.W)
    stats["active_pixels"] = int(np.count_nonzero(edge_img > 0))
    if (
        stats["no_path_pairs"]
        or stats["missing_node_pairs"]
        or stats["short_path_pairs"]
    ):
        warnings.warn(
            "GTFS routed-path render diagnostics (%s): pairs=%d routed=%d no_path=%d missing_node=%d short_path=%d snap=%s examples=%s"
            % (
                routing_direction,
                stats["pair_count"],
                stats["routed_pairs"],
                stats["no_path_pairs"],
                stats["missing_node_pairs"],
                stats["short_path_pairs"],
                stats["snap_stats"],
                stats["failure_examples"],
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    return (edge_img, stats)


def build_tile_graph_spec(
    grid_spec: dict,
    road_nodes_proj: gpd.GeoDataFrame,
    road_edges_proj: gpd.GeoDataFrame,
    gtfs_nodes_proj: Optional[gpd.GeoDataFrame] = None,
    gtfs_edges_proj: Optional[gpd.GeoDataFrame] = None,
    road_graph: Optional[nx.MultiDiGraph] = None,
    road_nodes_lookup_proj: Optional[gpd.GeoDataFrame] = None,
    *,
    log_context: str = "",
    candidate_edge_mode: str = "road_segments",
    snap_radius_m: float = 200.0,
    neighbor_k: int = 6,
    max_stops: int = 1500,
    stop_sigma_px: float = 2.0,
    edge_sigma_px: float = 1.5,
    routing_mode: str = "airline",
) -> TileGraphSpec:
    """Build a TileGraphSpec for one tile."""
    if routing_mode == "road_shortest_path" and road_graph is None:
        raise ValueError(
            "routing_mode='road_shortest_path' requires road_graph parameter to be provided."
        )
    if routing_mode not in ("airline", "road_shortest_path"):
        raise ValueError(
            f"Unknown routing_mode '{routing_mode}'. Supported: 'airline', 'road_shortest_path'."
        )
    if candidate_edge_mode not in ("road_segments", "gtfs_pairs"):
        raise ValueError("candidate_edge_mode must be 'road_segments' or 'gtfs_pairs'")
    bounds = grid_spec["bounds"]
    H = grid_spec["height"]
    W = grid_spec["width"]
    crs = grid_spec["crs"]
    if road_nodes_proj is None or road_nodes_proj.empty:
        warnings.warn(
            "No road nodes supplied; returning empty TileGraphSpec",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_spec(H, W, bounds, crs)
    road_xy = np.column_stack(
        [road_nodes_proj.geometry.x.values, road_nodes_proj.geometry.y.values]
    ).astype(np.float64)
    if "id" in road_nodes_proj.columns:
        road_node_ids = road_nodes_proj["id"].values
    else:
        road_node_ids = road_nodes_proj.index.values
    degrees: dict = {}
    if road_edges_proj is not None and (not road_edges_proj.empty):
        for col in ("u", "v"):
            if col in road_edges_proj.columns:
                for nid in road_edges_proj[col].values:
                    degrees[nid] = degrees.get(nid, 0) + 1
    intersection_mask = np.array(
        [degrees.get(nid, 0) >= 3 for nid in road_node_ids], dtype=bool
    )
    intersection_indices = np.where(intersection_mask)[0].tolist()
    gtfs_active_ids = None
    if gtfs_edges_proj is not None and (not gtfs_edges_proj.empty):
        if "u" in gtfs_edges_proj.columns and "v" in gtfs_edges_proj.columns:
            gtfs_active_ids = set(gtfs_edges_proj["u"].tolist()) | set(
                gtfs_edges_proj["v"].tolist()
            )
    if candidate_edge_mode == "road_segments":
        final_indices = list(range(len(road_node_ids)))
        n_gtfs = 0
    else:
        gtfs_snapped_set: set = set()
        if gtfs_nodes_proj is not None and (not gtfs_nodes_proj.empty):
            if "id" in gtfs_nodes_proj.columns:
                gtfs_ids = gtfs_nodes_proj["id"].values
            else:
                gtfs_ids = gtfs_nodes_proj.index.values
            if gtfs_active_ids is not None:
                keep_mask = np.array(
                    [gid in gtfs_active_ids for gid in gtfs_ids], dtype=bool
                )
            else:
                keep_mask = np.ones(len(gtfs_ids), dtype=bool)
            gtfs_xy = np.column_stack(
                [
                    gtfs_nodes_proj.geometry.x.values[keep_mask],
                    gtfs_nodes_proj.geometry.y.values[keep_mask],
                ]
            )
            road_tree = KDTree(road_xy)
            if len(gtfs_xy) > 0:
                dists, nn_idx = road_tree.query(gtfs_xy, k=1)
                for dist, ridx in zip(dists, nn_idx):
                    if dist <= snap_radius_m:
                        gtfs_snapped_set.add(int(ridx))
        n_gtfs = len(gtfs_snapped_set)
        gtfs_snapped_list = sorted(gtfs_snapped_set)
        intersection_only = [
            i for i in intersection_indices if i not in gtfs_snapped_set
        ]
        n_budget = max_stops - n_gtfs
        if n_budget <= 0:
            sampled_intersections: list = []
        elif len(intersection_only) > n_budget:
            step = len(intersection_only) / n_budget
            sampled_intersections = [
                intersection_only[int(j * step)] for j in range(n_budget)
            ]
        else:
            sampled_intersections = intersection_only
        seen: set = set()
        final_indices = []
        for i in sampled_intersections + gtfs_snapped_list:
            if i not in seen:
                seen.add(i)
                final_indices.append(i)
    if not final_indices:
        warnings.warn(
            "No candidate stops found in tile; returning empty TileGraphSpec",
            RuntimeWarning,
            stacklevel=2,
        )
        return _empty_spec(H, W, bounds, crs)
    stop_xy = road_xy[final_indices].astype(np.float32)
    stop_node_ids = road_node_ids[final_indices]
    n_stops = len(stop_xy)
    edge_dict: dict = {}
    edge_geom_dict: dict = {}
    if candidate_edge_mode == "road_segments":
        node_to_idx = {nid: i for i, nid in enumerate(stop_node_ids)}
        if road_edges_proj is not None and (not road_edges_proj.empty):
            for _, row in road_edges_proj.iterrows():
                u = row.get("u")
                v = row.get("v")
                geom = row.get("geometry")
                if u not in node_to_idx or v not in node_to_idx or geom is None:
                    continue
                u_idx = int(node_to_idx[u])
                v_idx = int(node_to_idx[v])
                if u_idx == v_idx:
                    continue
                key_idx = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
                try:
                    seg_len = float(geom.length)
                except Exception:
                    continue
                old_len = edge_dict.get(key_idx)
                if old_len is None or seg_len < old_len:
                    edge_dict[key_idx] = seg_len
                    edge_geom_dict[key_idx] = geom
    else:
        stop_tree = KDTree(stop_xy.astype(np.float64))
        gtfs_id_to_cand: dict = {}
        if gtfs_nodes_proj is not None and (not gtfs_nodes_proj.empty):
            gtfs_xy = np.column_stack(
                [gtfs_nodes_proj.geometry.x.values, gtfs_nodes_proj.geometry.y.values]
            )
            if "id" in gtfs_nodes_proj.columns:
                gtfs_ids = gtfs_nodes_proj["id"].values
            else:
                gtfs_ids = gtfs_nodes_proj.index.values
            if gtfs_active_ids is not None:
                keep_mask = np.array(
                    [gid in gtfs_active_ids for gid in gtfs_ids], dtype=bool
                )
                gtfs_xy = gtfs_xy[keep_mask]
                gtfs_ids = gtfs_ids[keep_mask]
            dists, cand_idx = stop_tree.query(gtfs_xy, k=1)
            for gid, dist, cidx in zip(gtfs_ids, dists, cand_idx):
                if dist <= snap_radius_m:
                    gtfs_id_to_cand[gid] = int(cidx)
        if (
            gtfs_edges_proj is not None
            and (not gtfs_edges_proj.empty)
            and gtfs_id_to_cand
        ):
            for _, row in gtfs_edges_proj.iterrows():
                u_id = row.get("u")
                v_id = row.get("v")
                u_cand = gtfs_id_to_cand.get(u_id)
                v_cand = gtfs_id_to_cand.get(v_id)
                if u_cand is None or v_cand is None or u_cand == v_cand:
                    continue
                u_idx, v_idx = (u_cand, v_cand) if u_cand < v_cand else (v_cand, u_cand)
                dx = float(stop_xy[u_idx, 0] - stop_xy[v_idx, 0])
                dy = float(stop_xy[u_idx, 1] - stop_xy[v_idx, 1])
                edge_dict[u_idx, v_idx] = float(np.hypot(dx, dy))
        else:
            k_query = min(neighbor_k + 1, n_stops)
            dists_knn, nn_indices = stop_tree.query(
                stop_xy.astype(np.float64), k=k_query
            )
            for i in range(n_stops):
                for j_pos in range(1, k_query):
                    j = int(nn_indices[i, j_pos])
                    if j == i:
                        continue
                    u_idx, v_idx = (i, j) if i < j else (j, i)
                    length = float(dists_knn[i, j_pos])
                    if (u_idx, v_idx) not in edge_dict:
                        edge_dict[u_idx, v_idx] = length
    if not edge_dict:
        edge_u_arr = np.array([], dtype=np.int64)
        edge_v_arr = np.array([], dtype=np.int64)
        edge_length_arr = np.array([], dtype=np.float32)
        edge_geom_arr = []
    else:
        edges_sorted = sorted(edge_dict.items())
        edge_u_arr = np.array([e[0] for e, _ in edges_sorted], dtype=np.int64)
        edge_v_arr = np.array([e[1] for e, _ in edges_sorted], dtype=np.int64)
        edge_length_arr = np.array([ln for _, ln in edges_sorted], dtype=np.float32)
        edge_geom_arr = [edge_geom_dict.get(e) for e, _ in edges_sorted]
    n_edges = len(edge_u_arr)
    stop_col_f, stop_row_f = _xy_to_pixel(stop_xy[:, 0], stop_xy[:, 1], bounds, H, W)
    sp_pix, sp_stop, sp_w = ([], [], [])
    for v_idx in range(n_stops):
        pix_idx, w = _gaussian_splat_pixels(
            float(stop_col_f[v_idx]), float(stop_row_f[v_idx]), H, W, stop_sigma_px
        )
        if len(pix_idx):
            sp_pix.append(pix_idx)
            sp_stop.append(np.full(len(pix_idx), v_idx, dtype=np.int64))
            sp_w.append(w)
    stop_pixel_idx = np.concatenate(sp_pix) if sp_pix else np.array([], dtype=np.int64)
    stop_stop_idx = np.concatenate(sp_stop) if sp_stop else np.array([], dtype=np.int64)
    stop_weight = np.concatenate(sp_w) if sp_w else np.array([], dtype=np.float32)
    ep_pix, ep_edge, ep_w = ([], [], [])
    edge_paths_pixel = {}
    edge_paths_proj = {}
    if candidate_edge_mode == "road_segments":
        for e_idx in range(n_edges):
            geom = edge_geom_arr[e_idx] if e_idx < len(edge_geom_arr) else None
            pix_idx, w = _geometry_splat_pixels(geom, bounds, H, W, edge_sigma_px)
            if len(pix_idx):
                ep_pix.append(pix_idx)
                ep_edge.append(np.full(len(pix_idx), e_idx, dtype=np.int64))
                ep_w.append(w)
    elif routing_mode == "road_shortest_path" and road_graph is not None:
        ctx = f" ({log_context})" if log_context else ""
        node_source = road_nodes_lookup_proj
        if node_source is None:
            node_source = road_nodes_proj
        road_xy_lookup = {}
        if node_source is not None and (not node_source.empty):
            id_col = "id" if "id" in node_source.columns else None
            if id_col is not None:
                for _, nrow in node_source[["id", "geometry"]].iterrows():
                    road_xy_lookup[nrow["id"]] = (
                        float(nrow.geometry.x),
                        float(nrow.geometry.y),
                    )
            else:
                for nid, nrow in node_source[["geometry"]].iterrows():
                    road_xy_lookup[nid] = (
                        float(nrow.geometry.x),
                        float(nrow.geometry.y),
                    )
        routed_count = 0
        no_path_count = 0
        missing_node_count = 0
        short_path_count = 0
        for e_idx in range(n_edges):
            u_idx = int(edge_u_arr[e_idx])
            v_idx = int(edge_v_arr[e_idx])
            road_u_id = stop_node_ids[u_idx]
            road_v_id = stop_node_ids[v_idx]
            try:
                if road_graph.has_node(road_u_id) and road_graph.has_node(road_v_id):
                    path = nx.shortest_path(
                        road_graph, road_u_id, road_v_id, weight="length"
                    )
                    path_coords_proj = []
                    for nid in path:
                        xy = road_xy_lookup.get(nid)
                        if xy is not None:
                            path_coords_proj.append(xy)
                    if len(path_coords_proj) >= 2:
                        path_coords_pixel = []
                        for x, y in path_coords_proj:
                            col, row = _xy_to_pixel(
                                np.array([x]), np.array([y]), bounds, H, W
                            )
                            path_coords_pixel.append((float(col[0]), float(row[0])))
                        edge_paths_pixel[e_idx] = path_coords_pixel
                        edge_paths_proj[e_idx] = np.asarray(
                            path_coords_proj, dtype=np.float32
                        )
                        routed_count += 1
                    else:
                        short_path_count += 1
                else:
                    missing_node_count += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                no_path_count += 1
        if routed_count != n_edges:
            raise RuntimeError(
                f"Road-shortest-path routing incomplete{ctx}: routed={routed_count}/{n_edges}, no_path={no_path_count}, missing_node={missing_node_count}, short_path={short_path_count}"
            )
        for e_idx in range(n_edges):
            u_idx = int(edge_u_arr[e_idx])
            v_idx = int(edge_v_arr[e_idx])
            if e_idx in edge_paths_pixel:
                path_coords = edge_paths_pixel[e_idx]
                pix_idx, w = _multiline_splat_pixels(path_coords, H, W, edge_sigma_px)
            else:
                raise RuntimeError(
                    f"Missing routed path for candidate edge {e_idx} in strict road_shortest_path mode."
                )
            if len(pix_idx):
                ep_pix.append(pix_idx)
                ep_edge.append(np.full(len(pix_idx), e_idx, dtype=np.int64))
                ep_w.append(w)
    else:
        for e_idx in range(n_edges):
            u_idx = int(edge_u_arr[e_idx])
            v_idx = int(edge_v_arr[e_idx])
            col0, row0 = (float(stop_col_f[u_idx]), float(stop_row_f[u_idx]))
            col1, row1 = (float(stop_col_f[v_idx]), float(stop_row_f[v_idx]))
            pix_idx, w = _line_splat_pixels(col0, row0, col1, row1, H, W, edge_sigma_px)
            if len(pix_idx):
                ep_pix.append(pix_idx)
                ep_edge.append(np.full(len(pix_idx), e_idx, dtype=np.int64))
                ep_w.append(w)
    edge_pixel_idx = np.concatenate(ep_pix) if ep_pix else np.array([], dtype=np.int64)
    edge_edge_idx = np.concatenate(ep_edge) if ep_edge else np.array([], dtype=np.int64)
    edge_weight = np.concatenate(ep_w) if ep_w else np.array([], dtype=np.float32)
    edge_path_xy = []
    for e_idx in range(n_edges):
        u_idx = int(edge_u_arr[e_idx])
        v_idx = int(edge_v_arr[e_idx])
        coords = np.zeros((0, 2), dtype=np.float32)
        if candidate_edge_mode == "road_segments":
            geom = edge_geom_arr[e_idx] if e_idx < len(edge_geom_arr) else None
            coords = _geometry_to_xy_path(geom)
        elif routing_mode == "road_shortest_path":
            coords = edge_paths_proj.get(e_idx, np.zeros((0, 2), dtype=np.float32))
        if coords.shape[0] < 2:
            coords = stop_xy[[u_idx, v_idx]].astype(np.float32)
        edge_path_xy.append(coords)
    return TileGraphSpec(
        stop_xy=stop_xy,
        stop_node_ids=stop_node_ids,
        edge_u=edge_u_arr,
        edge_v=edge_v_arr,
        edge_length=edge_length_arr,
        edge_path_xy=tuple(edge_path_xy),
        stop_pixel_idx=stop_pixel_idx,
        stop_stop_idx=stop_stop_idx,
        stop_weight=stop_weight,
        edge_pixel_idx=edge_pixel_idx,
        edge_edge_idx=edge_edge_idx,
        edge_weight=edge_weight,
        H=H,
        W=W,
        bounds=bounds,
        crs=crs,
        candidate_edge_mode=candidate_edge_mode,
        routing_mode=routing_mode,
    )


def gtfs_to_activations(
    spec: TileGraphSpec,
    gtfs_nodes_proj: Optional[gpd.GeoDataFrame],
    gtfs_edges_proj: Optional[gpd.GeoDataFrame],
    road_graph: Optional[nx.MultiDiGraph] = None,
    *,
    candidate_edge_mode: str = "road_segments",
    snap_radius_m: float = 300.0,
    routing_direction: str = "undirected",
) -> Tuple[np.ndarray, np.ndarray]:
    """Map GTFS stops and edges to candidate-space activations (w_real, z_real)."""
    n_stops = spec.num_stops()
    n_edges = spec.num_edges()
    routing_direction = _normalize_bus_path_routing_direction(routing_direction)
    z_real = np.zeros(n_stops, dtype=np.float32)
    w_real = np.zeros(n_edges, dtype=np.float32)
    if n_stops == 0:
        return (w_real, z_real)
    z_real, gtfs_id_to_cand, pair_counts, snap_stats = _map_gtfs_to_candidate_pairs(
        spec, gtfs_nodes_proj, gtfs_edges_proj, snap_radius_m=snap_radius_m
    )
    if n_edges > 0 and pair_counts:
        if candidate_edge_mode == "road_segments":
            if road_graph is None:
                raise ValueError(
                    "candidate_edge_mode='road_segments' requires road_graph for GTFS activation mapping"
                )
            routing_graph = _resolve_bus_path_routing_graph(
                road_graph, routing_direction
            )
            road_pair_to_edge: dict = {}
            for e_idx, (eu, ev) in enumerate(zip(spec.edge_u, spec.edge_v)):
                ru = int(spec.stop_node_ids[int(eu)])
                rv = int(spec.stop_node_ids[int(ev)])
                key = (min(ru, rv), max(ru, rv))
                if key not in road_pair_to_edge:
                    road_pair_to_edge[key] = e_idx
            routed_pairs = 0
            no_path_pairs = 0
            failure_examples = []
            component_lookup = None
            for (u_cand, v_cand), cnt in pair_counts.items():
                ru = int(spec.stop_node_ids[int(u_cand)])
                rv = int(spec.stop_node_ids[int(v_cand)])
                try:
                    path = nx.shortest_path(routing_graph, ru, rv, weight="length")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    no_path_pairs += 1
                    if component_lookup is None:
                        component_lookup = _road_component_lookup(routing_graph)
                    if len(failure_examples) < 5:
                        failure_examples.append(
                            {
                                "reason": _failure_reason(
                                    routing_graph, component_lookup, ru, rv
                                ),
                                "count": float(cnt),
                                "candidate_u": int(u_cand),
                                "candidate_v": int(v_cand),
                                "road_u_id": ru,
                                "road_v_id": rv,
                                "component_u": component_lookup.get(ru),
                                "component_v": component_lookup.get(rv),
                                "straight_line_m": round(
                                    float(
                                        np.linalg.norm(
                                            spec.stop_xy[int(u_cand)]
                                            - spec.stop_xy[int(v_cand)]
                                        )
                                    ),
                                    1,
                                ),
                            }
                        )
                    continue
                routed_pairs += 1
                for a, b in zip(path[:-1], path[1:]):
                    key = (min(int(a), int(b)), max(int(a), int(b)))
                    e_idx = road_pair_to_edge.get(key)
                    if e_idx is not None:
                        w_real[e_idx] += float(cnt)
            if no_path_pairs:
                warnings.warn(
                    "GTFS->road edge activation diagnostics: snap=%s examples=%s"
                    % (snap_stats, failure_examples),
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            edge_lookup: dict = {}
            for e_idx, (eu, ev) in enumerate(zip(spec.edge_u, spec.edge_v)):
                key = (int(min(eu, ev)), int(max(eu, ev)))
                edge_lookup[key] = e_idx
            for u_cand, v_cand in pair_counts:
                key = (int(min(u_cand, v_cand)), int(max(u_cand, v_cand)))
                cnt = pair_counts[u_cand, v_cand]
                e_idx = edge_lookup.get(key)
                if e_idx is not None:
                    w_real[e_idx] += float(cnt)
    return (w_real, z_real)


def _empty_spec(H: int, W: int, bounds: tuple, crs: str) -> TileGraphSpec:
    """Empty TileGraphSpec for tiles with no road data."""
    return TileGraphSpec(
        stop_xy=np.zeros((0, 2), dtype=np.float32),
        stop_node_ids=np.array([], dtype=np.int64),
        edge_u=np.array([], dtype=np.int64),
        edge_v=np.array([], dtype=np.int64),
        edge_length=np.array([], dtype=np.float32),
        edge_path_xy=tuple(),
        stop_pixel_idx=np.array([], dtype=np.int64),
        stop_stop_idx=np.array([], dtype=np.int64),
        stop_weight=np.array([], dtype=np.float32),
        edge_pixel_idx=np.array([], dtype=np.int64),
        edge_edge_idx=np.array([], dtype=np.int64),
        edge_weight=np.array([], dtype=np.float32),
        H=H,
        W=W,
        bounds=bounds,
        crs=crs,
        candidate_edge_mode="road_segments",
        routing_mode="airline",
    )


__all__ = [
    "TileGraphSpec",
    "build_tile_graph_spec",
    "gtfs_to_activations",
    "render_gtfs_paths_to_grid",
]
