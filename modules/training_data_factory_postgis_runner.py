"""Config-driven PostGIS-backed training data factory with resume support."""

from __future__ import annotations
import warnings
import argparse
import os
import pickle
import signal
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import geopandas as gpd
import pandas as pd
from shapely import wkb as shapely_wkb
from shapely.geometry import box
from modules.postgis_tile_source import SpatialTileSource
from modules.training_data import sample_tiles_nonoverlap
from modules.training_data_factory import (
    build_feature_groups,
    build_tensor_schema,
    filter_feature_groups_for_aux_flags,
    load_config,
    process_tile,
)
from modules.training_runs import save_manifest, save_tile_npz


def _cfg_payload(cfg) -> dict:
    return {
        "GRID_CRS": cfg.GRID_CRS,
        "PIXEL_SIZE_M": cfg.PIXEL_SIZE_M,
        "GRID_WIDTH": cfg.GRID_WIDTH,
        "GRID_HEIGHT": cfg.GRID_HEIGHT,
        "RASTER_METHOD": cfg.RASTER_METHOD,
        "H3_RES": cfg.H3_RES,
        "ACCESS_MIN_VALUE": getattr(cfg, "ACCESS_MIN_VALUE", 0.0),
        "ACCESS_MAX_VALUE": getattr(cfg, "ACCESS_MAX_VALUE", 30.0),
        "BUILD_POI_TENSORS": cfg.BUILD_POI_TENSORS,
        "POI_SMOOTH": cfg.POI_SMOOTH,
        "POI_SMOOTH_SIGMA_PX": cfg.POI_SMOOTH_SIGMA_PX,
        "POI_BOOST": cfg.POI_BOOST,
        "POI_MIN_VALUE": cfg.POI_MIN_VALUE,
        "POI_CLIP_MIN_VALUE": getattr(cfg, "POI_CLIP_MIN_VALUE", 0.0),
        "POI_CLIP_MAX_VALUE": getattr(cfg, "POI_CLIP_MAX_VALUE", 1.0),
        "POI_KERNEL_PEAK_VALUE": getattr(cfg, "POI_KERNEL_PEAK_VALUE", 1.0),
        "BUS_TENSOR_ENABLED": cfg.BUS_TENSOR_ENABLED,
        "BUS_TILE_BUFFER_M": cfg.BUS_TILE_BUFFER_M,
        "TILE_GRAPH_SNAP_RADIUS_M": getattr(cfg, "TILE_GRAPH_SNAP_RADIUS_M", 200.0),
        "TILE_GRAPH_NEIGHBOR_K": getattr(cfg, "TILE_GRAPH_NEIGHBOR_K", 6),
        "TILE_GRAPH_MAX_STOPS": getattr(cfg, "TILE_GRAPH_MAX_STOPS", 1500),
        "TILE_GRAPH_STOP_SIGMA_PX": getattr(cfg, "TILE_GRAPH_STOP_SIGMA_PX", 2.0),
        "TILE_GRAPH_EDGE_SIGMA_PX": getattr(cfg, "TILE_GRAPH_EDGE_SIGMA_PX", 1.5),
        "TILE_GRAPH_ROUTING_MODE": getattr(
            cfg, "TILE_GRAPH_ROUTING_MODE", "road_shortest_path"
        ),
        "TILE_GRAPH_CANDIDATE_EDGE_MODE": getattr(
            cfg, "TILE_GRAPH_CANDIDATE_EDGE_MODE", "road_segments"
        ),
        "BUS_W_REAL_MODE": getattr(cfg, "BUS_W_REAL_MODE", "binary"),
        "BUS_EDGE_RENDER_MODE": getattr(cfg, "BUS_EDGE_RENDER_MODE", "routed_paths"),
        "BUS_PATH_RENDER_WEIGHT_MODE": getattr(
            cfg, "BUS_PATH_RENDER_WEIGHT_MODE", "count"
        ),
        "BUS_PATH_ROUTING_DIRECTION": getattr(
            cfg, "BUS_PATH_ROUTING_DIRECTION", "undirected"
        ),
        "RENDERER_EDGE_GATE_MODE": getattr(cfg, "RENDERER_EDGE_GATE_MODE", "auto"),
        "RENDERER_SOFT_CAP_MAX": getattr(cfg, "RENDERER_SOFT_CAP_MAX", 1.0),
        "RENDERER_SOFT_CAP_BETA": getattr(cfg, "RENDERER_SOFT_CAP_BETA", 10.0),
    }


def _derive_bus_delta_and_normalize_modes(cells, cfg_modes=None):
    """Derive bus_delta = walk_transit - walk_transit_rail; drop raw walk if not in MODES."""
    import numpy as np

    if len(cells.columns) == 0:
        return cells
    cols = list(cells.columns)
    want_bus_delta = cfg_modes is None or "bus_delta" in (cfg_modes or [])
    want_walk = cfg_modes is not None and "walk" in cfg_modes
    if want_bus_delta:
        wt_cols = {
            c[: -len("_walk_transit")]: c
            for c in cols
            if c.endswith("_walk_transit")
            and (not c.endswith("_walk_transit_bus"))
            and (not c.endswith("_walk_transit_rail"))
        }
        wtr_cols = {
            c[: -len("_walk_transit_rail")]: c
            for c in cols
            if c.endswith("_walk_transit_rail")
        }
        derived = {}
        for base, wt_col in wt_cols.items():
            wtr_col = wtr_cols.get(base)
            if wtr_col:
                derived[f"{base}_bus_delta"] = (
                    cells[wt_col].to_numpy() - cells[wtr_col].to_numpy()
                )
        if derived:
            geom_col = cells.geometry.name if hasattr(cells, "geometry") else None
            crs = cells.crs if hasattr(cells, "crs") else None
            new_cols = pd.DataFrame(
                {
                    col: np.maximum(vals, 0.0).astype(np.float32)
                    for col, vals in derived.items()
                },
                index=cells.index,
            )
            cells = pd.concat([cells, new_cols], axis=1)
            if (
                geom_col is not None
                and crs is not None
                and (not isinstance(cells, gpd.GeoDataFrame))
            ):
                cells = gpd.GeoDataFrame(cells, geometry=geom_col, crs=crs)
    if not want_walk:
        walk_only = [
            c
            for c in cells.columns
            if c.endswith("_walk")
            and (not c.endswith("_walk_transit"))
            and (not c.endswith("_walk_transit_bus"))
            and (not c.endswith("_walk_transit_rail"))
        ]
        if walk_only:
            cells = cells.drop(columns=walk_only)
    return cells


@dataclass
class TilePlan:
    tiles_wkb: list[bytes]
    grid_crs: str
    grid_width: int
    grid_height: int
    pixel_size_m: float
    run_id: str
    poi_dataset_key: str
    road_graph_key: str
    bus_graph_key: str
    cfg_payload: dict
    feature_groups: dict
    poi_order: list[str]
    meta: dict
    cfg_modes: list[str]
    tensor_schema: dict
    created_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    group: str = ""
    comment: str = ""


def _plan_path(out_dir: Path) -> Path:
    return out_dir / "tile_plan.pkl"


def _boundary_path(out_dir: Path) -> Path:
    return out_dir / "sampling_boundary.pkl"


def _load_sampling_boundary(out_dir: Path):
    p = _boundary_path(out_dir)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _save_sampling_boundary(out_dir: Path, boundary) -> None:
    with open(_boundary_path(out_dir), "wb") as f:
        pickle.dump(boundary, f, protocol=5)


def _build_sampling_boundary(
    run_bounds_wgs84, grid_crs: str, *, require_bundeslaender: bool = False
) -> "Any":
    """Build the sampling boundary polygon for tile placement."""
    from shapely.ops import unary_union

    germany_path = (
        Path(__file__).resolve().parent
        / "data"
        / "Deutschland_Bundesländergrenzen.geojson"
    )
    if germany_path.exists():
        bundeslaender = gpd.read_file(germany_path)
        if str(bundeslaender.crs) != grid_crs:
            bundeslaender = bundeslaender.to_crs(grid_crs)
        germany_union = unary_union(bundeslaender.geometry).buffer(5000).buffer(-2000)
        if germany_union.is_empty:
            raise RuntimeError(
                "Bundesländer GeoJSON loaded but produced an empty geometry after buffering."
            )
        return germany_union
    if require_bundeslaender:
        raise RuntimeError(
            f"Bundesländer GeoJSON not found at {germany_path}. Required for this run — copy the file or set require_bundeslaender=False to use run bounds."
        )
    bounds_proj = (
        gpd.GeoSeries([run_bounds_wgs84], crs="EPSG:4326").to_crs(grid_crs).iloc[0]
    )
    return bounds_proj


def save_tile_plan(out_dir: Path, plan: TilePlan) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _plan_path(out_dir)
    with open(p, "wb") as f:
        pickle.dump(plan, f, protocol=5)
    return p


def load_tile_plan(out_dir: Path) -> TilePlan | None:
    p = _plan_path(out_dir)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        plan = pickle.load(f)
    return plan


def _completed_indices(out_dir: Path) -> set[int]:
    tiles_dir = out_dir / "tiles"
    if not tiles_dir.exists():
        return set()
    done = set()
    for p in tiles_dir.glob("tile_*.npz"):
        try:
            idx = int(p.stem.split("_")[1])
            done.add(idx)
        except (ValueError, IndexError):
            pass
    return done


def _tile_worker(packed: dict) -> dict:
    """Process one tile in a subprocess. Creates its own DB connection."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    mem_limit_gb = packed.get("mem_limit_gb")
    if mem_limit_gb:
        import resource

        limit = int(mem_limit_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    from modules.postgis_bridge import reload_engine

    reload_engine()
    tile_idx: int = packed["tile_idx"]
    tile_proj = shapely_wkb.loads(packed["tile_wkb"])
    plan: dict = packed["plan"]
    tile_start = time.perf_counter()
    tile_timings: dict[str, float] = {}
    src = SpatialTileSource()
    cfg_payload = plan["cfg_payload"]
    grid_crs = plan["grid_crs"]
    run_id = plan["run_id"]
    poi_dataset_key = plan["poi_dataset_key"]
    road_graph_key = plan["road_graph_key"]
    bus_graph_key = plan["bus_graph_key"]
    feature_groups = plan["feature_groups"]
    poi_order = plan["poi_order"]
    meta = plan["meta"]
    cfg_modes = plan["cfg_modes"]
    graph_buffer_m = plan["graph_buffer_m"]
    poi_buffer_m = plan["poi_buffer_m"]
    raster_workers = plan["raster_workers"]
    chunk_rows = plan["chunk_rows"]
    tensor_schema = plan["tensor_schema"]
    out_dir = Path(plan["out_dir"])
    tile_wgs = gpd.GeoSeries([tile_proj], crs=grid_crs).to_crs("EPSG:4326").iloc[0]
    t0 = time.perf_counter()
    cells = src.get_access_cells(run_id, tile_wgs)
    tile_timings["db.access_cells"] = time.perf_counter() - t0
    if cells.empty:
        raise RuntimeError(f"tile={tile_idx}: PostGIS access query returned no cells")
    cells = _derive_bus_delta_and_normalize_modes(cells, cfg_modes)
    t0 = time.perf_counter()
    poi_features = src.get_poi_features(
        poi_dataset_key,
        tile_wgs,
        layer_names=poi_order,
        output_crs=grid_crs,
        buffer_m=poi_buffer_m,
    )
    tile_timings["db.poi_features"] = time.perf_counter() - t0
    poi_layers = {
        name: group.drop(columns=["dataset_key", "layer_name"], errors="ignore").copy()
        for name, group in poi_features.groupby("layer_name", dropna=True)
    }
    t0 = time.perf_counter()
    road_nodes_proj, road_edges_proj = src.get_graph_slice(
        road_graph_key,
        tile_proj,
        tile_crs=grid_crs,
        buffer_m=graph_buffer_m,
        output_crs=grid_crs,
    )
    tile_timings["db.road_graph"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bus_nodes_proj, bus_edges_proj = src.get_graph_slice(
        bus_graph_key,
        tile_proj,
        tile_crs=grid_crs,
        buffer_m=graph_buffer_m,
        output_crs=grid_crs,
    )
    tile_timings["db.bus_graph"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    road_graph = src.build_networkx_subgraph(
        road_graph_key, tile_proj, tile_crs=grid_crs, buffer_m=graph_buffer_m
    )
    tile_timings["db.road_networkx"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    arrays, rec = process_tile(
        tile_idx,
        tile_proj,
        cfg_payload,
        cells,
        meta,
        feature_groups,
        poi_order,
        poi_layers,
        bus_nodes_proj,
        bus_edges_proj,
        ["stops", "edges"],
        road_nodes_proj,
        road_edges_proj,
        road_graph,
        raster_workers=raster_workers,
        chunk_rows=chunk_rows,
    )
    tile_timings["rasterize"] = time.perf_counter() - t0
    rec["tensor_schema"] = tensor_schema
    t0 = time.perf_counter()
    paths = save_tile_npz(
        out_dir / "tiles",
        tile_idx,
        arrays,
        {k: v for k, v in rec.items() if k != "spec"},
    )
    spec = rec.get("spec")
    spec_path = None
    if spec is not None:
        spec_dir = out_dir / "tile_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"tile_{tile_idx:04d}.spec.pkl"
        spec.save(spec_path)
    tile_timings["io.save"] = time.perf_counter() - t0
    tile_total = time.perf_counter() - tile_start
    tile_timings["total"] = tile_total
    db_total = sum((v for k, v in tile_timings.items() if k.startswith("db.")))
    tile_timings["db.total"] = db_total
    return {
        "tile_idx": tile_idx,
        "npz_path": str(paths.npz_path),
        "meta_path": str(paths.meta_path),
        "spec_path": str(spec_path) if spec_path else None,
        "arrays": {k: list(v.shape) for k, v in arrays.items()},
        "candidate_stops": int(spec.num_stops()) if spec else 0,
        "candidate_edges": int(spec.num_edges()) if spec else 0,
        "access_cells": int(len(cells)),
        "poi_features": int(len(poi_features)),
        "road_nodes": int(len(road_nodes_proj)),
        "road_edges": int(len(road_edges_proj)),
        "bus_nodes": int(len(bus_nodes_proj)),
        "bus_edges": int(len(bus_edges_proj)),
        "timings_s": {k: round(v, 4) for k, v in tile_timings.items()},
        "worker_pid": os.getpid(),
    }


_MIN_POI_FEATURES_WARN = 100000
_MIN_POI_COVERAGE_FRACTION = 0.8


def _preflight_check(
    src: SpatialTileSource,
    run_id: str,
    poi_dataset_key: str,
    road_graph_key: str,
    bus_graph_key: str,
) -> None:
    """Fail fast if PostGIS is missing required data for a tile-factory run."""
    from sqlalchemy import text as _text

    errors: list[str] = []
    warnings: list[str] = []
    with src.engine.connect() as conn:
        row = (
            conn.execute(
                _text(
                    "SELECT run_id, ST_AsBinary(bounds) AS b FROM access.runs WHERE run_id = :r"
                ),
                {"r": run_id},
            )
            .mappings()
            .first()
        )
        if row is None:
            errors.append(
                f"Access run not ingested in PostGIS: {run_id!r}  → run scripts/postgis/ingest_access_run.py first"
            )
            run_bounds = None
        else:
            from shapely import wkb as _wkb

            run_bounds = _wkb.loads(bytes(row["b"])) if row["b"] else None
        n_cells = conn.execute(
            _text("SELECT COUNT(*) FROM access.cells WHERE run_id = :r"), {"r": run_id}
        ).scalar()
        if not n_cells:
            errors.append(
                f"Access run {run_id!r} has no cells in PostGIS — ingest may have failed or is still running"
            )
        poi_row = conn.execute(
            _text("SELECT dataset_key FROM poi.datasets WHERE dataset_key = :k"),
            {"k": poi_dataset_key},
        ).first()
        if poi_row is None:
            errors.append(
                f"POI bundle not ingested: {poi_dataset_key!r}  → run scripts/postgis/ingest_poi_bundles.py first"
            )
            n_poi = 0
            poi_bounds = None
        else:
            n_poi = conn.execute(
                _text("SELECT COUNT(*) FROM poi.features WHERE dataset_key = :k"),
                {"k": poi_dataset_key},
            ).scalar()
            if n_poi < _MIN_POI_FEATURES_WARN:
                warnings.append(
                    f"POI bundle {poi_dataset_key!r} has only {n_poi:,} features — expected ≥{_MIN_POI_FEATURES_WARN:,}. The MAPPER run may lack POI classes. Re-run ingest_poi_bundles.py on the complete run directory."
                )
            if run_bounds is not None and n_poi > 0:
                poi_ext = conn.execute(
                    _text(
                        "SELECT ST_Extent(geom) FROM poi.features WHERE dataset_key = :k"
                    ),
                    {"k": poi_dataset_key},
                ).scalar()
                if poi_ext:
                    from shapely import from_wkt as _from_wkt

                    try:
                        poi_bounds = _from_wkt(str(poi_ext))
                        intersection = run_bounds.intersection(poi_bounds)
                        coverage = (
                            intersection.area / run_bounds.area
                            if run_bounds.area > 0
                            else 0
                        )
                        if coverage < _MIN_POI_COVERAGE_FRACTION:
                            warnings.append(
                                f"POI bundle covers only {coverage:.0%} of the access run's bounding box (need ≥{_MIN_POI_COVERAGE_FRACTION:.0%}). POI data may be from a different region or partial run."
                            )
                    except Exception:
                        pass
        for gkey, gname in [(road_graph_key, "road"), (bus_graph_key, "bus")]:
            if gkey:
                g_row = conn.execute(
                    _text("SELECT graph_key FROM graph.datasets WHERE graph_key = :k"),
                    {"k": gkey},
                ).first()
                if g_row is None:
                    errors.append(
                        f"{gname} graph not ingested: {gkey!r}  → run scripts/postgis/ingest_graph.py first"
                    )
    for w in warnings:
        warnings.warn("PREFLIGHT WARNING: %s" % (w,), RuntimeWarning, stacklevel=2)
    if errors:
        msg = "\n".join((f"  • {e}" for e in errors))
        raise RuntimeError(
            f"Tile-factory preflight check failed — fix these issues before running:\n{msg}"
        )
    if not warnings:
        pass


def _build_tile_plan(cfg, out_dir: Path, tile_plan_path: str | None = None) -> TilePlan:
    run_id = cfg.RUN_ID
    poi_dataset_key = cfg.POI_DATASET_KEY
    road_graph_key = cfg.ROAD_GRAPH_KEY
    bus_graph_key = cfg.BUS_GRAPH_KEY
    src = SpatialTileSource()
    _preflight_check(src, run_id, poi_dataset_key, road_graph_key, bus_graph_key)
    run_meta = src.get_access_run_meta(run_id)
    meta = dict(run_meta.get("config_json") or {})
    meta["run_id"] = run_id
    meta["region"] = run_meta.get("region")
    meta["h3_res"] = run_meta.get("h3_res") or getattr(cfg, "H3_RES", None)
    grid_width = int(cfg.GRID_WIDTH)
    grid_height = int(cfg.GRID_HEIGHT)
    pixel_size_m = float(cfg.PIXEL_SIZE_M)
    tile_scale = float(getattr(cfg, "TILE_SCALE", 1.0))
    grid_width = max(1, int(round(grid_width * tile_scale)))
    grid_height = max(1, int(round(grid_height * tile_scale)))
    schema_cols = list((run_meta.get("tensor_schema_json") or {}).get("columns") or [])
    cfg_modes = list(getattr(cfg, "MODES", None) or [])
    dummy = pd.DataFrame(columns=schema_cols)
    dummy = _derive_bus_delta_and_normalize_modes(dummy, cfg_modes)
    feature_groups = filter_feature_groups_for_aux_flags(
        build_feature_groups(dummy, getattr(cfg, "ID_COLS")), cfg
    )
    allow_fallback = bool(getattr(cfg, "ALLOW_POI_LAYER_ORDER_FALLBACK", False))
    poi_order = _resolve_poi_order(src, poi_dataset_key, allow_fallback=allow_fallback)
    if tile_plan_path:
        with open(tile_plan_path, "rb") as f:
            ext_plan = pickle.load(f)
        if hasattr(ext_plan, "tiles_wkb"):
            tiles_proj = [shapely_wkb.loads(b) for b in ext_plan.tiles_wkb]
            plan_crs = ext_plan.grid_crs
        else:
            tiles_proj = [shapely_wkb.loads(b) for b in ext_plan["tiles_wkb"]]
            plan_crs = ext_plan.get("grid_crs", cfg.GRID_CRS)
        if plan_crs != cfg.GRID_CRS:
            tiles_proj = (
                gpd.GeoSeries(tiles_proj, crs=plan_crs).to_crs(cfg.GRID_CRS).tolist()
            )
    elif int(cfg.MAX_SAMPLES) == 1 and (not bool(getattr(cfg, "USE_SAMPLER", False))):
        cx = run_meta["bounds"].centroid
        center = gpd.GeoSeries([cx], crs="EPSG:4326").to_crs(cfg.GRID_CRS).iloc[0]
        half_w = grid_width * pixel_size_m / 2.0
        half_h = grid_height * pixel_size_m / 2.0
        tiles_proj = [
            box(
                center.x - half_w,
                center.y - half_h,
                center.x + half_w,
                center.y + half_h,
            )
        ]
    else:
        sampling_boundary = _load_sampling_boundary(out_dir)
        if sampling_boundary is not None:
            pass
        else:
            sampling_boundary = _build_sampling_boundary(
                run_meta["bounds"],
                cfg.GRID_CRS,
                require_bundeslaender=bool(
                    getattr(cfg, "REQUIRE_BUNDESLAENDER_BOUNDARY", False)
                ),
            )
            _save_sampling_boundary(out_dir, sampling_boundary)
        empty_cells = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=cfg.GRID_CRS))
        tiles_proj, region_proj = sample_tiles_nonoverlap(
            empty_cells,
            width=grid_width,
            height=grid_height,
            pixel_size=pixel_size_m,
            crs_out=cfg.GRID_CRS,
            max_samples=int(cfg.MAX_SAMPLES),
            overlap_tol=float(cfg.OVERLAP_TOL),
            max_overlap_stop=float(cfg.MAX_OVERLAP_STOP),
            max_tries_per_tile=int(cfg.MAX_TRIES_PER_TILE),
            seed=cfg.TILE_SEED,
            hull_buffer_m=float(cfg.HULL_BUFFER_M),
            sampling_boundary=sampling_boundary,
        )
    if not tiles_proj:
        raise RuntimeError("No tiles planned")
    min_coverage = float(getattr(cfg, "MIN_COVERAGE_FRACTION", 0.7))
    if not bool(getattr(cfg, "SKIP_COVERAGE_CHECK", False)):
        coverage_workers = int(getattr(cfg, "COVERAGE_WORKERS", 32))

        def _check_coverage(tile):
            tile_wgs = (
                gpd.GeoSeries([tile], crs=cfg.GRID_CRS).to_crs("EPSG:4326").iloc[0]
            )
            return (tile, src.coverage_fraction(run_id, tile_wgs))

        valid_tiles = []
        skipped = 0
        with ThreadPoolExecutor(max_workers=coverage_workers) as pool:
            futures = {pool.submit(_check_coverage, t): t for t in tiles_proj}
            for fut in as_completed(futures):
                tile, frac = fut.result()
                if frac >= min_coverage:
                    valid_tiles.append(tile)
                else:
                    skipped += 1
        if skipped:
            warnings.warn(
                "Dropped %d/%d tiles below %.0f%% coverage threshold"
                % (skipped, len(tiles_proj), min_coverage * 100),
                RuntimeWarning,
                stacklevel=2,
            )
        tiles_proj = valid_tiles
        if not tiles_proj:
            raise RuntimeError(
                f"No tiles meet the {min_coverage:.0%} coverage threshold — check RUN_ID and tile bounds, or lower MIN_COVERAGE_FRACTION in the config"
            )
    cfg_payload_full = {
        **_cfg_payload(cfg),
        "GRID_WIDTH": grid_width,
        "GRID_HEIGHT": grid_height,
        "PIXEL_SIZE_M": pixel_size_m,
        "BUS_TILE_BUFFER_M": float(cfg.BUS_TILE_BUFFER_M),
    }
    tensor_schema = build_tensor_schema(feature_groups, poi_order, ["stops", "edges"])
    return TilePlan(
        tiles_wkb=[t.wkb for t in tiles_proj],
        grid_crs=cfg.GRID_CRS,
        grid_width=grid_width,
        grid_height=grid_height,
        pixel_size_m=pixel_size_m,
        run_id=run_id,
        poi_dataset_key=poi_dataset_key,
        road_graph_key=road_graph_key,
        bus_graph_key=bus_graph_key,
        cfg_payload=cfg_payload_full,
        feature_groups={k: list(v) for k, v in feature_groups.items()},
        poi_order=list(poi_order),
        meta=meta,
        cfg_modes=cfg_modes,
        tensor_schema=tensor_schema,
        group=getattr(cfg, "RUN_GROUP", ""),
        comment=getattr(cfg, "RUN_COMMENT", ""),
    )


def _resolve_poi_order(
    src: SpatialTileSource, dataset_key: str, *, allow_fallback: bool = False
) -> list[str]:
    try:
        meta = src.get_poi_dataset_metadata(dataset_key)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Cannot resolve POI layer order for {dataset_key!r}: {exc}"
        ) from exc
    rows = meta.get("alignment_rows") or []
    order = [r.get("resolved_layer_name") for r in rows if r.get("resolved_layer_name")]
    if order:
        return order
    if not allow_fallback:
        raise RuntimeError(
            f"POI dataset {dataset_key!r} has no alignment_rows — cannot verify tensor channel order. Pass --allow-poi-layer-order-fallback only for debugging."
        )
    warnings.warn(
        "POI dataset %s has no alignment_rows; using alphabetical fallback."
        % (dataset_key,),
        RuntimeWarning,
        stacklevel=2,
    )
    return src.get_poi_layers(dataset_key)["layer_name"].tolist()


def _apply_cfg_overrides(cfg, overrides: dict[str, Any] | None = None):
    for attr, value in (overrides or {}).items():
        if value is not None:
            setattr(cfg, attr, value)
    return cfg


def run_parallel_cfg(cfg, tile_plan_path: str | None = None) -> dict:
    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = load_tile_plan(out_dir)
    if plan is None:
        plan = _build_tile_plan(cfg, out_dir, tile_plan_path=tile_plan_path)
        save_tile_plan(out_dir, plan)
    total_tiles = len(plan.tiles_wkb)
    done = _completed_indices(out_dir)
    pending = [i for i in range(total_tiles) if i not in done]
    if done:
        pass
    if not pending:
        return _assemble_manifest(out_dir, plan, total_tiles)
    plan_dict = {
        "grid_crs": plan.grid_crs,
        "run_id": plan.run_id,
        "poi_dataset_key": plan.poi_dataset_key,
        "road_graph_key": plan.road_graph_key,
        "bus_graph_key": plan.bus_graph_key,
        "cfg_payload": plan.cfg_payload,
        "feature_groups": plan.feature_groups,
        "poi_order": plan.poi_order,
        "meta": plan.meta,
        "cfg_modes": plan.cfg_modes,
        "tensor_schema": plan.tensor_schema,
        "graph_buffer_m": float(getattr(cfg, "GRAPH_BUFFER_M", 5000.0)),
        "poi_buffer_m": float(getattr(cfg, "POI_BUFFER_M", 500.0)),
        "raster_workers": int(getattr(cfg, "RASTER_WORKERS", 1)),
        "chunk_rows": int(getattr(cfg, "RASTER_CHUNK_ROWS", 32)),
        "out_dir": str(out_dir),
    }
    workers = int(cfg.TILE_WORKERS)
    mem_limit_gb = float(getattr(cfg, "WORKER_MEM_LIMIT_GB", None) or 0)
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / 1024**3
        est_per_worker = mem_limit_gb if mem_limit_gb else 6.0
        estimated_total = workers * est_per_worker
        if estimated_total > avail_gb:
            warnings.warn(
                "RAM check: %d workers × ~%.0f GB = ~%.0f GB estimated, only %.1f GB available — consider fewer workers or --worker-mem-limit-gb"
                % (workers, est_per_worker, estimated_total, avail_gb),
                RuntimeWarning,
                stacklevel=2,
            )
    except ImportError:
        pass
    plan_dict["mem_limit_gb"] = mem_limit_gb if mem_limit_gb else None
    run_start = time.perf_counter()
    completed_count = len(done)
    tile_results: dict[int, dict] = {}
    errors: list[dict] = []
    mp_ctx = __import__("multiprocessing").get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
            futures = {
                pool.submit(
                    _tile_worker,
                    {"tile_idx": i, "tile_wkb": plan.tiles_wkb[i], "plan": plan_dict},
                ): i
                for i in pending
            }
            try:
                for fut in as_completed(futures):
                    tile_idx = futures[fut]
                    try:
                        result = fut.result()
                        tile_results[tile_idx] = result
                        completed_count += 1
                    except Exception as exc:
                        errors.append({"tile_idx": tile_idx, "error": str(exc)})
            except KeyboardInterrupt:
                warnings.warn(
                    "Interrupted — partial results saved; rerun to resume.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                warnings.warn(
                    "Worker pool broken (%s) — saving progress and exiting. Rerun to resume."
                    % (exc,),
                    RuntimeWarning,
                    stacklevel=2,
                )
                pool.shutdown(wait=False, cancel_futures=True)
    finally:
        pass
    total_elapsed = time.perf_counter() - run_start
    if errors:
        warnings.warn(
            "Failed tiles: %s" % ([e["tile_idx"] for e in errors],),
            RuntimeWarning,
            stacklevel=2,
        )
    return _assemble_manifest(
        out_dir,
        plan,
        total_tiles,
        tile_results=tile_results,
        errors=errors,
        elapsed=total_elapsed,
    )


def run_parallel(
    config_path: str,
    tile_plan_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict:
    cfg = _apply_cfg_overrides(load_config(config_path), overrides)
    return run_parallel_cfg(cfg, tile_plan_path=tile_plan_path)


def _assemble_manifest(
    out_dir: Path,
    plan: TilePlan,
    total_tiles: int,
    *,
    tile_results: dict | None = None,
    errors: list | None = None,
    elapsed: float = 0.0,
) -> dict:
    """Assemble manifest.json from tile_results and any pre-existing tile JSON files."""
    tiles_dir = out_dir / "tiles"
    tile_records = []
    for i in range(total_tiles):
        npz = tiles_dir / f"tile_{i:04d}.npz"
        meta_f = tiles_dir / f"tile_{i:04d}.json"
        if not npz.exists():
            continue
        if tile_results and i in tile_results:
            tile_records.append(tile_results[i])
        elif meta_f.exists():
            pass

            tile_records.append(
                {"tile_idx": i, "npz_path": str(npz), "meta_path": str(meta_f)}
            )
    poi_layers = plan.poi_order
    manifest = {
        "run_id": plan.run_id,
        "region": plan.meta.get("region", ""),
        "group": plan.group,
        "comment": plan.comment,
        "created_at": plan.created_at,
        "config": plan.cfg_payload,
        "feature_groups": plan.feature_groups,
        "bus_channels": ["stops", "edges"],
        "poi_layers": poi_layers,
        "bus_bases": [f"poi_{l}" for l in poi_layers],
        "tensor_schema": plan.tensor_schema,
        "tiles": tile_records,
        "mode": "postgis_training_data_parallel",
        "poi_dataset_key": plan.poi_dataset_key,
        "road_graph_key": plan.road_graph_key,
        "bus_graph_key": plan.bus_graph_key,
        "grid_width": plan.grid_width,
        "grid_height": plan.grid_height,
        "pixel_size_m": plan.pixel_size_m,
        "total_planned_tiles": total_tiles,
        "completed_tiles": len(tile_records),
        "failed_tiles": errors or [],
        "elapsed_s": round(elapsed, 2),
    }
    save_manifest(out_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Config-driven PostGIS training data factory with resume support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Config module path, e.g. path.to.tile_factory_config",
    )
    parser.add_argument(
        "--tile-plan-path",
        default=None,
        help="External tile_plan.pkl to override sampling (from a prior run).",
    )
    parser.add_argument(
        "--run-id", default=None, help="Optional override for cfg.RUN_ID"
    )
    parser.add_argument(
        "--poi-dataset-key",
        default=None,
        help="Optional override for cfg.POI_DATASET_KEY",
    )
    parser.add_argument(
        "--road-graph-key",
        default=None,
        help="Optional override for cfg.ROAD_GRAPH_KEY",
    )
    parser.add_argument(
        "--bus-graph-key", default=None, help="Optional override for cfg.BUS_GRAPH_KEY"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Optional override for cfg.OUTPUT_DIR"
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=None,
        help="Optional override for cfg.GRID_WIDTH",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=None,
        help="Optional override for cfg.GRID_HEIGHT",
    )
    parser.add_argument(
        "--pixel-size-m",
        type=float,
        default=None,
        help="Optional override for cfg.PIXEL_SIZE_M",
    )
    parser.add_argument(
        "--graph-buffer-m",
        type=float,
        default=None,
        help="Optional override for cfg.GRAPH_BUFFER_M",
    )
    parser.add_argument(
        "--poi-buffer-m",
        type=float,
        default=None,
        help="Optional override for cfg.POI_BUFFER_M",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional override for cfg.MAX_SAMPLES",
    )
    parser.add_argument(
        "--tile-seed",
        type=int,
        default=None,
        help="Optional override for cfg.TILE_SEED",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional override for cfg.TILE_WORKERS",
    )
    parser.add_argument(
        "--worker-mem-limit-gb",
        type=float,
        default=None,
        help="Optional override for cfg.WORKER_MEM_LIMIT_GB",
    )
    parser.add_argument(
        "--allow-poi-layer-order-fallback",
        action="store_true",
        help="Debug only: allow alphabetical POI order if metadata lacks alignment_rows.",
    )
    args = parser.parse_args()
    overrides = {
        "RUN_ID": args.run_id,
        "POI_DATASET_KEY": args.poi_dataset_key,
        "ROAD_GRAPH_KEY": args.road_graph_key,
        "BUS_GRAPH_KEY": args.bus_graph_key,
        "OUTPUT_DIR": Path(args.output_dir) if args.output_dir else None,
        "GRID_WIDTH": args.grid_width,
        "GRID_HEIGHT": args.grid_height,
        "PIXEL_SIZE_M": args.pixel_size_m,
        "GRAPH_BUFFER_M": args.graph_buffer_m,
        "POI_BUFFER_M": args.poi_buffer_m,
        "MAX_SAMPLES": args.max_samples,
        "TILE_SEED": args.tile_seed,
        "TILE_WORKERS": args.workers,
        "WORKER_MEM_LIMIT_GB": args.worker_mem_limit_gb,
        "ALLOW_POI_LAYER_ORDER_FALLBACK": (
            True if args.allow_poi_layer_order_fallback else None
        ),
    }
    result = run_parallel(
        args.config, tile_plan_path=args.tile_plan_path, overrides=overrides
    )
    import json

    summary = {k: v for k, v in result.items() if k not in ("tiles", "failed_tiles")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
