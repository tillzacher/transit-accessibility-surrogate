"""Geographic-holdout tile splitting for region-generalization validation."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _tile_meta(tile: Dict) -> Optional[Dict]:
    meta = tile.get("meta")
    if isinstance(meta, dict):
        return meta
    meta_path = tile.get("meta_path")
    if meta_path and Path(meta_path).exists():
        with open(meta_path) as fh:
            return json.load(fh)
    return None


def tile_box_3857(tile: Dict):
    """Return the tile footprint as a shapely box in EPSG:3857 (from meta grid.bounds)."""
    from shapely.geometry import box

    meta = _tile_meta(tile)
    grid = (meta or {}).get("grid") or {}
    bounds = grid.get("bounds")
    if not bounds or len(bounds) != 4:
        raise ValueError(
            f"tile {tile.get('tile_idx')} has no grid.bounds in meta; cannot place it geographically"
        )
    crs = str(grid.get("crs") or "EPSG:3857")
    if crs not in ("EPSG:3857", "epsg:3857", "3857"):
        raise ValueError(f"tile grid CRS is {crs!r}; geo_split assumes EPSG:3857")
    return box(float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))


def _to_3857(geom_4326):
    from pyproj import Transformer
    from shapely.ops import transform

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transform(lambda x, y, z=None: tr.transform(x, y), geom_4326)


def resolve_region_polygon(spec: Dict[str, Any]):
    """Resolve a holdout_region spec to a shapely polygon in EPSG:3857."""
    from shapely.geometry import Point, box, shape

    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("holdout_region list is empty")
        from shapely.ops import unary_union

        return unary_union([resolve_region_polygon(s) for s in spec])
    if not isinstance(spec, dict):
        raise ValueError(
            "holdout_region must be a dict or list of dicts (see geo_split.resolve_region_polygon)"
        )
    kind = str(spec.get("kind", "bbox")).lower()
    if kind == "bbox":
        lon_min, lat_min, lon_max, lat_max = spec["bounds"]
        return _to_3857(
            box(float(lon_min), float(lat_min), float(lon_max), float(lat_max))
        )
    if kind == "radius":
        center = _to_3857(Point(float(spec["lon"]), float(spec["lat"])))
        return center.buffer(float(spec["radius_km"]) * 1000.0)
    if kind == "admin":
        path = spec.get("geojson")
        if not path:
            raise NotImplementedError(
                "admin region needs a 'geojson' path (EPSG:4326) under data/regions/"
            )
        gj = json.load(open(path))
        if gj.get("type") == "FeatureCollection":
            from shapely.ops import unary_union

            geom = unary_union([shape(f["geometry"]) for f in gj["features"]])
        else:
            geom = shape(gj.get("geometry", gj))
        return _to_3857(geom)
    raise ValueError(f"unknown holdout_region kind {kind!r}")


def split_tiles_geographic(
    tiles: List[Dict],
    holdout_region: Dict[str, Any],
    buffer_m: float = 24000.0,
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
    dump_path: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, Any]]:
    """Split tiles into (train, val, test) where test = held-out region interior."""
    region = resolve_region_polygon(holdout_region)
    buffered = region.buffer(float(buffer_m))
    train_pool, test, dropped = ([], [], [])
    assignment: Dict[Any, str] = {}
    for t in tiles:
        g = tile_box_3857(t)
        tid = t.get("tile_idx")
        if region.contains(g.centroid):
            test.append(t)
            assignment[tid] = "test_heldout"
        elif buffered.intersects(g):
            dropped.append(t)
            assignment[tid] = "dropped_buffer"
        else:
            train_pool.append(t)
            assignment[tid] = "train_pool"
    rng = np.random.default_rng(seed)
    order = list(range(len(train_pool)))
    rng.shuffle(order)
    n_val = int(round(float(val_fraction) * len(train_pool)))
    val_set = set(order[:n_val])
    train = [train_pool[i] for i in range(len(train_pool)) if i not in val_set]
    val = [train_pool[i] for i in range(len(train_pool)) if i in val_set]
    for i in val_set:
        assignment[train_pool[i].get("tile_idx")] = "val"
    summary = {
        "split_mode": "geographic",
        "holdout_region": holdout_region,
        "buffer_m": float(buffer_m),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "counts": {
            "train": len(train),
            "val": len(val),
            "test_heldout": len(test),
            "dropped_buffer": len(dropped),
        },
    }
    if not test:
        raise ValueError(
            f"Geographic holdout produced 0 test tiles — the region does not contain any tile centroid. Check holdout_region (in lon/lat): {holdout_region!r}"
        )
    if dump_path:
        Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w") as fh:
            json.dump({**summary, "assignment": assignment}, fh, default=str)
    return (train, val, test, summary)
