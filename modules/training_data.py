"""Tile-factory primitives: tiling, rasterization, POI and auxiliary layers, bus tensors."""

from __future__ import annotations
import warnings
import math
import multiprocessing as mp
from typing import List, Mapping, Sequence, Tuple
import numpy as np
import pandas as pd
import pyproj

try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
except ImportError:
    gpd = None
    Point = box = unary_union = STRtree = None
try:
    import h3
except ImportError:
    h3 = None
from modules.h3_functions import estimate_h3_cell_area_m2, _polyfill_geom, id_to_polygon


def clip_cells(cells_gdf: gpd.GeoDataFrame | pd.DataFrame, polygon):
    if polygon is None:
        return cells_gdf
    if "geometry" not in cells_gdf.columns:
        if cells_gdf.empty:
            return cells_gdf
        import h3

        sample = cells_gdf["h3"].iloc[0]
        try:
            res = h3.get_resolution(sample)
        except AttributeError:
            res = h3.h3_get_resolution(sample)
        buffer_deg = min(estimate_h3_cell_area_m2(res) ** 0.5 * 2 / 111320.0, 0.05)
        poly_buf = polygon.buffer(buffer_deg)
        if poly_buf.geom_type == "Polygon":
            geo_shape = poly_buf.__geo_interface__
        elif poly_buf.geom_type == "MultiPolygon":
            candidates = set()
            for geom in poly_buf.geoms:
                candidates.update(_polyfill_geom(geom, res))
            candidates = list(candidates)
            geo_shape = None
        else:
            geo_shape = poly_buf.__geo_interface__
        if geo_shape:
            candidates = _polyfill_geom(poly_buf, res)
        subset = cells_gdf[cells_gdf["h3"].isin(candidates)].copy()
        if subset.empty:
            df_empty = gpd.GeoDataFrame(subset, geometry=[], crs="EPSG:4326")
            return df_empty
        subset["geometry"] = subset["h3"].apply(id_to_polygon)
        subset = gpd.GeoDataFrame(subset, geometry="geometry", crs="EPSG:4326")
        if str(subset.crs) != str(getattr(polygon, "crs", "EPSG:4326")):
            pass
        clipped = subset.loc[subset.geometry.intersects(polygon)].copy()
        clipped.reset_index(drop=True, inplace=True)
        return clipped
    clipped = cells_gdf.loc[cells_gdf.geometry.intersects(polygon)].copy()
    clipped.reset_index(drop=True, inplace=True)
    return clipped


def region_union_poly(cells_gdf: gpd.GeoDataFrame, crs_out="EPSG:3857"):
    if cells_gdf.empty:
        raise ValueError("cells_gdf is empty")
    if "geometry" not in cells_gdf.columns:
        if "h3" not in cells_gdf.columns:
            raise ValueError(
                "Cannot compute region union: neither geometry nor h3 column present."
            )
        import numpy as np
        from scipy.spatial import ConvexHull
        from shapely.geometry import Polygon

        h3_vals = cells_gdf["h3"].values
        _sample_n = min(100000, len(h3_vals))
        if len(h3_vals) > _sample_n:
            h3_vals = h3_vals[np.random.choice(len(h3_vals), _sample_n, replace=False)]
        try:
            coords = np.array([h3.cell_to_latlng(c) for c in h3_vals], dtype=np.float32)
        except AttributeError:
            coords = np.array([h3.h3_to_geo(c) for c in h3_vals], dtype=np.float32)
        xy = coords[:, [1, 0]]
        del coords
        try:
            ch = ConvexHull(xy)
            hull_pts = xy[ch.vertices]
        except Exception:
            hull_pts = np.array(
                [
                    [xy[:, 0].min(), xy[:, 1].min()],
                    [xy[:, 0].max(), xy[:, 1].min()],
                    [xy[:, 0].max(), xy[:, 1].max()],
                    [xy[:, 0].min(), xy[:, 1].max()],
                ]
            )
        del xy
        hull = Polygon(hull_pts.tolist())
        if hull.is_empty:
            raise ValueError("cells convex hull is empty")
        return gpd.GeoSeries([hull], crs="EPSG:4326").to_crs(crs_out).iloc[0]
    region = unary_union(cells_gdf.geometry.values)
    if region.is_empty:
        raise ValueError("cells union is empty")
    return gpd.GeoSeries([region], crs=cells_gdf.crs).to_crs(crs_out).iloc[0]


class SimpleSpatialIndex:
    """A simple grid-based spatial index for faster collision checks."""

    def __init__(self, cell_size=20000):
        self.cell_size = cell_size
        self.grid = {}

    def _keys(self, bounds):
        minx, miny, maxx, maxy = bounds
        c_minx = int(math.floor(minx / self.cell_size))
        c_miny = int(math.floor(miny / self.cell_size))
        c_maxx = int(math.floor(maxx / self.cell_size))
        c_maxy = int(math.floor(maxy / self.cell_size))
        keys = []
        for x in range(c_minx, c_maxx + 1):
            for y in range(c_miny, c_maxy + 1):
                keys.append((x, y))
        return keys

    def insert(self, item, bounds):
        for key in self._keys(bounds):
            self.grid.setdefault(key, []).append(item)

    def query(self, bounds):
        candidates = []
        for key in self._keys(bounds):
            if key in self.grid:
                candidates.extend(self.grid[key])
        seen = set()
        unique = []
        for cand in candidates:
            if id(cand) not in seen:
                seen.add(id(cand))
                unique.append(cand)
        return unique


def sample_tile_within_hull(
    hull_proj, width, height, pixel_size, *, max_tries=200, rng=None
):
    minx, miny, maxx, maxy = hull_proj.bounds
    tile_w = width * pixel_size
    tile_h = height * pixel_size
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("width/height must be positive")
    if tile_w > maxx - minx or tile_h > maxy - miny:
        warnings.warn(
            "Tile size %.1fm x %.1fm exceeds hull bounds." % (tile_w, tile_h),
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    rng = rng or np.random.default_rng()
    for _ in range(max_tries):
        x0 = rng.uniform(minx, maxx - tile_w)
        y0 = rng.uniform(miny, maxy - tile_h)
        tile = box(x0, y0, x0 + tile_w, y0 + tile_h)
        if hull_proj.covers(tile):
            return tile
    return None


def sample_tiles_nonoverlap(
    cells_gdf: gpd.GeoDataFrame,
    *,
    width: int,
    height: int,
    pixel_size: float,
    crs_out: str = "EPSG:3857",
    max_samples: int | None = 3,
    overlap_tol: float = 0.2,
    max_overlap_stop: float = 0.5,
    max_tries_per_tile: int = 200,
    hull_buffer_m: float = 0.0,
    seed: int | None = None,
    sampling_boundary=None,
    initial_tiles: Sequence | None = None,
):
    if sampling_boundary is not None:
        hull_proj = sampling_boundary
    else:
        hull_proj = region_union_poly(cells_gdf, crs_out=crs_out)
    if hull_buffer_m != 0:
        hull_proj = hull_proj.buffer(hull_buffer_m)
        hull_proj = hull_proj.buffer(0)
        if hull_proj.is_empty:
            raise ValueError("Buffered hull became empty; adjust HULL_BUFFER_M.")
    tiles = list(initial_tiles or [])
    est_tile_size = pixel_size * width
    spatial_index = SimpleSpatialIndex(cell_size=est_tile_size * 5)
    rng = np.random.default_rng(seed)
    tries = 0
    target = max_samples if max_samples is not None else max_tries_per_tile
    for existing in tiles:
        spatial_index.insert(existing, existing.bounds)
    pbar = None
    if max_samples and max_samples > 100:
        from tqdm.auto import tqdm

        pbar = tqdm(total=max_samples, desc="Sampling tiles")
        if tiles:
            pbar.update(len(tiles))
    while (
        max_samples is None or len(tiles) < max_samples
    ) and tries < target * max_tries_per_tile:
        tries += 1
        tile = sample_tile_within_hull(
            hull_proj, width, height, pixel_size, max_tries=10, rng=rng
        )
        if tile is None:
            continue
        potential_overlaps = spatial_index.query(tile.bounds)
        overlaps = []
        for existing in potential_overlaps:
            if not tile.intersects(existing):
                continue
            inter = tile.intersection(existing)
            if inter.is_empty:
                continue
            overlap_frac = inter.area / tile.area
            overlaps.append(overlap_frac)
        if overlaps and max(overlaps) > overlap_tol:
            if (
                max_samples is not None
                and len(tiles) < max_samples
                and (tries > max_samples * max_tries_per_tile * 0.5)
            ):
                pass
            else:
                continue
        if overlaps and max(overlaps) > max_overlap_stop and (max_samples is None):
            break
        tiles.append(tile)
        spatial_index.insert(tile, tile.bounds)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()
    if max_samples is not None and len(tiles) < max_samples:
        warnings.warn(
            "Requested %s samples, placed %s (tries=%s). Consider relaxing overlap preference."
            % (max_samples, len(tiles), tries),
            RuntimeWarning,
            stacklevel=2,
        )
    return (tiles, hull_proj)


def build_grid_spec(
    cells_gdf,
    pixel_size_m,
    *,
    crs_out="EPSG:3857",
    clip_poly=None,
    width=None,
    height=None,
):
    use_cells = clip_cells(cells_gdf, clip_poly)
    if use_cells.empty:
        raise ValueError("No cells intersect the clip polygon.")
    if clip_poly is None or width is None or height is None:
        raise ValueError(
            "clip_poly, width, and height are required for fixed-size tiles."
        )
    clip_proj = gpd.GeoSeries([clip_poly], crs=use_cells.crs).to_crs(crs_out)
    minx, miny, maxx, maxy = clip_proj.total_bounds
    bounds = (minx, miny, minx + width * pixel_size_m, miny + height * pixel_size_m)
    transform = (bounds[0], pixel_size_m, 0.0, bounds[3], 0.0, -pixel_size_m)
    return {
        "crs": crs_out,
        "pixel_size": pixel_size_m,
        "bounds": bounds,
        "width": width,
        "height": height,
        "transform": transform,
        "clip_poly": clip_poly,
    }


def paired_features_by_suffix(df, suffix_a, suffix_b, id_cols):
    cols_a = [c for c in df.columns if c.endswith(suffix_a) and c not in id_cols]
    base_to_a = {c[: -len(suffix_a)]: c for c in cols_a}
    cols_a_sorted = []
    cols_b_sorted = []
    missing_bases = []
    for base in sorted(base_to_a):
        col_a = base_to_a[base]
        col_b = f"{base}{suffix_b}"
        if col_b in df.columns:
            cols_a_sorted.append(col_a)
            cols_b_sorted.append(col_b)
        else:
            missing_bases.append(base)
    if missing_bases:
        warnings.warn(
            "Suffix pair %s/%s missing %s counterparts for %s bases"
            % (suffix_a, suffix_b, suffix_b, len(missing_bases)),
            RuntimeWarning,
            stacklevel=2,
        )
    return (cols_a_sorted, cols_b_sorted)


def infer_poi_specs_from_run(
    feature_groups: Mapping[str, Sequence[str]], meta: Mapping
) -> List[str]:
    specs = []
    amen = meta.get("amenities")
    if amen:
        specs = list(amen)
    if not specs and "bus_delta" in feature_groups:
        specs = [c[: -len("_bus_delta")] for c in feature_groups["bus_delta"]]
    return specs


def _area_weighted_worker(args):
    (
        row_start,
        row_end,
        width,
        pixel,
        minx,
        maxy,
        geoms,
        geom_wkb_to_idx,
        values,
        bands,
        fill_value,
    ) = args
    tree = STRtree(geoms)
    arr_chunk = np.full((bands, row_end - row_start, width), fill_value, dtype=float)
    mask_chunk = np.zeros((row_end - row_start, width), dtype=bool)
    for local_r, r in enumerate(range(row_start, row_end)):
        y_top = maxy - r * pixel
        y_bottom = y_top - pixel
        for c in range(width):
            x_left = minx + c * pixel
            x_right = x_left + pixel
            pix_poly = box(x_left, y_bottom, x_right, y_top)
            hits = tree.query(pix_poly)
            if len(hits) == 0:
                continue
            total_area = 0.0
            accum = np.zeros(bands, dtype=float)
            for h in hits:
                idx = geom_wkb_to_idx.get(h.wkb) if hasattr(h, "wkb") else None
                if idx is None:
                    try:
                        idx = geoms.index(h)
                    except ValueError:
                        continue
                inter = pix_poly.intersection(geoms[idx])
                if inter.is_empty:
                    continue
                a = inter.area
                total_area += a
                accum += a * values[idx]
            if total_area > 0:
                arr_chunk[:, local_r, c] = accum / total_area
                mask_chunk[local_r, c] = True
    return (row_start, arr_chunk, mask_chunk)


def area_weighted_rasterize(
    cells_gdf, feature_cols, grid_spec, *, fill_value=0.0, n_workers=1, chunk_rows=32
):
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    minx, miny, maxx, maxy = grid_spec["bounds"]
    pixel = grid_spec["pixel_size"]
    width, height = (grid_spec["width"], grid_spec["height"])
    gdf = clip_cells(cells_gdf, grid_spec.get("clip_poly"))
    gdf = gdf.to_crs(grid_spec["crs"])
    values = gdf[feature_cols].to_numpy(dtype=float)
    geoms = list(gdf.geometry.values)
    geom_wkb_to_idx = {g.wkb: i for i, g in enumerate(geoms)}
    bands = len(feature_cols)
    if n_workers <= 1:
        return _area_weighted_worker(
            (
                0,
                height,
                width,
                pixel,
                minx,
                maxy,
                geoms,
                geom_wkb_to_idx,
                values,
                bands,
                fill_value,
            )
        )[1:]
    tasks = []
    for row_start in range(0, height, chunk_rows):
        row_end = min(height, row_start + chunk_rows)
        tasks.append(
            (
                row_start,
                row_end,
                width,
                pixel,
                minx,
                maxy,
                geoms,
                geom_wkb_to_idx,
                values,
                bands,
                fill_value,
            )
        )
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_area_weighted_worker, tasks)
    arr = np.full((bands, height, width), fill_value, dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    for row_start, arr_chunk, mask_chunk in results:
        r_end = row_start + arr_chunk.shape[1]
        arr[:, row_start:r_end, :] = arr_chunk
        mask[row_start:r_end, :] = mask_chunk
    return (arr, mask)


def _center_worker(args):
    (
        row_start,
        row_end,
        width,
        pixel,
        minx,
        maxy,
        h3_res,
        value_map,
        bands,
        fill_value,
        to_wgs84,
    ) = args
    arr_chunk = np.full((bands, row_end - row_start, width), fill_value, dtype=float)
    mask_chunk = np.zeros((row_end - row_start, width), dtype=bool)
    for local_r, r in enumerate(range(row_start, row_end)):
        cy = maxy - (r + 0.5) * pixel
        for c in range(width):
            cx = minx + (c + 0.5) * pixel
            lon, lat = to_wgs84(cx, cy)
            h3_id = h3.latlng_to_cell(lat, lon, h3_res)
            vals = value_map.get(h3_id)
            if vals is not None:
                arr_chunk[:, local_r, c] = vals
                mask_chunk[local_r, c] = True
    return (row_start, arr_chunk, mask_chunk)


def center_sample_rasterize(
    cells_gdf,
    feature_cols,
    grid_spec,
    *,
    h3_res=10,
    fill_value=0.0,
    n_workers=1,
    chunk_rows=128,
):
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    minx, miny, maxx, maxy = grid_spec["bounds"]
    pixel = grid_spec["pixel_size"]
    width, height = (grid_spec["width"], grid_spec["height"])
    values = cells_gdf[feature_cols].to_numpy(dtype=float)
    value_map = {h: vals for h, vals in zip(cells_gdf["h3"], values)}
    to_wgs84 = pyproj.Transformer.from_crs(
        grid_spec["crs"], "EPSG:4326", always_xy=True
    ).transform
    bands = len(feature_cols)
    if n_workers <= 1:
        return _center_worker(
            (
                0,
                height,
                width,
                pixel,
                minx,
                maxy,
                h3_res,
                value_map,
                bands,
                fill_value,
                to_wgs84,
            )
        )[1:]
    tasks = []
    for row_start in range(0, height, chunk_rows):
        row_end = min(height, row_start + chunk_rows)
        tasks.append(
            (
                row_start,
                row_end,
                width,
                pixel,
                minx,
                maxy,
                h3_res,
                value_map,
                bands,
                fill_value,
                to_wgs84,
            )
        )
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_center_worker, tasks)
    arr = np.full((bands, height, width), fill_value, dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    for row_start, arr_chunk, mask_chunk in results:
        r_end = row_start + arr_chunk.shape[1]
        arr[:, row_start:r_end, :] = arr_chunk
        mask[row_start:r_end, :] = mask_chunk
    return (arr, mask)


def tensor_metadata(
    grid_spec, feature_cols, meta, *, method, fill_value, band_dtype, tile_idx=None
):
    minx, miny, maxx, maxy = grid_spec["bounds"]
    to_wgs84 = pyproj.Transformer.from_crs(
        grid_spec["crs"], "EPSG:4326", always_xy=True
    ).transform
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    corners_wgs84 = [to_wgs84(x, y) for x, y in corners]
    return {
        "grid": {
            "crs": grid_spec["crs"],
            "pixel_size_m": grid_spec["pixel_size"],
            "bounds": grid_spec["bounds"],
            "bounds_wgs84": corners_wgs84,
            "width": grid_spec["width"],
            "height": grid_spec["height"],
            "transform": grid_spec["transform"],
            "tile_idx": tile_idx,
        },
        "features": list(feature_cols),
        "run": {
            "run_id": meta.get("run_id"),
            "region": meta.get("region"),
            "h3_res": meta.get("h3_res"),
        },
        "interpolation": {"method": method, "fill_value": fill_value},
        "array": {"dtype": str(band_dtype), "bands": len(feature_cols)},
    }


def _gaussian_splat_pixels(
    col_f: float, row_f: float, H: int, W: int, sigma_px: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel indices and Gaussian weights for a single point splat."""
    radius_px = int(np.ceil(3.0 * sigma_px))
    col_lo = max(0, int(col_f) - radius_px)
    col_hi = min(W - 1, int(col_f) + radius_px)
    row_lo = max(0, int(row_f) - radius_px)
    row_hi = min(H - 1, int(row_f) + radius_px)
    if col_lo > col_hi or row_lo > row_hi:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    cols = np.arange(col_lo, col_hi + 1)
    rows = np.arange(row_lo, row_hi + 1)
    cc, rr = np.meshgrid(cols, rows)
    dist_sq = (cc - col_f) ** 2 + (rr - row_f) ** 2
    w = np.exp(-0.5 * dist_sq / sigma_px**2).astype(np.float32)
    mask = w > 1e-06
    pixel_idx = (rr[mask] * W + cc[mask]).astype(np.int64)
    w_mask = w[mask]
    if w_mask.size:
        w_max = float(w_mask.max())
        if w_max > 0:
            w_mask = (w_mask / w_max).astype(np.float32)
    return (pixel_idx, w_mask)


def rasterize_poi_splats(
    pois_gdf,
    grid_spec,
    *,
    smooth: bool = True,
    sigma_px: float = 1.5,
    peak_value: float = 1.0,
    weight_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize POIs as Gaussian splats using sub-pixel precise coordinates."""
    H, W = (grid_spec["height"], grid_spec["width"])
    arr = np.zeros((H, W), dtype=np.float32)
    if pois_gdf is None or pois_gdf.empty:
        return (arr, arr > 0)
    gdf_proj = pois_gdf.to_crs(grid_spec["crs"])
    valid_geom = np.array(
        [geom is not None and (not geom.is_empty) for geom in gdf_proj.geometry],
        dtype=bool,
    )
    gdf_proj = gdf_proj.loc[valid_geom]
    if gdf_proj.empty:
        return (arr, arr > 0)
    pts = gdf_proj.geometry.centroid
    xs = pts.x.to_numpy()
    ys = pts.y.to_numpy()
    active = np.ones(len(xs), dtype=bool)
    if weight_col and weight_col in gdf_proj.columns:
        w = gdf_proj[weight_col].to_numpy(dtype=float)
        if not np.all(np.isnan(w)):
            active = np.where(np.isnan(w), True, w > 0)
    xs = xs[active]
    ys = ys[active]
    if len(xs) == 0:
        return (arr, arr > 0)
    minx, _miny, _maxx, maxy = grid_spec["bounds"]
    pixel = grid_spec["pixel_size"]
    col_f = (xs - minx) / pixel
    row_f = (maxy - ys) / pixel
    peak_value = float(peak_value)
    if not smooth or sigma_px is None or sigma_px <= 0:
        cols = np.clip(col_f.astype(int), 0, W - 1)
        rows = np.clip(row_f.astype(int), 0, H - 1)
        in_bounds = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
        arr[rows[in_bounds], cols[in_bounds]] = peak_value
    else:
        for cf, rf in zip(col_f, row_f):
            pix_idx, w = _gaussian_splat_pixels(cf, rf, H, W, sigma_px)
            if pix_idx.size == 0:
                continue
            rows_i, cols_i = np.divmod(pix_idx, W)
            np.maximum.at(arr, (rows_i, cols_i), w * peak_value)
    mask = arr > 0
    return (arr, mask)


def build_poi_tensor(
    pois_by_layer,
    layer_order,
    grid_spec,
    *,
    smooth=True,
    sigma_px=1.5,
    boost=1.0,
    min_value=0.0,
    clip_min=0.0,
    clip_max=1.0,
    kernel_peak=1.0,
):
    bands = len(layer_order)
    tensor = np.zeros((bands, grid_spec["height"], grid_spec["width"]), dtype=float)
    band_masks = []
    mask_union = np.zeros((grid_spec["height"], grid_spec["width"]), dtype=bool)
    for i, layer in enumerate(layer_order):
        gdf = pois_by_layer.get(layer)
        weight_col = "weight" if gdf is not None and "weight" in gdf.columns else None
        arr, msk = rasterize_poi_splats(
            gdf,
            grid_spec,
            smooth=smooth,
            sigma_px=sigma_px,
            peak_value=kernel_peak,
            weight_col=weight_col,
        )
        if boost != 1.0:
            arr = arr * boost
        if min_value and min_value > 0:
            arr = np.where(arr > 0, np.maximum(arr, min_value), arr)
        if clip_max is not None:
            np.clip(arr, clip_min, clip_max, out=arr)
        elif clip_min not in (None, 0):
            np.maximum(arr, clip_min, out=arr)
        tensor[i] = arr
        band_masks.append(msk)
        mask_union |= msk
    return (tensor, band_masks, mask_union)


__all__ = [
    "clip_cells",
    "sample_tiles_nonoverlap",
    "build_grid_spec",
    "paired_features_by_suffix",
    "infer_poi_specs_from_run",
    "area_weighted_rasterize",
    "center_sample_rasterize",
    "tensor_metadata",
    "rasterize_poi_splats",
    "build_poi_tensor",
]
