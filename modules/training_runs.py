from __future__ import annotations
import warnings
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np
import pandas as pd


@dataclass
class TilePaths:
    tile_idx: int
    npz_path: Path
    meta_path: Path
    spec_path: Optional[Path] = None


def resolve_manifest_asset_path(manifest_path: Path, asset_path: Path | str) -> Path:
    manifest_path = Path(manifest_path).resolve()
    path = Path(asset_path)
    if path.exists():
        return path
    if path.is_absolute():
        return path
    for parent in manifest_path.parents:
        candidate = parent / path
        if candidate.exists():
            return candidate
    manifest_dir = manifest_path.parent
    for anchor in ("tiles", "tile_specs"):
        if anchor in path.parts:
            tail = path.parts[path.parts.index(anchor) :]
            candidate = manifest_dir.joinpath(*tail)
            if candidate.exists():
                return candidate
            return candidate
    candidate = manifest_dir / path.name
    if candidate.exists():
        return candidate
    return manifest_dir / path


def normalize_manifest_tile_records(manifest_path: Path, manifest: Dict) -> Dict:
    manifest = dict(manifest)
    normalized_tiles = []
    for tile in manifest.get("tiles", []):
        record = dict(tile)
        if record.get("npz_path"):
            record["npz_path"] = str(
                resolve_manifest_asset_path(manifest_path, record["npz_path"])
            )
        if record.get("meta_path"):
            record["meta_path"] = str(
                resolve_manifest_asset_path(manifest_path, record["meta_path"])
            )
        normalized_tiles.append(record)
    manifest["tiles"] = normalized_tiles
    return manifest


def save_tile_npz(
    out_dir: Path, tile_idx: int, arrays: Dict[str, np.ndarray], meta: Dict
) -> TilePaths:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"tile_{tile_idx:04d}"
    np.savez_compressed(base.with_suffix(".npz"), **arrays)
    with base.with_suffix(".json").open("w") as f:
        json.dump(meta, f, indent=2)
    return TilePaths(
        tile_idx=tile_idx,
        npz_path=base.with_suffix(".npz"),
        meta_path=base.with_suffix(".json"),
    )


def save_manifest(out_dir: Path, manifest: Dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    return path


def list_training_runs(output_root: Path) -> pd.DataFrame:
    rows = []
    for manifest in output_root.glob("*/manifest.json"):
        try:
            with manifest.open() as f:
                data = json.load(f)
            rows.append(
                {
                    "run_id": data.get("run_id"),
                    "region": data.get("region"),
                    "group": data.get("group"),
                    "comment": data.get("comment"),
                    "timestamp": data.get("created_at"),
                    "path": str(manifest),
                    "tiles": len(data.get("tiles", [])),
                }
            )
        except Exception as exc:
            warnings.warn(
                "Failed to read manifest %s: %s" % (manifest, exc),
                RuntimeWarning,
                stacklevel=2,
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "region",
                "group",
                "comment",
                "timestamp",
                "path",
                "tiles",
            ]
        )
    df = pd.DataFrame(rows)
    df.sort_values("timestamp", ascending=False, inplace=True)
    return df


WALK_TRANSIT_TOTAL_KEY = "access_walk_transit_total"
_ACCESS_ALL_KEY = "access_all"


def _walk_transit_total_from_all(all_arr: np.ndarray) -> np.ndarray:
    """Extract the leading walk_transit (total access) block from the stacked access_all tensor."""
    n = int(all_arr.shape[0])
    if n == 0 or n % 4 != 0:
        raise ValueError(
            f"Cannot derive '{WALK_TRANSIT_TOTAL_KEY}': access_all has {n} channels, expected a non-zero multiple of 4 (walk_transit | walk_transit_bus | walk_transit_rail | bus_delta)."
        )
    block = n // 4
    return np.array(all_arr[:block])


def list_tile_array_keys(npz_path: Path) -> List[str]:
    with np.load(npz_path, allow_pickle=False) as npz:
        files = list(npz.files)
    if WALK_TRANSIT_TOTAL_KEY not in files and _ACCESS_ALL_KEY in files:
        files.append(WALK_TRANSIT_TOTAL_KEY)
    return files


def load_tile_arrays(
    npz_path: Path, keys: Optional[Sequence[str]] = None
) -> Dict[str, np.ndarray]:
    arrays = {}
    with np.load(npz_path, allow_pickle=False) as npz:
        files = npz.files
        want_total = (
            WALK_TRANSIT_TOTAL_KEY not in files
            and _ACCESS_ALL_KEY in files
            and (keys is None or WALK_TRANSIT_TOTAL_KEY in keys)
        )
        selected_keys = (
            list(files) if keys is None else [key for key in keys if key in files]
        )
        for key in selected_keys:
            arrays[key] = np.array(npz[key])
        if want_total:
            arrays[WALK_TRANSIT_TOTAL_KEY] = _walk_transit_total_from_all(
                npz[_ACCESS_ALL_KEY]
            )
    return arrays


def load_tile_meta(meta_path: Path) -> Dict:
    with meta_path.open() as f:
        meta = json.load(f)
    return meta


def load_tile(
    npz_path: Path,
    meta_path: Path,
    keys: Optional[Sequence[str]] = None,
    *,
    load_meta: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    arrays = load_tile_arrays(npz_path, keys=keys)
    meta = load_tile_meta(meta_path) if load_meta else {}
    return (arrays, meta)


def load_run(manifest_path: Path, limit: Optional[int] = None):
    with manifest_path.open() as f:
        manifest = normalize_manifest_tile_records(manifest_path, json.load(f))
    tiles = []
    for idx, tile in enumerate(manifest.get("tiles", []), 1):
        arrays, meta = load_tile(Path(tile["npz_path"]), Path(tile["meta_path"]))
        tiles.append(
            {
                "arrays": arrays,
                "meta": meta,
                "tile_idx": tile["tile_idx"],
                "run_id": manifest.get("run_id"),
                "npz_path": tile["npz_path"],
                "meta_path": tile["meta_path"],
            }
        )
        if limit is not None and idx >= int(limit):
            break
    return (manifest, tiles)


def load_run_lazy(manifest_path: Path, limit: Optional[int] = None):
    """Return manifest and tiles without loading arrays to keep memory low."""
    with manifest_path.open() as f:
        manifest = normalize_manifest_tile_records(manifest_path, json.load(f))
    tiles = []
    for idx, tile in enumerate(manifest.get("tiles", []), 1):
        tiles.append(
            {
                "tile_idx": tile["tile_idx"],
                "run_id": manifest.get("run_id"),
                "npz_path": tile["npz_path"],
                "meta_path": tile["meta_path"],
            }
        )
        if limit is not None and idx >= int(limit):
            break
    return (manifest, tiles)


def _pick_tensor_key(
    arrays: Dict[str, np.ndarray], preferred, fallback_prefix: Optional[str] = None
) -> Optional[str]:
    for key in preferred:
        if key and key in arrays:
            return key
    return None


SIGNATURE_KEYS = [
    "H3_RES",
    "GRID_WIDTH",
    "GRID_HEIGHT",
    "GRID_CRS",
    "PIXEL_SIZE_M",
    "HULL_BUFFER_M",
    "RASTER_METHOD",
    "POI_SMOOTH",
    "POI_SMOOTH_SIGMA_PX",
    "POI_BOOST",
    "POI_MIN_VALUE",
    "POI_USE_MASK",
    "BUS_TENSOR_ENABLED",
    "BUS_MODE",
    "BUS_TILE_BUFFER_M",
    "BUS_EDGE_RENDER_MODE",
    "BUS_PATH_RENDER_WEIGHT_MODE",
    "BUS_PATH_ROUTING_DIRECTION",
    "BUS_SMOOTH_STOPS",
    "BUS_SMOOTH_STOPS_SIGMA",
    "BUS_SMOOTH_EDGES",
    "BUS_SMOOTH_EDGES_SIGMA",
    "BUS_STOP_CLIP",
]


def _config_signature(cfg: Dict) -> str:
    parts = []
    for k in SIGNATURE_KEYS:
        parts.append(f"{k}={cfg.get(k)}")
    return "|".join(parts)


def summarize_training_run(manifest_path: Path, inspect_tile: bool = True) -> Dict:
    """Return metadata summary for a run (grid size, channels from first tile, etc.)."""
    with manifest_path.open() as f:
        manifest = normalize_manifest_tile_records(manifest_path, json.load(f))
    cfg = manifest.get("config", {})
    tile_entries = manifest.get("tiles", [])
    bus_ch = rail_ch = poi_ch = target_ch = None
    tile_height = tile_width = None
    tile_meta_height = tile_meta_width = None
    if inspect_tile and tile_entries:
        meta_path = Path(tile_entries[0]["meta_path"])
        try:
            with meta_path.open() as f:
                tile_meta = json.load(f)
            grid_meta = tile_meta.get("grid") or {}
            tile_meta_width = grid_meta.get("width")
            tile_meta_height = grid_meta.get("height")
        except Exception as exc:
            warnings.warn(
                "Failed to read tile meta for %s: %s" % (manifest_path, exc),
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            arrays, _ = load_tile(
                Path(tile_entries[0]["npz_path"]), Path(tile_entries[0]["meta_path"])
            )
            bus_key = _pick_tensor_key(arrays, ["bus_tensor"])
            rail_key = _pick_tensor_key(
                arrays,
                ["access_walk_transit_rail", "walk_transit_rail"],
                fallback_prefix="access_",
            )
            poi_key = _pick_tensor_key(arrays, ["poi_tensor"])
            target_key = _pick_tensor_key(
                arrays, ["access_bus_delta", "bus_delta"], fallback_prefix="access_"
            )
            if bus_key:
                bus_ch = arrays[bus_key].shape[0]
            if rail_key:
                rail_ch = arrays[rail_key].shape[0]
                tile_height, tile_width = (
                    arrays[rail_key].shape[1],
                    arrays[rail_key].shape[2],
                )
            if poi_key:
                poi_ch = arrays[poi_key].shape[0]
                tile_height = tile_height or arrays[poi_key].shape[1]
                tile_width = tile_width or arrays[poi_key].shape[2]
            if target_key:
                target_ch = arrays[target_key].shape[0]
                tile_height = tile_height or arrays[target_key].shape[1]
                tile_width = tile_width or arrays[target_key].shape[2]
        except Exception as exc:
            warnings.warn(
                "Failed to inspect tile arrays for %s: %s" % (manifest_path, exc),
                RuntimeWarning,
                stacklevel=2,
            )
    tile_width = tile_width or tile_meta_width
    tile_height = tile_height or tile_meta_height
    resolved_width = tile_width or cfg.get("GRID_WIDTH")
    resolved_height = tile_height or cfg.get("GRID_HEIGHT")
    if resolved_width is None or resolved_height is None:
        warnings.warn(
            "Could not determine tile dimensions for %s (width=%s, height=%s). This run may be grouped incorrectly."
            % (manifest_path, resolved_width, resolved_height),
            RuntimeWarning,
            stacklevel=2,
        )
    signature = _config_signature(cfg)
    signature = f"{signature}|TILE_WIDTH={resolved_width}|TILE_HEIGHT={resolved_height}"
    return {
        "run_id": manifest.get("run_id"),
        "region": manifest.get("region"),
        "group": manifest.get("group"),
        "comment": manifest.get("comment"),
        "timestamp": manifest.get("created_at"),
        "path": str(manifest_path),
        "tiles": len(tile_entries),
        "grid_width": cfg.get("GRID_WIDTH"),
        "grid_height": cfg.get("GRID_HEIGHT"),
        "bus_channels": bus_ch,
        "rail_channels": rail_ch,
        "poi_channels": poi_ch,
        "target_channels": target_ch,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "signature": signature,
    }


def list_training_runs_detailed(
    output_root: Path, inspect_tile: bool = True
) -> pd.DataFrame:
    rows = []
    for manifest in output_root.glob("*/manifest.json"):
        try:
            rows.append(summarize_training_run(manifest, inspect_tile=inspect_tile))
        except Exception as exc:
            warnings.warn(
                "Failed to summarize manifest %s: %s" % (manifest, exc),
                RuntimeWarning,
                stacklevel=2,
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "region",
                "group",
                "comment",
                "timestamp",
                "path",
                "tiles",
                "grid_width",
                "grid_height",
                "bus_channels",
                "rail_channels",
                "poi_channels",
                "target_channels",
                "tile_width",
                "tile_height",
            ]
        )
    df = pd.DataFrame(rows)
    df.sort_values("timestamp", ascending=False, inplace=True)
    return df


def group_training_runs(output_root: Path) -> pd.DataFrame:
    """Group runs by configuration signature (grid/smoothing/bus/poi settings, ignoring region)."""
    df = list_training_runs_detailed(output_root, inspect_tile=True)
    if df.empty:
        return df
    valid_mask = df["tile_width"].notna() & df["tile_height"].notna()
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        invalid_runs = df[~valid_mask]["run_id"].tolist()
        warnings.warn(
            "Excluding %d run(s) with indeterminate tile dimensions from grouping: %s"
            % (invalid_count, invalid_runs),
            RuntimeWarning,
            stacklevel=2,
        )
        df = df[valid_mask]
    if df.empty:
        warnings.warn(
            "No valid runs remaining after filtering out runs with indeterminate dimensions",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame()
    grouped = (
        df.groupby("signature")
        .agg(
            run_ids=("run_id", list),
            run_paths=("path", list),
            regions=("region", list),
            count=("run_id", "count"),
            samples=("tiles", "sum"),
            sample_path=("path", "first"),
            grid_width=("grid_width", "first"),
            grid_height=("grid_height", "first"),
            bus_channels=("bus_channels", "first"),
            rail_channels=("rail_channels", "first"),
            poi_channels=("poi_channels", "first"),
            target_channels=("target_channels", "first"),
        )
        .reset_index()
    )
    return grouped


__all__ = [
    "save_tile_npz",
    "save_manifest",
    "list_training_runs",
    "list_training_runs_detailed",
    "group_training_runs",
    "list_tile_array_keys",
    "load_tile_arrays",
    "load_tile_meta",
    "load_tile",
    "resolve_manifest_asset_path",
    "normalize_manifest_tile_records",
    "load_run",
    "load_run_lazy",
    "summarize_training_run",
]
