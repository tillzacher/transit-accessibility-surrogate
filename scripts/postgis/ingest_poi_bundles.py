#!/usr/bin/env python3
"""Ingest the POI and workplace layers of one MAPPER run into poi.*.

POIs come from the run's `pois.csv`, workplaces from `src_wrk_<mode>.csv` placed at the nodes of
that mode's graph (--workplace-graph). Layers are aligned with the run's
`<base>_walk_transit_bus` / `<base>_bus_delta` and `<base>_walk_transit_rail` columns. The
alignment is stored in poi.datasets.metadata_json["alignment_rows"]; the tile factory takes
its POI channel order from there. Missing layers are an error unless --allow-incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
from psycopg2.extras import Json, execute_values

from _common import get_logger, jsonable, read_run_meta, run_column_names
from _mapper_graph import read_mapper_graph

from modules.postgis_bridge import get_engine, ping
from modules.training_data import infer_poi_specs_from_run
from modules.training_data_factory import aux_layer_flags, build_feature_groups, filter_feature_groups_for_aux_flags

logger = get_logger("ingest_poi_bundles")

ID_COLS = ["h3", "geometry", "centroid", "run_id", "region"]


def _suffix_columns(columns: list[str], suffix: str) -> list[str]:
    return sorted(c for c in columns if c.endswith(suffix) and c not in ID_COLS and not c.startswith(("geometry", "centroid")))


def _feature_groups_for_run(columns: list[str]) -> dict[str, list[str]]:
    groups = build_feature_groups(pd.DataFrame(columns=columns), ID_COLS)
    for name in ("bus_delta", "walk_transit", "walk_transit_bus", "walk_transit_rail"):
        groups[name] = _suffix_columns(columns, f"_{name}")
    return groups


def _alignment_family(groups: dict[str, list[str]]) -> tuple[str | None, list[str], str | None]:
    if groups.get("walk_transit_bus"):
        return "walk_transit_bus", list(groups["walk_transit_bus"]), "_walk_transit_bus"
    if groups.get("bus_delta"):
        return "bus_delta", list(groups["bus_delta"]), "_bus_delta"
    return None, [], None


def _bus_bases(groups) -> list[str]:
    _, cols, suffix = _alignment_family(groups)
    return [c[: -len(suffix)] for c in cols] if suffix else []


def _resolve_layer_name(base: str, layers: dict) -> str | None:
    for candidate in (base, base[4:] if base.startswith("poi_") else None):
        if candidate and candidate in layers:
            return candidate
    return None


def _layer_kind(base: str) -> str:
    if base.startswith("wrk_"):
        return "workplace"
    if base.startswith("sat_"):
        return "satellite"
    return "poi"


def _poi_layers(run_dir: Path) -> dict[str, gpd.GeoDataFrame]:
    """POI points per class from the run's `pois.csv` (layer, lon, lat in WGS84)."""
    path = run_dir / "pois.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; the MAPPER run must extract POIs (POI_CLASSES)")
    frame = pd.read_csv(path)
    points = gpd.GeoDataFrame(frame[["layer"]], geometry=gpd.points_from_xy(frame["lon"], frame["lat"]), crs="EPSG:4326")
    return {str(name): group.reset_index(drop=True) for name, group in points.groupby("layer")}


def _workplace_layers(run_dir: Path, graph_path: Path) -> dict[str, gpd.GeoDataFrame]:
    """Workplace weights per category from `src_wrk_<mode>.csv`, placed at the snapped nodes.

    The CSV holds one row per (node, access penalty); weights are summed per node.
    """
    path = run_dir / f"src_wrk_{graph_path.stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; pass the graph of a mode the run computed workplaces for")
    graph = read_mapper_graph(graph_path)
    row_of = pd.Series(np.arange(len(graph.node_ids)), index=graph.node_ids.astype(np.int64))
    frame = pd.read_csv(path).groupby(["layer", "node"], as_index=False)["weight"].sum()
    rows = row_of.reindex(frame["node"].astype(np.int64)).to_numpy()
    if np.isnan(rows).any():
        raise ValueError(f"{path} references nodes that are not in {graph_path.name}")
    rows = rows.astype(np.int64)
    points = gpd.GeoDataFrame(frame[["layer", "weight"]],
                              geometry=gpd.points_from_xy(graph.lon[rows], graph.lat[rows]), crs="EPSG:4326")
    return {str(name): group.drop(columns="layer").reset_index(drop=True) for name, group in points.groupby("layer")}


def _alignment_report(groups, layers) -> tuple[list[dict], list[dict]]:
    family, _, _ = _alignment_family(groups)
    rail = set(groups.get("walk_transit_rail") or [])
    bus_access = set(groups.get("walk_transit_bus") or [])
    bus_delta = set(groups.get("bus_delta") or [])
    rows, missing = [], []
    for base in _bus_bases(groups):
        rail_col = f"{base}_walk_transit_rail"
        layer = _resolve_layer_name(base, layers)
        row = {
            "base_name": base,
            "layer_kind": _layer_kind(base),
            "alignment_family": family,
            "bus_access_column": f"{base}_walk_transit_bus" if f"{base}_walk_transit_bus" in bus_access else None,
            "bus_delta_column": f"{base}_bus_delta" if f"{base}_bus_delta" in bus_delta else None,
            "rail_baseline_column": rail_col,
            "resolved_layer_name": layer,
            "has_bus_reference": True,
            "has_rail_baseline": rail_col in rail,
            "has_bundle_layer": layer is not None,
        }
        rows.append(row)
        if not (row["has_rail_baseline"] and row["has_bundle_layer"]):
            missing.append(row)
    return rows, missing


def _insert_layers(cur, dataset_key, alignment_rows, layers) -> int:
    rows = []
    for row in alignment_rows:
        name = row["resolved_layer_name"]
        if not name:
            continue
        gdf = layers[name]
        geom_types = sorted({str(t) for t in gdf.geometry.geom_type.dropna().unique()}) if not gdf.empty else []
        rows.append((dataset_key, name, row["base_name"], row["layer_kind"], row["bus_access_column"], row["bus_delta_column"],
                     row["rail_baseline_column"], int(len(gdf)),
                     Json({"columns": [c for c in gdf.columns if c != "geometry"], "geometry_types": geom_types})))
    if rows:
        execute_values(
            cur,
            """
            INSERT INTO poi.layers (dataset_key, layer_name, base_name, layer_kind, bus_access_column,
                                    bus_delta_column, rail_baseline_column, feature_count, metadata_json)
            VALUES %s
            """,
            rows,
            page_size=1000,
        )
    return len(rows)


def _insert_features(cur, dataset_key, alignment_rows, layers, chunk_size) -> int:
    sql = "INSERT INTO poi.features (dataset_key, layer_name, geom, weight, attrs) VALUES %s"
    template = "(%s, %s, ST_GeomFromWKB(%s, 4326), %s, %s)"
    rows, inserted = [], 0
    for row in alignment_rows:
        name = row["resolved_layer_name"]
        if not name:
            continue
        gdf = layers[name]
        gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
        attr_cols = [c for c in gdf.columns if c not in {"geometry", "weight"}]
        weights = gdf["weight"].tolist() if "weight" in gdf.columns else [None] * len(gdf)
        attr_arrays = [gdf[c].tolist() for c in attr_cols]
        for i, geom in enumerate(gdf.geometry):
            if geom is None:
                continue
            attrs = {c: jsonable(arr[i]) for c, arr in zip(attr_cols, attr_arrays)}
            weight = jsonable(weights[i])
            rows.append((dataset_key, name, geom.wkb, float(weight) if weight is not None else None, Json(attrs)))
            if len(rows) >= chunk_size:
                execute_values(cur, sql, rows, template=template, page_size=chunk_size)
                inserted += len(rows)
                rows.clear()
    if rows:
        execute_values(cur, sql, rows, template=template, page_size=chunk_size)
        inserted += len(rows)
    return inserted


def ingest_poi_bundles(*, run_dir: Path, run_id: str, workplace_graph: Path | None, dataset_key: str | None,
                       chunk_size: int, dry_run: bool, allow_incomplete: bool) -> dict:
    meta, _ = read_run_meta(run_dir)
    region = str(meta.get("region") or "").strip()
    if not region:
        raise ValueError(f"Run metadata in {run_dir} has no region")
    cfg = SimpleNamespace(LOAD_SAT=False, LOAD_WRK=workplace_graph is not None)
    groups = filter_feature_groups_for_aux_flags(_feature_groups_for_run(run_column_names(run_dir)), cfg)
    load_sat, load_wrk = aux_layer_flags(cfg)
    layers = _poi_layers(run_dir)
    if load_wrk:
        layers.update(_workplace_layers(run_dir, workplace_graph))
    alignment_rows, missing = _alignment_report(groups, layers)
    dataset_key = dataset_key or f"poi_bundle::{run_id}::training_tiles"
    summary = {
        "run_id": run_id,
        "dataset_key": dataset_key,
        "region": region,
        "load_wrk": load_wrk,
        "bundle_layer_count": len(layers),
        "aligned_layer_count": sum(1 for r in alignment_rows if r["resolved_layer_name"]),
        "missing_triplets": [r["base_name"] for r in missing],
    }
    if missing and not allow_incomplete:
        raise RuntimeError(f"POI bundle alignment incomplete for bases {summary['missing_triplets'][:20]}")
    if dry_run:
        return summary

    ping()
    metadata = {
        "amenities": meta.get("amenities") or [],
        "workplaces": meta.get("workplaces") or [],
        "poi_specs": infer_poi_specs_from_run(groups, meta),
        "load_sat": load_sat,
        "load_wrk": load_wrk,
        "feature_groups": groups,
        "alignment_rows": alignment_rows,
        "missing_triplets": missing,
    }
    signature = hashlib.sha256(json.dumps({"run_id": run_id, "meta": meta, "alignment": alignment_rows},
                                          sort_keys=True, default=jsonable).encode("utf-8")).hexdigest()
    raw = get_engine().raw_connection()
    try:
        cur = raw.cursor()
        try:
            cur.execute("DELETE FROM poi.features WHERE dataset_key = %s", (dataset_key,))
            cur.execute("DELETE FROM poi.datasets WHERE dataset_key = %s", (dataset_key,))
            cur.execute(
                """
                INSERT INTO poi.datasets (dataset_key, run_id, region, cache_dir, config_module, source_signature, metadata_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (dataset_key, run_id, region, run_dir.name, "scripts/postgis/ingest_poi_bundles.py", signature, Json(jsonable(metadata))),
            )
            summary["layers_inserted"] = _insert_layers(cur, dataset_key, alignment_rows, layers)
            raw.commit()
            summary["features_inserted"] = _insert_features(cur, dataset_key, alignment_rows, layers, chunk_size)
            raw.commit()
            cur.close()
        except Exception:
            raw.rollback()
            raise
    finally:
        raw.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest aligned POI and workplace layers into PostGIS")
    parser.add_argument("--run-dir", required=True, help="MAPPER run output directory (cells.parquet, run_manifest.json, pois.csv, src_wrk_*.csv)")
    parser.add_argument("--workplace-graph", help="Graph file <mode>.bin whose src_wrk_<mode>.csv supplies the workplace layers")
    parser.add_argument("--no-workplaces", action="store_true", help="Skip workplace layers (tile factory default: included)")
    parser.add_argument("--run-id", help="Access run id registered by ingest_access_run.py (default: the run directory name)")
    parser.add_argument("--dataset-key", help="Key to register (default: poi_bundle::<run-id>::training_tiles); use it as POI_DATASET_KEY")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--allow-incomplete", action="store_true", help="Ingest even if some access columns have no matching layer")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and align layers without writing")
    args = parser.parse_args()
    if not args.no_workplaces and not args.workplace_graph:
        parser.error("pass --workplace-graph, or --no-workplaces")

    run_dir = Path(args.run_dir).expanduser().resolve()
    summary = ingest_poi_bundles(
        run_dir=run_dir,
        run_id=args.run_id or run_dir.name,
        workplace_graph=None if args.no_workplaces else Path(args.workplace_graph).expanduser(),
        dataset_key=args.dataset_key,
        chunk_size=args.chunk_size,
        dry_run=args.dry_run,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
