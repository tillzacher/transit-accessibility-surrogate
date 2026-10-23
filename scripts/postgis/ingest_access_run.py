#!/usr/bin/env python3
"""Ingest one MAPPER accessibility run (cells.parquet + run_manifest.json) into PostGIS.

Writes access.runs (one row) and access.cells (one row per H3 cell). Cell values are stored
as float32-rounded JSON, one key per mode x opportunity column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
from psycopg2.extras import Json, execute_values

from _common import file_signature, get_logger, jsonable, read_run_meta, run_column_names

from modules.h3_functions import id_to_polygon
from modules.postgis_bridge import get_engine, ping

logger = get_logger("ingest_access_run")

BASE_COLS = {"h3", "run_id", "region", "geometry", "centroid"}
MODE_SUFFIXES = {
    "_walk_transit": "walk_transit",
    "_walk_transit_bus": "walk_transit_bus",
    "_walk_transit_rail": "walk_transit_rail",
    "_bus_delta": "bus_delta",
}


def _source_signature(run_dir: Path) -> str:
    paths = (run_dir / "run_manifest.json", run_dir / "cells.parquet")
    return "|".join(file_signature(p) for p in paths if p.exists())


def _feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups = {name: [] for name in MODE_SUFFIXES.values()}
    groups["other"] = []
    for col in columns:
        for suffix, group in MODE_SUFFIXES.items():
            if col.endswith(suffix):
                groups[group].append(col)
                break
        else:
            groups["other"].append(col)
    return groups


def _utm_crs(lon: float, lat: float) -> str:
    if lat >= 84:
        return "EPSG:32661"
    if lat <= -80:
        return "EPSG:32761"
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def _batch_to_gdf(df) -> gpd.GeoDataFrame:
    if "h3" not in df.columns:
        raise KeyError("cells.parquet must contain an 'h3' column")
    float_cols = [c for c in df.columns if df[c].dtype.kind == "f"]
    if float_cols:
        df[float_cols] = df[float_cols].astype(np.float32)
    gdf = gpd.GeoDataFrame(df, geometry=[id_to_polygon(c) for c in df["h3"]], crs="EPSG:4326")
    if gdf.empty:
        gdf["centroid"] = gpd.GeoSeries([], index=gdf.index, crs=gdf.crs)
        return gdf
    rep = gdf.geometry.iloc[0].representative_point()
    gdf["centroid"] = gdf.to_crs(_utm_crs(rep.x, rep.y)).centroid.to_crs(gdf.crs)
    return gdf


def _upsert_run(cur, run_id, meta, manifest_path, signature, bounds, columns) -> None:
    value_cols = [c for c in columns if c not in BASE_COLS]
    minx, miny, maxx, maxy = bounds
    cur.execute(
        """
        INSERT INTO access.runs (
            run_id, manifest_path, region, grid_crs, h3_res, created_at,
            config_json, tensor_schema_json, feature_groups_json, bounds, source_signature
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s,
            ST_Multi(ST_MakeEnvelope(%s, %s, %s, %s, 4326)), %s
        )
        ON CONFLICT (run_id) DO UPDATE SET
            manifest_path = EXCLUDED.manifest_path,
            region = EXCLUDED.region,
            grid_crs = EXCLUDED.grid_crs,
            h3_res = EXCLUDED.h3_res,
            created_at = EXCLUDED.created_at,
            config_json = EXCLUDED.config_json,
            tensor_schema_json = EXCLUDED.tensor_schema_json,
            feature_groups_json = EXCLUDED.feature_groups_json,
            bounds = EXCLUDED.bounds,
            source_signature = EXCLUDED.source_signature
        """,
        (
            run_id,
            manifest_path.name,
            meta.get("region"),
            "EPSG:4326",
            meta.get("h3_res"),
            meta.get("timestamp"),
            Json(jsonable(meta)),
            Json({"version": 1, "columns": [c for c in columns if c not in {"geometry", "centroid"}]}),
            Json(_feature_groups(value_cols)),
            float(minx), float(miny), float(maxx), float(maxy),
            signature,
        ),
    )


def _insert_cells(cur, gdf: gpd.GeoDataFrame, run_id: str, region, chunk_size: int) -> int:
    value_cols = [c for c in gdf.columns if c not in BASE_COLS]
    rows = []
    inserted = 0
    sql = "INSERT INTO access.cells (run_id, cell_id, geom, centroid, values_json, row_meta) VALUES %s"
    template = "(%s, %s, ST_GeomFromWKB(%s, 4326), ST_GeomFromWKB(%s, 4326), %s, %s)"
    n = len(gdf)
    regions = gdf["region"].tolist() if "region" in gdf.columns else [region] * n
    source_runs = gdf["run_id"].tolist() if "run_id" in gdf.columns else [None] * n
    value_arrays = [gdf[c].tolist() for c in value_cols]
    for i, (h3_id, geom, centroid) in enumerate(zip(gdf["h3"].tolist(), gdf.geometry, gdf["centroid"])):
        values = {c: jsonable(arr[i]) for c, arr in zip(value_cols, value_arrays)}
        row_meta = {"h3": h3_id, "region": regions[i], "source_run_id": source_runs[i] or run_id}
        rows.append((run_id, str(h3_id), geom.wkb, centroid.wkb if centroid is not None else None, Json(values), Json(row_meta)))
        if len(rows) >= chunk_size:
            execute_values(cur, sql, rows, template=template, page_size=chunk_size)
            inserted += len(rows)
            rows.clear()
    if rows:
        execute_values(cur, sql, rows, template=template, page_size=chunk_size)
        inserted += len(rows)
    return inserted


def ingest_access_run(run_dir: Path, run_id: str, chunk_size: int = 2000, reset_run: bool = True) -> int:
    meta, manifest_path = read_run_meta(run_dir)
    columns = run_column_names(run_dir)
    signature = _source_signature(run_dir)
    parquet = pq.ParquetFile(run_dir / "cells.parquet")
    ping()
    raw = get_engine().raw_connection()
    total = 0
    bounds = None
    try:
        cur = raw.cursor()
        try:
            if reset_run:
                cur.execute("DELETE FROM access.cells WHERE run_id = %s", (run_id,))
                cur.execute("DELETE FROM access.runs WHERE run_id = %s", (run_id,))
                raw.commit()
            for rg in range(parquet.metadata.num_row_groups):
                gdf = _batch_to_gdf(parquet.read_row_group(rg).to_pandas())
                if gdf.empty:
                    continue
                b = tuple(float(v) for v in gdf.total_bounds)
                bounds = b if bounds is None else (min(bounds[0], b[0]), min(bounds[1], b[1]), max(bounds[2], b[2]), max(bounds[3], b[3]))
                _upsert_run(cur, run_id, meta, manifest_path, signature, bounds, columns)
                total += _insert_cells(cur, gdf, run_id, meta.get("region"), chunk_size)
                raw.commit()
                logger.info("Row group %d/%d committed | cells=%d", rg + 1, parquet.metadata.num_row_groups, total)
            cur.close()
        except Exception:
            raw.rollback()
            raise
    finally:
        raw.close()
    if total == 0:
        raise RuntimeError(f"No cells ingested from {run_dir / 'cells.parquet'}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest one MAPPER accessibility run into PostGIS")
    parser.add_argument("--run-dir", required=True, help="MAPPER run output directory (cells.parquet, run_manifest.json)")
    parser.add_argument("--run-id", help="Identifier to register (default: the run directory name); use it as REFERENCE_RUN_ID")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--no-reset-run", dest="reset_run", action="store_false", help="Keep existing rows for this run id")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_id = args.run_id or run_dir.name
    inserted = ingest_access_run(run_dir, run_id, chunk_size=args.chunk_size, reset_run=args.reset_run)
    print(json.dumps({"run_id": run_id, "rows_inserted": inserted}, indent=2))


if __name__ == "__main__":
    main()
