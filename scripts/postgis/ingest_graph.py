#!/usr/bin/env python3
"""Ingest a MAPPER graph into graph.datasets/nodes/edges.

The tile factory needs two graphs: the road graph (ROAD_GRAPH_KEY) and the bus service graph
(BUS_GRAPH_KEY). The road graph is the `drive.bin` a MAPPER run writes (--graph). The bus graph
holds only GTFS service legs and their stops; it is built from the bus feed with MAPPER's
`build_transit_cpp` (--gtfs), without walk transfers, and the synthetic multi-route stop nodes
are dropped. Node and edge ids are stored with typed `_source_*` records so the tile factory
restores them exactly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from psycopg2.extras import Json, execute_values
from pyproj import Geod
from shapely.geometry import LineString, Point

from _common import REPO_ROOT, file_signature, get_logger, id_payload
from _mapper_graph import MapperGraph, read_mapper_graph

from modules.postgis_bridge import get_engine, ping

logger = get_logger("ingest_graph")

GEOD = Geod(ellps="WGS84")
# build_transit_cpp adds a walk transfer between stops of different routes within this radius
# (metres). It has to be positive, so a millimetre stands in for "no transfers".
NO_TRANSFER_RADIUS_M = 0.001


def build_bus_graph(engine: Path, gtfs: Path, ring: Path, crs: str, window: tuple[str, str], out: Path) -> Path:
    binary = engine / "source" / "cpp" / ".build" / "bin" / "build_transit_cpp"
    if not binary.exists():
        raise FileNotFoundError(f"{binary} not found; build MAPPER first (source/infra/bootstrap_cpp_worker.sh)")
    cmd = [str(binary), "--zip", str(gtfs), "--ring", str(ring), "--crs", crs,
           "--window", f"{window[0]}:00", f"{window[1]}:00",
           "--walk-transfer", str(NO_TRANSFER_RADIUS_M), "--out", str(out)]
    subprocess.run(cmd, check=True)
    return out


def drop_stop_nodes(graph: MapperGraph) -> np.ndarray:
    """Edge mask without the board/alight edges of synthetic stop nodes (node_kind 1)."""
    stop = graph.node_kind == 1
    return ~(stop[graph.edge_u] | stop[graph.edge_v])


def _edge_rows(graph: MapperGraph, keep: np.ndarray, mode: str):
    for e in np.flatnonzero(keep):
        u, v = int(graph.edge_u[e]), int(graph.edge_v[e])
        shape = graph.shapes[e]
        if graph.direction[e] == 2:
            u, v = v, u
            shape = shape[::-1] if shape is not None else None
        if shape is not None and len(shape) >= 2:
            lats, lons = shape[:, 0].astype(float), shape[:, 1].astype(float)
        else:
            lats = np.array([graph.lat[u], graph.lat[v]])
            lons = np.array([graph.lon[u], graph.lon[v]])
        if lats[0] == lats[-1] and lons[0] == lons[-1] and len(lats) == 2:
            continue
        seconds = float(graph.weight_ms[e]) / 1000.0
        uid, vid = int(graph.node_ids[u]), int(graph.node_ids[v])
        payload = {
            "travel_time": seconds,
            "direction": int(graph.direction[e]),
            "_source_u_id": id_payload(uid),
            "_source_v_id": id_payload(vid),
            "_source_edge_key": id_payload(int(e)),
        }
        yield (uid, vid, int(e), LineString(zip(lons, lats)).wkb,
               float(GEOD.line_length(lons, lats)), mode, seconds, payload)


def _insert(cur, sql, template, rows, chunk_size) -> int:
    batch, inserted = [], 0
    for row in rows:
        batch.append(row)
        if len(batch) >= chunk_size:
            execute_values(cur, sql, batch, template=template, page_size=chunk_size)
            inserted += len(batch)
            batch.clear()
    if batch:
        execute_values(cur, sql, batch, template=template, page_size=chunk_size)
        inserted += len(batch)
    return inserted


def ingest_graph(*, graph_path: Path, graph_key: str, region: str, mode: str, edge_mode: str,
                 transit: bool, build_meta: dict, chunk_size: int, dry_run: bool) -> dict:
    graph = read_mapper_graph(graph_path)
    keep = drop_stop_nodes(graph) if transit else np.ones(len(graph.edge_u), dtype=bool)
    used = np.zeros(len(graph.node_ids), dtype=bool)
    used[graph.edge_u[keep]] = True
    used[graph.edge_v[keep]] = True
    summary = {
        "graph_key": graph_key,
        "graph_file": graph_path.name,
        "region": region,
        "mode": mode,
        "format_version": graph.version,
        "node_count": int(used.sum()),
        "edge_count": int(keep.sum()),
    }
    if dry_run:
        return summary

    ping()
    raw = get_engine().raw_connection()
    try:
        cur = raw.cursor()
        try:
            cur.execute("DELETE FROM graph.datasets WHERE graph_key = %s", (graph_key,))
            cur.execute(
                """
                INSERT INTO graph.datasets (graph_key, region, mode, cache_path, crs, build_meta, source_signature)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (graph_key, region, mode, graph_path.name, "EPSG:4326", Json(build_meta), file_signature(graph_path)),
            )
            node_rows = (
                (graph_key, str(int(graph.node_ids[i])), Point(graph.lon[i], graph.lat[i]).wkb,
                 Json({"_source_node_id": id_payload(int(graph.node_ids[i]))}))
                for i in np.flatnonzero(used)
            )
            summary["nodes_inserted"] = _insert(
                cur, "INSERT INTO graph.nodes (graph_key, node_id, geom, attrs) VALUES %s",
                "(%s, %s, ST_GeomFromWKB(%s, 4326), %s)", node_rows, chunk_size)
            edge_rows = (
                (graph_key, str(u), str(v), str(k), wkb, length, m, cost, Json(payload))
                for u, v, k, wkb, length, m, cost, payload in _edge_rows(graph, keep, edge_mode)
            )
            summary["edges_inserted"] = _insert(
                cur,
                "INSERT INTO graph.edges (graph_key, u_id, v_id, edge_key, geom, length_m, mode, cost, attrs) VALUES %s",
                "(%s, %s, %s, %s, ST_GeomFromWKB(%s, 4326), %s, %s, %s, %s)", edge_rows, chunk_size)
            raw.commit()
            cur.close()
        except Exception:
            raw.rollback()
            raise
    finally:
        raw.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a MAPPER graph into PostGIS")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--graph", help="MAPPER graph file, e.g. the run's drive.bin (road graph)")
    source.add_argument("--gtfs", help="Bus GTFS feed; builds the bus service graph with MAPPER")
    parser.add_argument("--region", required=True, help="MAPPER region slug, e.g. europe/germany")
    parser.add_argument("--mode", help="Mode label (default: drive for --graph, transit_bus for --gtfs)")
    parser.add_argument("--graph-key", help="Key to register (default: <region>|<mode>); use it as ROAD_GRAPH_KEY/BUS_GRAPH_KEY")
    parser.add_argument("--ring", help="Clip ring of the MAPPER run (lat,lon per line); required with --gtfs")
    parser.add_argument("--crs", default="EPSG:32632", help="Metric CRS of the MAPPER run (with --gtfs)")
    parser.add_argument("--time-window-start", default="07:00")
    parser.add_argument("--time-window-end", default="09:00")
    parser.add_argument("--engine", default=str(REPO_ROOT.parent / "MAPPER"), help="MAPPER checkout (with --gtfs)")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true", help="Read and count the graph without writing")
    args = parser.parse_args()
    if args.gtfs and not args.ring:
        parser.error("--gtfs needs --ring")

    transit = bool(args.gtfs)
    mode = args.mode or ("transit_bus" if transit else "drive")
    graph_key = args.graph_key or f"{args.region}|{mode}"
    window = (args.time_window_start, args.time_window_end)
    with tempfile.TemporaryDirectory() as tmp:
        if transit:
            graph_path = build_bus_graph(Path(args.engine).expanduser().resolve(), Path(args.gtfs).expanduser(),
                                         Path(args.ring).expanduser(), args.crs, window, Path(tmp) / f"{mode}.bin")
            build_meta = {"gtfs": Path(args.gtfs).name, "time_window": {"start": window[0], "end": window[1]}}
        else:
            graph_path = Path(args.graph).expanduser()
            build_meta = {}
        summary = ingest_graph(
            graph_path=graph_path,
            graph_key=graph_key,
            region=args.region,
            mode=mode,
            edge_mode="transit" if transit else mode,
            transit=transit,
            build_meta=build_meta,
            chunk_size=args.chunk_size,
            dry_run=args.dry_run,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
