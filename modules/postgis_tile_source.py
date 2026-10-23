"""Tile-local PostGIS query helpers for the spatial serving store."""

from __future__ import annotations
from typing import Any
import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry.base import BaseGeometry
from sqlalchemy import text
from modules.postgis_bridge import get_engine


def _metric_buffer_wgs84(geom: BaseGeometry, crs: str, buffer_m: float) -> BaseGeometry:
    series = gpd.GeoSeries([geom], crs=crs)
    if buffer_m:
        series = series.to_crs("EPSG:3857").buffer(float(buffer_m)).to_crs("EPSG:4326")
    elif str(crs).upper() != "EPSG:4326":
        series = series.to_crs("EPSG:4326")
    return series.iloc[0]


def _restore_id(value: Any, attrs: dict | None, key: str) -> Any:
    payload = (attrs or {}).get(key)
    if isinstance(payload, dict):
        typ = payload.get("type")
        raw = payload.get("value")
        if typ == "int":
            try:
                return int(raw)
            except Exception:
                return raw
        if typ == "float":
            try:
                return float(raw)
            except Exception:
                return raw
        if typ == "str":
            return str(raw)
    return value


class SpatialTileSource:
    """Narrow query boundary for PostGIS-backed tile construction."""

    def __init__(self, engine=None):
        self.engine = engine or get_engine()

    def get_access_run_meta(self, run_id: str) -> dict:
        sql = text(
            "\n            SELECT run_id, manifest_path, region, grid_crs, h3_res, created_at,\n                   config_json, tensor_schema_json, feature_groups_json,\n                   ST_AsBinary(bounds) AS bounds_wkb, source_signature\n            FROM access.runs\n            WHERE run_id = :run_id\n            "
        )
        with self.engine.begin() as conn:
            row = conn.execute(sql, {"run_id": run_id}).mappings().first()
        if row is None:
            raise RuntimeError(f"Access run not found in PostGIS: {run_id}")
        out = dict(row)
        bounds_wkb = out.pop("bounds_wkb", None)
        if bounds_wkb is not None:
            from shapely import wkb

            out["bounds"] = wkb.loads(bytes(bounds_wkb))
        return out

    def get_access_cells(
        self,
        run_id: str,
        tile_geom: BaseGeometry,
        *,
        tile_crs: str = "EPSG:4326",
        buffer_m: float = 0.0,
    ) -> gpd.GeoDataFrame:
        query_geom = _metric_buffer_wgs84(tile_geom, tile_crs, buffer_m)
        sql = text(
            "\n            SELECT cell_id, values_json, row_meta, geom, centroid\n            FROM access.cells\n            WHERE run_id = :run_id\n              AND ST_Intersects(geom, ST_GeomFromWKB(:geom_wkb, 4326))\n            "
        )
        gdf = gpd.read_postgis(
            sql,
            self.engine,
            geom_col="geom",
            params={"run_id": run_id, "geom_wkb": query_geom.wkb},
            crs="EPSG:4326",
        )
        if gdf.empty:
            return gpd.GeoDataFrame(
                columns=["h3", "geometry", "centroid"],
                geometry="geometry",
                crs="EPSG:4326",
            )
        values = pd.json_normalize(gdf["values_json"]).set_index(gdf.index)
        meta = pd.json_normalize(gdf["row_meta"]).set_index(gdf.index)
        out = pd.concat([meta, values], axis=1)
        out["h3"] = out.get("h3", gdf["cell_id"])
        out["geometry"] = gdf["geom"]
        if "centroid" in gdf:
            out["centroid"] = gpd.GeoSeries.from_wkb(gdf["centroid"], crs="EPSG:4326")
        else:
            out["centroid"] = gpd.GeoSeries(
                [geom.centroid for geom in out["geometry"]], crs="EPSG:4326"
            )
        return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")

    def coverage_fraction(
        self, run_id: str, tile_geom: BaseGeometry, *, tile_crs: str = "EPSG:4326"
    ) -> float:
        """Return the fraction [0, 1] of tile_geom's area covered by access cells."""
        query_geom = _metric_buffer_wgs84(tile_geom, tile_crs, 0.0)
        sql = text(
            "\n            SELECT\n                COALESCE(\n                    ST_Area(\n                        ST_Union(ST_Intersection(geom, ST_GeomFromWKB(:geom_wkb, 4326)))\n                        ::geography\n                    ), 0\n                ) AS covered_m2,\n                ST_Area(ST_GeomFromWKB(:geom_wkb, 4326)::geography) AS tile_m2\n            FROM access.cells\n            WHERE run_id = :run_id\n              AND ST_Intersects(geom, ST_GeomFromWKB(:geom_wkb, 4326))\n            "
        )
        params = {"run_id": run_id, "geom_wkb": query_geom.wkb}
        with self.engine.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        if not row or not row[1]:
            return 0.0
        return float(row[0]) / float(row[1])

    def get_access_cell_geometries(
        self,
        run_id: str,
        tile_geom: BaseGeometry,
        *,
        tile_crs: str = "EPSG:4326",
        buffer_m: float = 0.0,
    ) -> gpd.GeoDataFrame:
        """Return only access cell ids/geometries for lightweight tile planning."""
        query_geom = _metric_buffer_wgs84(tile_geom, tile_crs, buffer_m)
        sql = text(
            "\n            SELECT cell_id AS h3, geom\n            FROM access.cells\n            WHERE run_id = :run_id\n              AND ST_Intersects(geom, ST_GeomFromWKB(:geom_wkb, 4326))\n            "
        )
        gdf = gpd.read_postgis(
            sql,
            self.engine,
            geom_col="geom",
            params={"run_id": run_id, "geom_wkb": query_geom.wkb},
            crs="EPSG:4326",
        )
        return gdf.rename(columns={"geom": "geometry"}).set_geometry("geometry")

    def get_graph_slice(
        self,
        graph_key: str,
        tile_geom: BaseGeometry,
        *,
        tile_crs: str = "EPSG:4326",
        buffer_m: float = 0.0,
        output_crs: str | None = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Return edge-windowed graph GDFs with endpoint-node closure."""
        query_geom = _metric_buffer_wgs84(tile_geom, tile_crs, buffer_m)
        edges_sql = text(
            "\n            SELECT edge_id, graph_key, u_id, v_id, edge_key, length_m, mode, cost, reverse_cost,\n                   attrs, geom\n            FROM graph.edges\n            WHERE graph_key = :graph_key\n              AND ST_Intersects(geom, ST_GeomFromWKB(:geom_wkb, 4326))\n            "
        )
        edges = gpd.read_postgis(
            edges_sql,
            self.engine,
            geom_col="geom",
            params={"graph_key": graph_key, "geom_wkb": query_geom.wkb},
            crs="EPSG:4326",
        )
        node_ids: set[str] = set()
        if not edges.empty:
            node_ids.update((str(v) for v in edges["u_id"].dropna().tolist()))
            node_ids.update((str(v) for v in edges["v_id"].dropna().tolist()))
        if node_ids:
            nodes_sql = text(
                "\n                SELECT graph_key, node_id, attrs, geom\n                FROM graph.nodes\n                WHERE graph_key = :graph_key\n                  AND node_id = ANY(:node_ids)\n                "
            )
            nodes = gpd.read_postgis(
                nodes_sql,
                self.engine,
                geom_col="geom",
                params={"graph_key": graph_key, "node_ids": sorted(node_ids)},
                crs="EPSG:4326",
            )
        else:
            nodes = gpd.GeoDataFrame(
                columns=["graph_key", "node_id", "attrs", "geom"],
                geometry="geom",
                crs="EPSG:4326",
            )
        nodes = self._graph_nodes_to_contract(nodes)
        edges = self._graph_edges_to_contract(edges)
        if output_crs:
            nodes = nodes.to_crs(output_crs)
            edges = edges.to_crs(output_crs)
        return (nodes, edges)

    def build_networkx_subgraph(
        self,
        graph_key: str,
        tile_geom: BaseGeometry,
        *,
        tile_crs: str = "EPSG:4326",
        buffer_m: float = 0.0,
    ) -> nx.MultiDiGraph:
        nodes, edges = self.get_graph_slice(
            graph_key, tile_geom, tile_crs=tile_crs, buffer_m=buffer_m
        )
        graph = nx.MultiDiGraph()
        graph.graph["crs"] = "EPSG:4326"
        for row in nodes.itertuples(index=False):
            attrs = dict(getattr(row, "attrs", None) or {})
            node_id = getattr(row, "id")
            if node_id is None:
                raise RuntimeError(
                    f"Graph slice for {graph_key} contains a node with no restorable id"
                )
            attrs.setdefault("x", float(row.geometry.x))
            attrs.setdefault("y", float(row.geometry.y))
            graph.add_node(node_id, **attrs)
        for row in edges.itertuples(index=False):
            attrs = dict(getattr(row, "attrs", None) or {})
            attrs["geometry"] = row.geometry
            if getattr(row, "length_m", None) is not None and "length" not in attrs:
                attrs["length"] = float(row.length_m)
            if getattr(row, "mode", None) is not None and "mode" not in attrs:
                attrs["mode"] = row.mode
            u_id = getattr(row, "u")
            v_id = getattr(row, "v")
            if u_id is None or v_id is None:
                raise RuntimeError(
                    f"Graph slice for {graph_key} contains an edge with missing endpoint id"
                )
            edge_key = _restore_id(
                getattr(row, "edge_key", None), attrs, "_source_edge_key"
            )
            graph.add_edge(u_id, v_id, key=edge_key, **attrs)
        return graph

    def get_poi_layers(self, dataset_key: str) -> pd.DataFrame:
        sql = text(
            "\n            SELECT layer_name, base_name, layer_kind, bus_access_column,\n                   bus_delta_column, rail_baseline_column, feature_count,\n                   metadata_json\n            FROM poi.layers\n            WHERE dataset_key = :dataset_key\n            ORDER BY layer_name\n            "
        )
        return pd.read_sql(sql, self.engine, params={"dataset_key": dataset_key})

    def get_poi_dataset_metadata(self, dataset_key: str) -> dict:
        sql = text(
            "SELECT metadata_json FROM poi.datasets WHERE dataset_key = :dataset_key"
        )
        with self.engine.begin() as conn:
            value = conn.execute(sql, {"dataset_key": dataset_key}).scalar()
        if value is None:
            raise RuntimeError(
                f"POI dataset not found in PostGIS: {dataset_key}. Check --poi-dataset-key and run scripts/postgis/ingest_poi_bundles.py first."
            )
        return dict(value)

    def get_poi_features(
        self,
        dataset_key: str,
        tile_geom: BaseGeometry,
        *,
        tile_crs: str = "EPSG:4326",
        buffer_m: float = 0.0,
        layer_names: list[str] | None = None,
        output_crs: str | None = None,
    ) -> gpd.GeoDataFrame:
        query_geom = _metric_buffer_wgs84(tile_geom, tile_crs, buffer_m)
        layer_filter = "AND layer_name = ANY(:layer_names)" if layer_names else ""
        sql = text(
            f"\n            SELECT dataset_key, layer_name, weight, attrs, geom\n            FROM poi.features\n            WHERE dataset_key = :dataset_key\n              {layer_filter}\n              AND ST_Intersects(geom, ST_GeomFromWKB(:geom_wkb, 4326))\n            "
        )
        params = {"dataset_key": dataset_key, "geom_wkb": query_geom.wkb}
        if layer_names:
            params["layer_names"] = list(layer_names)
        gdf = gpd.read_postgis(
            sql, self.engine, geom_col="geom", params=params, crs="EPSG:4326"
        )
        gdf = gdf.rename(columns={"geom": "geometry"}).set_geometry("geometry")
        if output_crs and (not gdf.empty):
            gdf = gdf.to_crs(output_crs)
        return gdf

    @staticmethod
    def _graph_nodes_to_contract(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if nodes.empty:
            return gpd.GeoDataFrame(
                columns=["id", "attrs", "geometry"], geometry="geometry", crs=nodes.crs
            )
        out = nodes.rename(columns={"geom": "geometry"}).set_geometry("geometry").copy()
        out["id"] = [
            _restore_id(row.node_id, row.attrs, "_source_node_id")
            for row in out.itertuples(index=False)
        ]
        return out[["id", "attrs", "geometry"]]

    @staticmethod
    def _graph_edges_to_contract(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if edges.empty:
            return gpd.GeoDataFrame(
                columns=["u", "v", "edge_key", "length_m", "mode", "attrs", "geometry"],
                geometry="geometry",
                crs=edges.crs,
            )
        out = edges.rename(columns={"geom": "geometry"}).set_geometry("geometry").copy()
        out["u"] = [
            _restore_id(row.u_id, row.attrs, "_source_u_id")
            for row in out.itertuples(index=False)
        ]
        out["v"] = [
            _restore_id(row.v_id, row.attrs, "_source_v_id")
            for row in out.itertuples(index=False)
        ]
        return out[["u", "v", "edge_key", "length_m", "mode", "attrs", "geometry"]]


__all__ = ["SpatialTileSource"]
