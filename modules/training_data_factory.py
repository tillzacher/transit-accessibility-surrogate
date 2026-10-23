"""Shared tile-factory utilities used by the PostGIS runner."""

from __future__ import annotations
import warnings
import importlib
import geopandas as gpd
import numpy as np
import torch
from modules.training_data import (
    area_weighted_rasterize,
    build_grid_spec,
    build_poi_tensor,
    center_sample_rasterize,
    clip_cells,
    paired_features_by_suffix,
    tensor_metadata,
)
from modules.tile_graph_spec import (
    build_tile_graph_spec,
    gtfs_to_activations,
    render_gtfs_paths_to_grid,
)
from modules.renderer import Renderer


def load_config(module_path: str):
    if module_path.endswith(".py"):
        module_path = module_path[:-3].replace("/", ".").replace("\\", ".")
    if "." not in module_path:
        module_path = f"configs.{module_path}"
    return importlib.import_module(module_path)


def build_tensor_schema(feature_groups, poi_layer_order, bus_channel_names):
    """Persist the exact tensor channel ordering used for each training run."""
    feature_groups = feature_groups or {}
    return {
        "version": 1,
        "bus": list(bus_channel_names or []),
        "rail": list(feature_groups.get("walk_transit_rail") or []),
        "poi": list(poi_layer_order or []),
        "target": list(feature_groups.get("bus_delta") or []),
        "feature_groups": {
            str(name): list(cols or []) for name, cols in feature_groups.items()
        },
    }


def build_feature_groups(cells, id_cols):
    bus_cols, rail_cols = paired_features_by_suffix(
        cells, "_bus_delta", "_walk_transit_rail", id_cols
    )
    feature_cols = [
        c
        for c in cells.columns
        if c not in id_cols
        and (not c.startswith("geometry"))
        and (not c.startswith("centroid"))
    ]
    groups = {
        "bus_delta": bus_cols,
        "walk_transit_rail": rail_cols,
        "all": feature_cols,
    }
    return groups


def aux_layer_flags(cfg) -> tuple[bool, bool]:
    """Return (load_satellite, load_workplace), preferring split flags when present."""
    legacy = bool(getattr(cfg, "LOAD_SAT_WRK", True))
    load_sat = bool(getattr(cfg, "LOAD_SAT")) if hasattr(cfg, "LOAD_SAT") else legacy
    load_wrk = bool(getattr(cfg, "LOAD_WRK")) if hasattr(cfg, "LOAD_WRK") else legacy
    return (load_sat, load_wrk)


def filter_feature_groups_for_aux_flags(feature_groups, cfg):
    """Drop disabled satellite/workplace bases from all aligned access groups."""
    load_sat, load_wrk = aux_layer_flags(cfg)
    disabled_prefixes = []
    if not load_sat:
        disabled_prefixes.append("sat_")
    if not load_wrk:
        disabled_prefixes.append("wrk_")
    if not disabled_prefixes:
        return feature_groups
    suffixes = (
        "_bus_delta",
        "_walk_transit_rail",
        "_walk_transit_bus",
        "_walk_transit",
    )

    def keep(col):
        base = str(col)
        for suffix in suffixes:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return not any((base.startswith(prefix) for prefix in disabled_prefixes))

    return {
        name: [col for col in cols if keep(col)]
        for name, cols in (feature_groups or {}).items()
    }


def rasterize_access(
    cells_use, cols, grid, method, h3_res, fill_value, workers, chunk_rows
):
    if method == "center":
        return center_sample_rasterize(
            cells_use,
            cols,
            grid,
            h3_res=h3_res,
            fill_value=fill_value,
            n_workers=workers,
            chunk_rows=chunk_rows,
        )
    return area_weighted_rasterize(
        cells_use,
        cols,
        grid,
        fill_value=fill_value,
        n_workers=workers,
        chunk_rows=chunk_rows,
    )


def process_tile(
    tile_idx,
    tile_proj,
    cfg_payload,
    cells,
    meta,
    feature_groups,
    poi_layer_order,
    poi_layers,
    bus_nodes_proj,
    bus_edges_proj,
    bus_channel_names,
    road_nodes_proj,
    road_edges_proj,
    road_graph,
    *,
    raster_workers,
    chunk_rows,
):
    fill_value = 0.0
    tile_wgs = (
        gpd.GeoSeries([tile_proj], crs=cfg_payload["GRID_CRS"])
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    cells_use = clip_cells(cells, tile_wgs)
    grid = build_grid_spec(
        cells_use,
        cfg_payload["PIXEL_SIZE_M"],
        crs_out=cfg_payload["GRID_CRS"],
        clip_poly=tile_wgs,
        width=cfg_payload["GRID_WIDTH"],
        height=cfg_payload["GRID_HEIGHT"],
    )
    grid_meta = dict(grid)
    grid_meta.pop("clip_poly", None)
    tile_record = {
        "tile_idx": tile_idx,
        "grid": grid_meta,
        "meta": meta,
        "region": meta.get("region"),
        "tensor_schema": build_tensor_schema(
            feature_groups, poi_layer_order, bus_channel_names
        ),
    }
    arrays = {}
    access_min = cfg_payload.get("ACCESS_MIN_VALUE", 0.0)
    access_max = cfg_payload.get("ACCESS_MAX_VALUE", None)
    for group_name, cols in feature_groups.items():
        if not cols:
            continue
        tensor, mask = rasterize_access(
            cells_use,
            cols,
            grid,
            cfg_payload["RASTER_METHOD"],
            meta.get("h3_res", cfg_payload["H3_RES"]),
            fill_value,
            raster_workers,
            chunk_rows=chunk_rows,
        )
        if group_name == "bus_delta":
            tensor = np.maximum(tensor, 0)
        if access_max is not None:
            tensor = np.clip(tensor, access_min, access_max)
        elif access_min not in (None, 0):
            tensor = np.maximum(tensor, access_min)
        arrays[f"access_{group_name}"] = tensor
        arrays[f"mask_{group_name}"] = mask.astype(np.uint8)
        tile_record[f"meta_{group_name}"] = tensor_metadata(
            grid_meta,
            cols,
            meta,
            method=cfg_payload["RASTER_METHOD"],
            fill_value=fill_value,
            band_dtype=tensor.dtype,
            tile_idx=tile_idx,
        )
    if cfg_payload["BUILD_POI_TENSORS"] and poi_layer_order:
        tensor_poi, band_masks, mask_union = build_poi_tensor(
            poi_layers,
            poi_layer_order,
            grid,
            smooth=cfg_payload["POI_SMOOTH"],
            sigma_px=cfg_payload["POI_SMOOTH_SIGMA_PX"],
            boost=cfg_payload["POI_BOOST"],
            min_value=cfg_payload["POI_MIN_VALUE"],
            clip_min=cfg_payload["POI_CLIP_MIN_VALUE"],
            clip_max=cfg_payload["POI_CLIP_MAX_VALUE"],
            kernel_peak=cfg_payload["POI_KERNEL_PEAK_VALUE"],
        )
        arrays["poi_tensor"] = tensor_poi
        arrays["poi_mask"] = mask_union.astype(np.uint8)
        tile_record["meta_poi"] = {
            "grid": grid_meta,
            "layers": poi_layer_order,
            "region": meta.get("region"),
        }
    if cfg_payload["BUS_TENSOR_ENABLED"] and road_nodes_proj is not None:
        tile_buf_m = cfg_payload["BUS_TILE_BUFFER_M"]
        tile_buf = (
            gpd.GeoSeries([tile_proj], crs=cfg_payload["GRID_CRS"])
            .buffer(tile_buf_m)
            .iloc[0]
        )
        if not road_nodes_proj.empty:
            hit_idx = road_nodes_proj.sindex.query(tile_buf)
            road_nodes_tile = road_nodes_proj.iloc[hit_idx].copy()
        else:
            road_nodes_tile = road_nodes_proj.copy()
        if road_edges_proj is not None and (not road_edges_proj.empty):
            hit_idx = road_edges_proj.sindex.query(tile_buf)
            road_edges_tile = road_edges_proj.iloc[hit_idx].copy()
        else:
            road_edges_tile = gpd.GeoDataFrame()
        if bus_nodes_proj is not None and (not bus_nodes_proj.empty):
            bus_nodes_tile = bus_nodes_proj[bus_nodes_proj.intersects(tile_buf)].copy()
        else:
            bus_nodes_tile = gpd.GeoDataFrame()
        if bus_edges_proj is not None and (not bus_edges_proj.empty):
            bus_edges_tile = bus_edges_proj[bus_edges_proj.intersects(tile_buf)].copy()
            if "mode" in bus_edges_tile.columns:
                bus_edges_tile = bus_edges_tile[bus_edges_tile["mode"] != "walk"]
        else:
            bus_edges_tile = gpd.GeoDataFrame()
        spec = build_tile_graph_spec(
            grid,
            road_nodes_tile,
            road_edges_tile,
            gtfs_nodes_proj=bus_nodes_tile if not bus_nodes_tile.empty else None,
            gtfs_edges_proj=bus_edges_tile if not bus_edges_tile.empty else None,
            road_graph=road_graph,
            road_nodes_lookup_proj=road_nodes_proj,
            log_context=f"tile={tile_idx}",
            candidate_edge_mode=cfg_payload.get(
                "TILE_GRAPH_CANDIDATE_EDGE_MODE", "road_segments"
            ),
            snap_radius_m=cfg_payload.get("TILE_GRAPH_SNAP_RADIUS_M", 200.0),
            neighbor_k=cfg_payload.get("TILE_GRAPH_NEIGHBOR_K", 6),
            max_stops=cfg_payload.get("TILE_GRAPH_MAX_STOPS", 1500),
            stop_sigma_px=cfg_payload.get("TILE_GRAPH_STOP_SIGMA_PX", 2.0),
            edge_sigma_px=cfg_payload.get("TILE_GRAPH_EDGE_SIGMA_PX", 1.5),
            routing_mode=cfg_payload.get(
                "TILE_GRAPH_ROUTING_MODE", "road_shortest_path"
            ),
        )
        w_real, z_real = gtfs_to_activations(
            spec,
            bus_nodes_tile if not bus_nodes_tile.empty else None,
            bus_edges_tile if not bus_edges_tile.empty else None,
            road_graph=road_graph,
            candidate_edge_mode=cfg_payload.get(
                "TILE_GRAPH_CANDIDATE_EDGE_MODE", "road_segments"
            ),
            snap_radius_m=cfg_payload.get("TILE_GRAPH_SNAP_RADIUS_M", 200.0) * 1.5,
            routing_direction=cfg_payload.get(
                "BUS_PATH_ROUTING_DIRECTION", "undirected"
            ),
        )
        w_real_mode = str(cfg_payload.get("BUS_W_REAL_MODE", "binary")).strip().lower()
        if w_real_mode == "binary":
            w_real = (w_real > 0).astype(np.float32)
        elif w_real_mode == "count":
            pass
        elif w_real_mode == "global_binary":
            w_real = (w_real > 0).astype(np.float32)
        else:
            warnings.warn(
                "Unknown BUS_W_REAL_MODE='%s'; falling back to 'binary'"
                % (w_real_mode,),
                RuntimeWarning,
                stacklevel=2,
            )
            w_real = (w_real > 0).astype(np.float32)
            w_real_mode = "binary"
        renderer = Renderer(
            spec,
            soft_cap_max=cfg_payload.get("RENDERER_SOFT_CAP_MAX", 1.0),
            soft_cap_beta=cfg_payload.get("RENDERER_SOFT_CAP_BETA", 10.0),
            edge_gate_mode=cfg_payload.get("RENDERER_EDGE_GATE_MODE", "auto"),
        )
        with torch.no_grad():
            bus_t = renderer.render_from_wz(
                torch.from_numpy(w_real), torch.from_numpy(z_real)
            )
        tensor_bus = bus_t.numpy().astype(np.float32)
        bus_edge_render_mode = (
            str(cfg_payload.get("BUS_EDGE_RENDER_MODE", "candidate_edges"))
            .strip()
            .lower()
        )
        bus_path_weight_mode = (
            str(cfg_payload.get("BUS_PATH_RENDER_WEIGHT_MODE", w_real_mode))
            .strip()
            .lower()
        )
        if w_real_mode == "global_binary":
            bus_path_weight_mode = "global_binary"
        path_render_stats = {}
        if bus_edge_render_mode == "routed_paths":
            edge_img, path_render_stats = render_gtfs_paths_to_grid(
                spec,
                bus_nodes_tile if not bus_nodes_tile.empty else None,
                bus_edges_tile if not bus_edges_tile.empty else None,
                road_graph=road_graph,
                road_nodes_lookup_proj=road_nodes_proj,
                snap_radius_m=cfg_payload.get("TILE_GRAPH_SNAP_RADIUS_M", 200.0) * 1.5,
                sigma_px=cfg_payload.get("TILE_GRAPH_EDGE_SIGMA_PX", 1.5),
                weight_mode=bus_path_weight_mode,
                routing_direction=cfg_payload.get(
                    "BUS_PATH_ROUTING_DIRECTION", "undirected"
                ),
            )
            with torch.no_grad():
                tensor_bus[1] = (
                    renderer._apply_soft_cap(
                        torch.from_numpy(edge_img.astype(np.float32))
                    )
                    .numpy()
                    .astype(np.float32)
                )
        elif bus_edge_render_mode != "candidate_edges":
            warnings.warn(
                "Unknown BUS_EDGE_RENDER_MODE='%s'; falling back to legacy candidate-edge rendering"
                % (bus_edge_render_mode,),
                RuntimeWarning,
                stacklevel=2,
            )
            bus_edge_render_mode = "candidate_edges"
        mask_bus = np.any(tensor_bus > 0, axis=0)
        arrays["bus_tensor"] = tensor_bus
        arrays["bus_w_real"] = w_real.astype(np.float32)
        arrays["bus_z_real"] = z_real.astype(np.float32)
        arrays["bus_mask"] = mask_bus.astype(np.uint8)
        tile_record["meta_bus"] = tensor_metadata(
            grid_meta,
            bus_channel_names,
            meta,
            method=cfg_payload["RASTER_METHOD"],
            fill_value=fill_value,
            band_dtype=tensor_bus.dtype,
            tile_idx=tile_idx,
        )
        tile_record["meta_bus"]["candidate_stops"] = int(spec.num_stops())
        tile_record["meta_bus"]["candidate_edges"] = int(spec.num_edges())
        tile_record["meta_bus"]["active_stops"] = int(np.count_nonzero(z_real > 0))
        tile_record["meta_bus"]["active_edges"] = int(np.count_nonzero(w_real > 0))
        tile_record["meta_bus"]["w_real_mode"] = w_real_mode
        tile_record["meta_bus"]["edge_render_mode"] = bus_edge_render_mode
        tile_record["meta_bus"]["path_render_weight_mode"] = bus_path_weight_mode
        tile_record["meta_bus"]["path_routing_direction"] = cfg_payload.get(
            "BUS_PATH_ROUTING_DIRECTION", "undirected"
        )
        tile_record["meta_bus"]["edge_gate_mode"] = renderer.edge_gate_mode
        if path_render_stats:
            tile_record["meta_bus"]["path_render_stats"] = path_render_stats
        tile_record["spec"] = spec
    return (arrays, tile_record)


def main():
    raise RuntimeError(
        "The classic file-backed training_data_factory runtime has been removed. Use the top-level training_data_factory.py entrypoint with a PostGIS-backed config."
    )


if __name__ == "__main__":
    raise SystemExit(main())
