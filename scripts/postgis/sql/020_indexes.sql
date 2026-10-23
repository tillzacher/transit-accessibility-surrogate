CREATE INDEX IF NOT EXISTS access_runs_region_idx ON access.runs (region, created_at DESC);
CREATE INDEX IF NOT EXISTS access_runs_bounds_gix ON access.runs USING GIST (bounds);
CREATE INDEX IF NOT EXISTS access_cells_run_idx ON access.cells (run_id);
CREATE INDEX IF NOT EXISTS access_cells_geom_gix ON access.cells USING GIST (geom);
CREATE INDEX IF NOT EXISTS access_cells_centroid_gix ON access.cells USING GIST (centroid);

CREATE INDEX IF NOT EXISTS graph_datasets_region_mode_idx ON graph.datasets (region, mode);
CREATE INDEX IF NOT EXISTS graph_nodes_graph_idx ON graph.nodes (graph_key);
CREATE INDEX IF NOT EXISTS graph_nodes_geom_gix ON graph.nodes USING GIST (geom);
CREATE INDEX IF NOT EXISTS graph_edges_graph_idx ON graph.edges (graph_key);
CREATE INDEX IF NOT EXISTS graph_edges_uv_idx ON graph.edges (graph_key, u_id, v_id);
CREATE INDEX IF NOT EXISTS graph_edges_key_idx ON graph.edges (graph_key, edge_key);
CREATE INDEX IF NOT EXISTS graph_edges_geom_gix ON graph.edges USING GIST (geom);

CREATE INDEX IF NOT EXISTS poi_datasets_region_idx ON poi.datasets (region, run_id);
CREATE INDEX IF NOT EXISTS poi_layers_dataset_kind_idx ON poi.layers (dataset_key, layer_kind);
CREATE INDEX IF NOT EXISTS poi_features_layer_idx ON poi.features (dataset_key, layer_name);
CREATE INDEX IF NOT EXISTS poi_features_geom_gix ON poi.features USING GIST (geom);
