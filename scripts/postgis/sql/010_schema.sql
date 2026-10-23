-- Spatial serving store read by modules/postgis_tile_source.py.
CREATE SCHEMA IF NOT EXISTS access;
CREATE SCHEMA IF NOT EXISTS graph;
CREATE SCHEMA IF NOT EXISTS poi;

CREATE TABLE IF NOT EXISTS access.runs (
    run_id TEXT PRIMARY KEY,
    manifest_path TEXT,
    region TEXT,
    grid_crs TEXT,
    h3_res INTEGER,
    created_at TIMESTAMPTZ,
    config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    tensor_schema_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    feature_groups_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    bounds geometry(MultiPolygon, 4326),
    source_signature TEXT,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS access.cells (
    run_id TEXT NOT NULL REFERENCES access.runs(run_id) ON DELETE CASCADE,
    cell_id TEXT NOT NULL,
    geom geometry(Polygon, 4326) NOT NULL,
    centroid geometry(Point, 4326),
    values_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    row_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, cell_id)
);

CREATE TABLE IF NOT EXISTS graph.datasets (
    graph_key TEXT PRIMARY KEY,
    region TEXT,
    mode TEXT,
    cache_path TEXT,
    crs TEXT,
    build_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_signature TEXT,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS graph.nodes (
    graph_key TEXT NOT NULL REFERENCES graph.datasets(graph_key) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    geom geometry(Point, 4326) NOT NULL,
    attrs JSONB NOT NULL DEFAULT '{}'::jsonb,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (graph_key, node_id)
);

CREATE TABLE IF NOT EXISTS graph.edges (
    edge_id BIGSERIAL PRIMARY KEY,
    graph_key TEXT NOT NULL REFERENCES graph.datasets(graph_key) ON DELETE CASCADE,
    u_id TEXT NOT NULL,
    v_id TEXT NOT NULL,
    edge_key TEXT,
    geom geometry(LineString, 4326) NOT NULL,
    length_m DOUBLE PRECISION,
    mode TEXT,
    cost DOUBLE PRECISION,
    reverse_cost DOUBLE PRECISION,
    attrs JSONB NOT NULL DEFAULT '{}'::jsonb,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS poi.datasets (
    dataset_key TEXT PRIMARY KEY,
    run_id TEXT,
    region TEXT,
    cache_dir TEXT,
    config_module TEXT,
    source_signature TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS poi.layers (
    dataset_key TEXT NOT NULL REFERENCES poi.datasets(dataset_key) ON DELETE CASCADE,
    layer_name TEXT NOT NULL,
    base_name TEXT NOT NULL,
    layer_kind TEXT NOT NULL,
    bus_access_column TEXT,
    bus_delta_column TEXT,
    rail_baseline_column TEXT,
    feature_count BIGINT NOT NULL DEFAULT 0,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (dataset_key, layer_name)
);

CREATE TABLE IF NOT EXISTS poi.features (
    feature_id BIGSERIAL PRIMARY KEY,
    dataset_key TEXT NOT NULL,
    layer_name TEXT NOT NULL,
    geom geometry(Geometry, 4326) NOT NULL,
    weight DOUBLE PRECISION,
    attrs JSONB NOT NULL DEFAULT '{}'::jsonb,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT poi_features_layer_fk
        FOREIGN KEY (dataset_key, layer_name)
        REFERENCES poi.layers(dataset_key, layer_name)
        ON DELETE CASCADE
);
