# Metamorphic Testing of Learned Spatial Surrogates for Transit Accessibility

Code, recorded configurations and per-model results for the TMLR paper

> Till Zacher and Markus Lienkamp. *Metamorphic Testing of Learned Spatial Surrogates for
> Transit Accessibility.* Transactions on Machine Learning Research, 2026.
> <https://openreview.net/forum?id=bG1yIXRC8Y>

Institute of Automotive Technology, Technical University of Munich.

The main contribution is a standalone behavioral instrument: generate controlled spatial
scenes, obtain predictions, and check eight relations per accessibility channel.

| Resource | Location |
|---|---|
| Paper | <https://openreview.net/forum?id=bG1yIXRC8Y> |
| Trained checkpoints | <https://doi.org/10.5281/zenodo.22750422> (also in `model_checkpoints.txt`) |
| Accessibility pipeline (MAPPER) | <https://github.com/TUMFTM/MAPPER>, release v2.0.0, <https://doi.org/10.5281/zenodo.23557362> |
| How to cite | `CITATION.cff` |

## 1. Generate and score the suite

Generation and array-level scoring require only NumPy and SciPy:

```bash
python -m pip install -r requirements-suite.txt
python scripts/generate_metamorphic_suite.py --calibration data/calibration.json --schema data/channel_schema_105.json --out generated_suite
```

This generates 50 sets of 14 scenes, including the background-only opportunity control.
Use `--n-sets 1` for a small smoke check. The corpus embeds its schema and calibration,
so it can be moved without depending on the original generation paths.

To evaluate any predictor, use the Python interface:

```python
from modules.generated_suite_scoring import load_generated_set, score_evaluation_set

tiles = load_generated_set(corpus_directory / "gen001")
scores = score_evaluation_set(tiles, predict, dead_reference_floor=channel_floors)
```

`predict(tile)` must return a raw accessibility-gain array shaped `(105, 200, 200)`.
Read `channel_floors` from the corpus manifest's
`calibration_values.target.recommended_dead_reference_floor`.
No training pipeline, database, or routing engine is required.

For the supplied model architectures, additionally install PyTorch and provide a checkpoint:

```bash
python scripts/score_metamorphic_suite.py --checkpoint checkpoints/model_001.pt --corpus generated_suite --device cpu --out analysis/model_001.json
```

The scorer writes JSON summaries and per-set/channel NPZ arrays. Unsupported observations
are excluded, not counted as passes. Opportunity scoring requires the background control.

## 2. Checkpoints

Weights are archived separately (see `model_checkpoints.txt`). `data/models.json` lists every
model with its identifier (`model_001` to `model_040`), the run name used in the paper's
experiment records (`run_name`), architecture, width, seed, configuration group and checkpoint
SHA-256. The ten weak-activation replications are listed in `results/seed_extension/models.json`.
Download the archived files into `checkpoints/` and verify them with
`sha256sum -c data/CHECKPOINT_SHA256SUMS`. `scripts/check_model_gains.py` checks a checkpoint
against its recorded hash and saved gains. The archived files carry neutral metadata paths, so
pass `data/DEPOSIT_MANIFEST.json`, which links each archived file hash to the recorded
checkpoint hash:

```bash
python scripts/check_model_gains.py --model model_001 --checkpoint checkpoints/model_001.pt --deposit-manifest data/DEPOSIT_MANIFEST.json --device cpu
```

## 3. Reproduce the reported numerical results

Install NumPy, SciPy, pandas and pytest, then run:

```bash
python scripts/reproduce.py --out analysis
python scripts/reanalyze_reference.py --recovery data/reference_cases --inputs data/reference_scoring --surrogates results/opportunity --out analysis/reference
python -m pytest -q -p no:cacheprovider smoke_tests
```

These commands recompute behavioral tables, opportunity effects, threshold sweeps, regional
prediction, per-channel results, runtime comparisons and ten-initialization replication
from included arrays. Model-construction and training-wiring tests also require PyTorch.
This is saved-result reanalysis, not independent retraining. Figure-ready numerical tables
are produced; publication-layout plotting scripts are not included.

## 4. Training and geographic evaluation

Install PyTorch and `requirements.txt`. The 40 recorded configurations are in
`configs/reported/`; seven replication configurations are in `configs/seed_extension/`.
Preprocessing arrays and channel metadata are under `data/`.

```bash
python train_surrogate.py --config configs.reported.model_001 --data-root data --checkpoint-dir checkpoints
python scripts/eval_raw_space.py --help
python scripts/eval_raw_space_per_channel.py --help
python training_data_factory.py --config configs.tile_factory
```

Training tiles are not redistributed; they can be rebuilt from open data with the tile factory
and MAPPER. The expected layout is

```text
<data-root>/                     # --data-root, SURROGATE_DATA_ROOT, or output_root in the config ("data")
└── training_tiles/              # run_selector in the config
    ├── manifest.json            # run manifest with the tile list
    └── ...                      # per-tile arrays referenced by the manifest
```

`--checkpoint-dir` (or `SURROGATE_CHECKPOINT_DIR`) overrides the configured `checkpoint_dir`.

The tile factory reads MAPPER outputs from a PostGIS store. Supply the registered source keys
through `REFERENCE_RUN_ID`, `POI_DATASET_KEY`, `ROAD_GRAPH_KEY` and `BUS_GRAPH_KEY`, and the
connection through either `SURROGATE_POSTGIS_URL` or the individual variables
`SURROGATE_POSTGIS_HOST`, `SURROGATE_POSTGIS_PORT` (default 5432), `SURROGATE_POSTGIS_DB`,
`SURROGATE_POSTGIS_USER`, `SURROGATE_POSTGIS_PASSWORD` and optionally `SURROGATE_POSTGIS_SSLMODE`.
The same keys can be placed in a `.env.postgis` file in the repository root or the working
directory, or in a file named by `SURROGATE_POSTGIS_ENV_FILE`. Credentials are never stored in
this repository.

### Building the PostGIS store

Copy `scripts/postgis/postgis.env.example` to `.env.postgis` (gitignored) and set a password.
Start a local database and create the schema; the compose file applies `scripts/postgis/sql/`
on first start, and `init_schema.py` applies the same files to an existing server:

```bash
docker compose --env-file .env.postgis -f docker-compose.postgis.yml up -d
python scripts/postgis/init_schema.py
```

Then ingest one MAPPER run in this order. `<run-dir>` is the run output directory
(`cells.parquet`, `run_manifest.json`, `pois.csv`, `src_wrk_<mode>.csv`), `<graph-dir>` holds
the run's graph files (`<mode>.bin`), and `<ring>` and `<gtfs-bus>` are the run's clip ring and
bus feed:

```bash
python scripts/postgis/ingest_access_run.py --run-dir <run-dir>
python scripts/postgis/ingest_graph.py --region <region> --graph <graph-dir>/drive.bin
python scripts/postgis/ingest_graph.py --region <region> --gtfs <gtfs-bus> --ring <ring>
python scripts/postgis/ingest_poi_bundles.py --run-dir <run-dir> --workplace-graph <graph-dir>/walk_transit_rail.bin
```

The bus graph holds the GTFS service legs and their stops; it is built with MAPPER's
`build_transit_cpp` (`--engine`, default `../MAPPER`, see Section 5). Each script prints the
key it registered. Use them as `REFERENCE_RUN_ID` (the run directory name), `ROAD_GRAPH_KEY`
and `BUS_GRAPH_KEY` (`<region>|<mode>`) and `POI_DATASET_KEY`
(`poi_bundle::<run id>::training_tiles`), then run the tile factory:

```bash
python training_data_factory.py --config configs.tile_factory
```

`--dry-run` on the graph and POI scripts resolves and checks the inputs without writing. The
POI ingest fails if a `*_walk_transit_bus` column has no matching layer or rail baseline.
Tiles are placed inside the German state boundaries in
`modules/data/Deutschland_Bundesländergrenzen.geojson` (© GeoBasis-DE / BKG, VG250, Data licence
Germany – attribution – 2.0).

## 5. Routing checks

The synthetic reference cases are calculated with MAPPER v2.0.0
(<https://doi.org/10.5281/zenodo.23557362>). Clone it next to this repository and build it as
described in its README (Linux, GCC 11 or newer, CMake 3.25 or newer, no root access needed):

```bash
git clone --branch v2.0.0 https://github.com/TUMFTM/MAPPER.git ../MAPPER
bash ../MAPPER/source/infra/bootstrap_cpp_worker.sh
python scripts/run_reference_case.py --case case_0000 --out analysis/reference_case_0000 --threads 2
```

Each run writes `comparison.json` with the largest absolute difference from the stored gains;
all 500 cases reproduce them exactly. Pass `--engine <path>` for a MAPPER checkout elsewhere.

The 500 synthetic conditions are in `data/reference_cases/`. Routing checks, analytical
zero-service controls and raster-only stress tests provide different evidence, as explained
in the paper. Original regional runtime inputs are not included.

### Regenerating the reference cases

The case inputs (OSM extract, GTFS feeds, clip ring) and the metadata files are rebuilt from
the generator in `modules/synthetic_world.py`. This needs `osmium` (osmium-tool) and the POI
taxonomy of the MAPPER checkout, but not a MAPPER build:

```bash
python scripts/generate_reference_cases.py --out analysis/reference_cases --jobs 8 --compare data/reference_cases
```

All 500 cases take about 25 CPU minutes (about 30 s per set). `--cases case_0000 case_0005`
limits the run, and `--compare` checks each file against the supplied tree. OSM extracts and
rings are byte-identical; GTFS archives are written with fixed timestamps and match the
supplied tables exactly. Gains come from routing, so `--stored-gains data/reference_cases`
copies the stored results, and the output can then replace `data/reference_cases`.

## Repository layout

`SOURCE_MAP.md` maps each component to its entry points and modules. `SHA256SUMS` lists file
checksums (`sha256sum -c SHA256SUMS`).

## License

The code in this repository is released under the Apache License 2.0 (`LICENSE`, `NOTICE`). Third-party dependencies retain
their upstream terms, and geographic source data carry their own licences (for example ODbL for
OpenStreetMap).

## Citation

Please cite the paper if you use this code; `CITATION.cff` contains the reference.

```bibtex
@article{zacher2026metamorphic,
  title   = {Metamorphic Testing of Learned Spatial Surrogates for Transit Accessibility},
  author  = {Zacher, Till and Lienkamp, Markus},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=bG1yIXRC8Y}
}
```
