# Implementation map

| Component | Entry points and core modules |
|---|---|
| Procedural generation | `scripts/generate_metamorphic_suite.py`, `modules/generated_metamorphic_suite.py` |
| Standalone relational scoring | `modules/generated_suite_scoring.py:score_evaluation_set`, `modules/behavioral_thresholds.py`, `modules/opportunity_attenuation.py` |
| Scenario arrays and masks | `modules/scoring_scenarios.py` |
| Checkpoint evaluation | `scripts/score_metamorphic_suite.py`, `modules/model_io.py`, `modules/surrogate_inference.py` |
| Saved-result reanalysis | `scripts/reproduce.py`, `scripts/reanalyze_*.py`, `results/` |
| Model definitions and training | `modules/unet_models.py`, `modules/training_pipeline.py`, `modules/training_losses.py`, `train_surrogate.py` |
| Recorded configurations | `configs/reported/`, `configs/seed_extension/`, `data/models.json` |
| Geographic prediction | `scripts/eval_raw_space.py`, `scripts/eval_raw_space_per_channel.py`, `modules/geo_split.py` |
| PostGIS store | `docker-compose.postgis.yml`, `scripts/postgis/` (schema and MAPPER ingest) |
| PostGIS tile factory | `training_data_factory.py`, `configs/tile_factory.py`, `modules/training_data_factory_postgis_runner.py`, spatial/raster helpers |
| Synthetic reference worlds | `scripts/generate_reference_cases.py`, `modules/synthetic_world.py` |
| Synthetic routing checks | `scripts/run_reference_case.py`, `scripts/reanalyze_reference.py`, `modules/synthetic_world.py`; engine: MAPPER v2.0.0 (external) |

The standalone instrument does not import the training or geographic pipeline.
Native-engine build output and standard Python warnings are diagnostic, not external logging services.
