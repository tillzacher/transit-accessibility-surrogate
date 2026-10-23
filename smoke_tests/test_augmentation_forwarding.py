"""Check that recorded augmentation settings reach the training dataset."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STUB_FIELDS = (
    "augment_stub_edge",
    "augment_stub_edge_prob",
    "augment_stub_edge_min_len_px",
    "augment_stub_edge_max_len_px",
)


class _DatasetBuilt(Exception):
    pass


def _tiles(n=6, size=8):
    rng = np.random.default_rng(0)
    return [
        {
            "run_id": "synthetic",
            "tile_idx": i,
            "arrays": {
                "bus_tensor": rng.random((2, size, size), dtype=np.float32),
                "access_walk_transit_rail": rng.random((3, size, size), dtype=np.float32),
                "poi_tensor": rng.random((3, size, size), dtype=np.float32),
                "access_bus_delta": rng.random((3, size, size), dtype=np.float32),
            },
        }
        for i in range(n)
    ]


def _stub_model():
    rows = json.loads((ROOT / "data/models.json").read_text())
    return next(
        r["model"]
        for r in rows
        if json.loads((ROOT / r["configuration"]).read_text())["data"]["augment_stub_edge"]
    )


@pytest.mark.parametrize("override", [None, (False, 0.5, 2, 4), (True, 0.7, 3, 11)])
def test_training_dataset_receives_stub_edge_settings(monkeypatch, tmp_path, override):
    from modules import training_pipeline
    from modules.training_config import DataConfig, ModelConfig, TrainConfig

    config = importlib.import_module("configs.reported." + _stub_model())
    data = dict(config.DATA_CFG)
    if override is not None:
        data.update(zip(STUB_FIELDS, override))
    expected = {field: data[field] for field in STUB_FIELDS}
    data.update(
        lazy_load=False,
        split_mode="random",
        holdout_region=None,
        pca_components=None,
        normalize_inputs=False,
        normalize_targets=False,
        preload_to_ram=False,
        precomputed_sample_cache=False,
        precompute_workers=0,
        num_workers=0,
    )
    tiles = _tiles()
    monkeypatch.setattr(
        training_pipeline, "_load_tiles", lambda *a, **k: ({}, tiles, ["synthetic"])
    )
    built = {}
    original = training_pipeline.TileDataset

    def record(*args, **kwargs):
        dataset = original(*args, **kwargs)
        if "limit" in kwargs:
            built["train"] = dataset
            raise _DatasetBuilt
        return dataset

    monkeypatch.setattr(training_pipeline, "TileDataset", record)
    train = dict(config.TRAIN_CFG, device="cpu", checkpoint_dir=str(tmp_path))
    with pytest.raises(_DatasetBuilt):
        training_pipeline.train_unet_pipeline(
            data_cfg=DataConfig(**data),
            model_cfg=ModelConfig(**config.MODEL_CFG),
            train_cfg=TrainConfig(**train),
        )
    dataset = built["train"]
    for field, value in expected.items():
        assert getattr(dataset, field) == value, field


def test_recorded_stub_configuration_is_trainable(monkeypatch):
    import train_surrogate

    seen = {}

    def capture(data_cfg, model_cfg, train_cfg):
        seen["data"] = data_cfg
        raise _DatasetBuilt

    monkeypatch.setattr(train_surrogate, "train_unet_pipeline", capture)
    monkeypatch.setattr(
        sys, "argv", ["train_surrogate.py", "--config", "configs.reported." + _stub_model()]
    )
    with pytest.raises(_DatasetBuilt):
        train_surrogate.main()
    assert seen["data"].augment_stub_edge is True
