"""Check the configuration inventory and public entry points."""

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_recorded_configs_and_model_families():
    import torch
    from modules.training_config import DataConfig, ModelConfig, TrainConfig
    from modules.unet_models import build_model

    rows = json.loads((ROOT / "data/models.json").read_text())
    assert len(rows) == 40
    families = set()
    for row in rows:
        config = importlib.import_module("configs.reported." + row["model"])
        DataConfig(**config.DATA_CFG)
        TrainConfig(**config.TRAIN_CFG)
        model_config = ModelConfig(**config.MODEL_CFG)
        metadata = json.loads(
            (ROOT / "data" / (row["model"] + "_preprocessing.json")).read_text()
        )
        with torch.device("meta"):
            model = build_model(
                model_config,
                model_config.in_channels,
                model_config.out_channels,
                channel_info=metadata["channel_info"],
            )
        assert sum((p.numel() for p in model.parameters())) > 0
        families.add(model_config.architecture)
    assert families == {
        "unet",
        "resunet_aspp",
        "dual_stream_resunet",
        "triple_stream_resunet",
    }


def test_background_control_is_required():
    import pytest
    from modules.generated_suite_scoring import score_evaluation_set

    with pytest.raises(ValueError, match="background control"):
        score_evaluation_set({}, lambda tile: None, dead_reference_floor=1e-6)


def test_runtime_reanalysis(tmp_path):
    from scripts.reanalyze_paper import runtime

    runtime(tmp_path)
    assert (tmp_path / "runtime.csv").is_file()
