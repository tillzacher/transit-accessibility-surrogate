"""Train one transit-accessibility surrogate from a configuration module."""

from __future__ import annotations
import argparse
import importlib
import os
from pathlib import Path
from modules.training_config import DataConfig, ModelConfig, TrainConfig
from modules.training_pipeline import train_unet_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="recorded module, e.g. configs.reported.model_001",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("SURROGATE_DATA_ROOT"),
        help="directory containing the training tile run (default: the configured output_root)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("SURROGATE_CHECKPOINT_DIR"),
        help="output directory for checkpoints (default: the configured checkpoint_dir)",
    )
    args = parser.parse_args()
    config = importlib.import_module(args.config)
    data_values = dict(config.DATA_CFG)
    train_values = dict(config.TRAIN_CFG)
    if args.data_root:
        data_values["output_root"] = str(Path(args.data_root).expanduser())
    if args.checkpoint_dir:
        train_values["checkpoint_dir"] = str(Path(args.checkpoint_dir).expanduser())
    data = DataConfig(**data_values)
    model = ModelConfig(**config.MODEL_CFG)
    train = TrainConfig(**train_values)
    result = train_unet_pipeline(data_cfg=data, model_cfg=model, train_cfg=train)
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"best validation loss: {result.best_val_loss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
