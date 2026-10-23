"""Recorded effective scientific configuration."""

import json
from pathlib import Path

_c = json.loads(Path(__file__).with_suffix(".json").read_text())
DATA_CFG = _c["data"]
MODEL_CFG = _c["model"]
TRAIN_CFG = _c["train"]
