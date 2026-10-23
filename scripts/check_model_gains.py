"""Check a checkpoint's A/B/background gains against the supplied set-zero record."""

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.model_io import load_model_checkpoint
from modules.scoring_scenarios import load_scenario, run_model_raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deposit-manifest", type=Path)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    row = next(
        (
            r
            for r in json.loads((ROOT / "data/models.json").read_text())
            if r["model"] == a.model
        )
    )
    expected_hash = row["checkpoint_sha256"]
    if a.deposit_manifest:
        deposited = next(
            (
                r
                for r in json.loads(a.deposit_manifest.read_text())["checkpoints"]
                if r["model"] == a.model
            )
        )
        assert deposited["original_sha256"] == expected_hash
        expected_hash = deposited["deposit_sha256"]
    with a.checkpoint.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == expected_hash
    model = load_model_checkpoint(str(a.checkpoint), device=a.device)
    path = ROOT / "data/opportunity_fixture"
    ref = load_scenario(path / "a.npz")
    probe = load_scenario(path / "b.npz")
    with np.load(path / "background.npz") as z:
        null = replace(ref, poi_raw=z["poi_raw"])
    errors = {}
    with np.load(ROOT / "results/opportunity" / (a.model + ".npz")) as expected:
        for name, tile in [("gain_a", ref), ("gain_b", probe), ("gain_n", null)]:
            gain = np.maximum(
                run_model_raw(model, tile, device=a.device)[:, ref.eval_mask], 0
            ).mean(1)
            reference = expected[name][0]
            errors[name] = float(np.max(np.abs(gain - reference)))
            # Saved gains come from CUDA inference; other devices differ by floating-point
            # rounding, below 0.1% of the largest channel gain for every reported model.
            assert errors[name] <= 5e-3 * float(np.max(np.abs(reference))), (name, errors[name])
    print(json.dumps({"model": a.model, "maximum_absolute_differences": errors}))


if __name__ == "__main__":
    main()
