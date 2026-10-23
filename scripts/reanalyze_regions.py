"""Recompute regional prediction and configuration means from per-tile errors."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.evaluation_statistics import write_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in json.loads((ROOT / "data/models.json").read_text()):
        with np.load(ROOT / "results/regional_arrays" / (model["model"] + ".npz")) as z:
            assert len(set(z["tile_idx"])) == 818
            for region, n in [
                ("Munich", 81),
                ("Mecklenburg-Vorpommern", 737),
                ("Combined", 818),
            ]:
                mask = (
                    np.ones(818, dtype=bool)
                    if region == "Combined"
                    else z["region"] == region
                )
                assert mask.sum() == n
                mae = float(z["raw_mae"][mask].mean())
                floor = float(z["floor_mae"][mask].mean())
                zero = float(z["zero_mae"][mask].mean())
                rows.append(
                    {
                        "model": model["model"],
                        "group": model["group"],
                        "region": region,
                        "n_tiles": n,
                        "mae": mae,
                        "zero_mae": zero,
                        "pca_reconstruction_mae": floor,
                        "skill": (zero - mae) / (zero - floor),
                    }
                )
    write_csv(a.out / "model_regions.csv", rows)
    pd.DataFrame(rows).groupby(["group", "region"])[["mae", "skill"]].agg(
        ["mean", "std"]
    ).to_csv(a.out / "configuration_regions.csv")
    print(f"Recomputed {len(rows)} model/region summaries on the original split.")


if __name__ == "__main__":
    main()
