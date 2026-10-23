"""Recompute the reported opportunity summaries from included per-model gains."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.opportunity_attenuation import score_attenuation
from modules.evaluation_statistics import summary, write_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    floor = np.array(
        json.loads((ROOT / "data/calibration.json").read_text())["target"][
            "recommended_dead_reference_floor"
        ]
    )
    inventory = json.loads((ROOT / "data/models.json").read_text())
    rows = []
    sweeps = []
    for model in inventory:
        with np.load(ROOT / "results/opportunity" / (model["model"] + ".npz")) as z:
            stack = {}
            for i in range(50):
                result = score_attenuation(
                    z["gain_a"][i],
                    z["gain_b"][i],
                    z["gain_n"][i],
                    floor,
                    eligible=z["eligible"][i],
                )
                for key, value in result.items():
                    if value.dtype.kind in "bUS":
                        np.testing.assert_array_equal(value, z[key][i])
                    else:
                        np.testing.assert_allclose(value, z[key][i], equal_nan=True)
                    stack.setdefault(key, []).append(value)
            stack = {k: np.stack(v) for k, v in stack.items()}
            rows.append(
                {"model": model["model"], "group": model["group"], **summary(stack)}
            )
            supported = stack["supported"]
            scores = stack["score"]
            for threshold in np.linspace(0.55, 0.9, 36):
                sweeps.append(
                    {
                        "model": model["model"],
                        "threshold": float(threshold),
                        "support": int(supported.sum()),
                        "pass_rate": float(
                            ((scores > threshold) & supported).sum() / supported.sum()
                        ),
                    }
                )
    write_csv(a.out / "opportunity_summary.csv", rows)
    write_csv(a.out / "opportunity_thresholds.csv", sweeps)
    print(f"All stored decisions reproduced for {len(rows)} models.")


if __name__ == "__main__":
    main()
