"""Recompute the complete weak-activation replication from saved set-level arrays."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    source = ROOT / "results/seed_extension"
    models = json.loads((source / "models.json").read_text())
    arrays = []
    for model in models:
        with np.load(source / (model["model"] + ".npz")) as z:
            arrays.append(dict(z))
    key = "weak_activation_leakage__"
    common = np.logical_and.reduce([a[key + "valid"] for a in arrays])
    rows = []
    for model, a in zip(models, arrays):
        score, valid, passed = (a[key + k] for k in ("score", "valid", "pass"))
        np.testing.assert_array_equal(passed, (score < 0.25) & valid)
        rows.append(
            {
                "model": model["model"],
                "seed": model["seed"],
                "pass_rate": float(passed[valid].mean()),
                "common_support_pass_rate": float(passed[common].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    expected = json.loads((source / "replication_summary.json").read_text())
    rates = frame.pass_rate
    for field, value in [
        ("mean", rates.mean()),
        ("sd", rates.std(ddof=1)),
        ("median", rates.median()),
        ("min", rates.min()),
        ("max", rates.max()),
    ]:
        assert np.isclose(value, expected["combined"][field], rtol=0, atol=1e-12)
    assert int(common.sum()) == expected["common_support_observations"]
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "weak_activation.csv", index=False)
    print("Reproduced all ten weak-activation pass rates and common-support results.")


if __name__ == "__main__":
    main()
