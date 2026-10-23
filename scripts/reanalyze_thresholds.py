"""Reproduce the complete fixed-support threshold sweep from included scores."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.evaluation_statistics import write_csv

ALIASES = dict(
    isl="isolated_stop_leakage",
    sel="stub_edge_leakage",
    sdl="spatial_delta_localization",
    wal="weak_activation_leakage",
    prs="poi_reachability_sensitivity",
    rcs="rail_connection_sensitivity",
    odr="overlap_diminishing_returns",
    bis="bus_intensity_saturation",
)


def decide(score, alias, gate):
    if alias in ("isl", "sel", "sdl", "wal"):
        return score < gate
    if alias in ("prs", "rcs"):
        return score > gate
    return np.abs(score - 0.5) <= gate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for model in json.loads((ROOT / "data/models.json").read_text()):
        with np.load(
            ROOT / "results/behavioral_arrays" / (model["model"] + ".npz")
        ) as z:
            for alias, relation in ALIASES.items():
                score, valid, passed = [
                    z[relation + "__" + k] for k in ("score", "valid", "pass")
                ]
                base = (
                    0.25
                    if alias in ("isl", "sel", "sdl", "wal")
                    else 0.75 if alias in ("prs", "rcs") else 0.15
                )
                np.testing.assert_array_equal(
                    decide(score, alias, base)[valid], passed[valid]
                )
                low, high = (
                    (0.1, 0.4)
                    if base == 0.25
                    else (0.6, 0.9) if base == 0.75 else (0.05, 0.25)
                )
                for gate in np.round(np.arange(low, high + 0.001, 0.01), 2):
                    rows.append(
                        {
                            "model": model["model"],
                            "group": model["group"],
                            "probe": alias,
                            "threshold": float(gate),
                            "support": int(valid.sum()),
                            "pass_rate": float(
                                decide(score, alias, gate)[valid].mean()
                            ),
                        }
                    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(a.out, rows)
    print(f"Reproduced baseline decisions and {len(rows)} sweep rows.")


if __name__ == "__main__":
    main()
