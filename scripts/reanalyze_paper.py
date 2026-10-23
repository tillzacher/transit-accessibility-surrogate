"""Rebuild behavioral, predictive, channel and runtime tables from recorded arrays."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def runtime(out):
    source = ROOT / "results/runtime"
    routing = pd.read_csv(source / "routing_repetitions.csv")
    surrogate = pd.read_csv(source / "surrogate_repetitions.csv")
    expected = pd.read_csv(source / "candidate_latency.csv").set_index(
        ["tile", "candidate"]
    )
    assert len(routing) == 12 and len(surrogate) == 160
    assert not routing.duplicated(["tile", "candidate", "repeat"]).any()
    assert not surrogate.duplicated(["tile", "candidate", "device", "repeat"]).any()
    assert (routing.reference_max_abs_diff == 0).all()
    assert (surrogate.batch_size == 1).all()
    rows = []
    for key, group in routing.groupby(["tile", "candidate"]):
        assert len(group) == 3
        row = dict(
            tile=key[0],
            candidate=key[1],
            routing_median_s=group.total_s.median(),
            routing_min_s=group.total_s.min(),
            routing_max_s=group.total_s.max(),
        )
        completed = (group.native_completed_work_s + group.extraction_s).median()
        for device, label in [("cuda", "gpu"), ("cpu", "cpu")]:
            samples = surrogate[
                (surrogate.tile == key[0])
                & (surrogate.candidate == key[1])
                & (surrogate.device == device)
            ]
            assert len(samples) == 20 and (samples.total_s > 0).all()
            median = samples.total_s.median()
            row[label + "_median_ms"] = 1000 * median
            row["routing_to_" + label + "_ratio"] = row["routing_median_s"] / median
            row["completed_work_plus_extraction_to_" + label + "_ratio"] = (
                completed / median
            )
        for field, value in row.items():
            if field not in ("tile", "candidate"):
                np.testing.assert_allclose(value, expected.loc[key, field], rtol=1e-12)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "runtime.csv", index=False)


def model_tables(out):
    models = json.loads((ROOT / "data/models.json").read_text())
    rows, configurations = ([], [])
    for model in models:
        name = model["model"]
        config = json.loads((ROOT / model["configuration"]).read_text())
        configurations.append(
            dict(
                model=name,
                group=model["group"],
                architecture=model["architecture"],
                width=model["width"],
                seed=model["seed"],
                target_mode=model["target_mode"],
                geometric=config["data"]["augment_flip"]
                or config["data"]["augment_rot90"],
                weak_activation=config["data"]["augment_bus_activation_scale"],
                isolated_stop=config["data"]["augment_isolated_stop"],
                stub_edge=config["data"]["augment_stub_edge"],
                normalize_bus=config["data"]["normalize_bus_channels"],
            )
        )
        with np.load(ROOT / "results/regional_arrays" / (name + ".npz")) as z:
            assert len(np.unique(z["tile_idx"])) == 818
            mae, zero, floor = [
                float(z[k].mean()) for k in ("raw_mae", "zero_mae", "floor_mae")
            ]
            skill = (zero - mae) / (zero - floor)
        with np.load(ROOT / "results/behavioral_arrays" / (name + ".npz")) as z:
            for key in sorted((k for k in z.files if k.endswith("__score"))):
                relation = key.removesuffix("__score")
                valid, passed = (z[relation + "__valid"], z[relation + "__pass"])
                assert valid.shape == (50, 105) and passed.shape == valid.shape
                rows.append(
                    dict(
                        model=name,
                        group=model["group"],
                        relation=relation,
                        support=int(valid.sum()),
                        pass_rate=float(passed[valid].mean()),
                        mae=mae,
                        zero_mae=zero,
                        pca_floor_mae=floor,
                        skill=skill,
                    )
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_model.csv", index=False)
    summary = frame.groupby(["group", "relation"])[["pass_rate", "mae", "skill"]].agg(
        ["mean", lambda v: np.std(v, ddof=0), "min", "max"]
    )
    summary.columns = [
        a + "_" + ("sd_population" if b == "<lambda_0>" else b)
        for a, b in summary.columns
    ]
    summary.to_csv(out / "configuration_results.csv")
    pd.DataFrame(configurations).to_csv(out / "model_configurations.csv", index=False)


def channel_tables(out):
    frames = []
    for path in sorted((ROOT / "results/per_channel").glob("*.json")):
        record = json.loads(path.read_text())
        assert record["tiles_evaluated"] == 818
        frame = pd.DataFrame(record["channels"])
        assert len(frame) == 105
        r2 = 1 - frame.ss_res / frame.ss_tot
        np.testing.assert_allclose(r2, frame.r2, rtol=1e-12)
        frame["r2"] = r2
        frame["target_sd"] = np.sqrt(frame.ss_tot / frame.n_pixels)
        frame["model"] = record["model"]
        frames.append(frame)
    assert len(frames) == 3
    combined = pd.concat(frames)
    combined.to_csv(out / "per_channel_models.csv", index=False)
    result = combined.groupby("name")[["r2", "target_sd"]].mean()
    result["rank_eligible"] = result.target_sd >= 0.2
    result.to_csv(out / "per_channel_summary.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runtime(args.out)
    model_tables(args.out)
    channel_tables(args.out)
    print("Recomputed paper tables, channel R² and all four runtime comparisons.")


if __name__ == "__main__":
    main()
