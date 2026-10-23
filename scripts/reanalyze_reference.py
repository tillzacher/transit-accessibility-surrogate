"""Audit complete reference recovery and summarize calculated/analytic checks."""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.generated_metamorphic_suite import build_set_with_scene
from modules.opportunity_attenuation import score_attenuation
from modules.reference_relations import score_relation
from modules.evaluation_statistics import summary, write_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recovery", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--surrogates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    records = {}
    for r in json.loads((a.recovery / "index.json").read_text()):
        key = (r["set_index"], r["family"], r["condition"].split(":")[0])
        if key in records:
            raise ValueError(f"Duplicate reference condition {key}")
        path = a.recovery / r["case"]
        with (path / "gains.npz").open("rb") as f:
            assert (
                hashlib.file_digest(f, "sha256").hexdigest() == r["files"]["gains.npz"]
            )
        records[key] = path
    assert len(records) == 500, f"Recovery incomplete: {len(records)}/500"
    manifest = json.loads((a.inputs / "manifest.json").read_text())
    channels = manifest["channels"]
    floor = np.array(manifest["floor"])
    cal = json.loads((ROOT / "data/calibration.json").read_text())
    schema = json.loads((ROOT / "data/channel_schema_105.json").read_text())
    arrays = {}
    relations = {}
    known_empty = []
    missing = []

    def load(i, fam, cond):
        with np.load(records[i, fam, cond] / "gains.npz") as z:
            return {k: z[k] for k in z.files}

    for i in range(50):
        _, scene = build_set_with_scene(cal, schema, 20260811, i)
        fam = scene.families["poi_reachability_sensitivity"]
        triplet = [
            load(i, "poi_reachability_sensitivity", c)
            for c in ("reachable", "displaced", "null_background")
        ]
        with np.load(a.inputs / f"set{i:03d}" / "background.npz") as z:
            eligible = z["eligible"].copy()
        eligible[channels.index("wrk_emp_total")] = False
        for j, c in enumerate(channels):
            if not eligible[j]:
                continue
            for condition, z in zip(
                ("reachable", "displaced", "null_background"), triplet
            ):
                if z["present"][j]:
                    continue
                if condition == "null_background" and (
                    not fam.poi_background.get(j, [])
                ):
                    assert z["gain_eval"][j] == 0
                    z["present"][j] = True
                    known_empty.append({"set": i, "channel": c, "condition": condition})
                else:
                    missing.append({"set": i, "channel": c, "condition": condition})
        available = eligible.copy()
        for z in triplet:
            available &= z["present"]
        res = score_attenuation(
            *(z["gain_eval"] for z in triplet), floor, eligible=available
        )
        for k, v in res.items():
            arrays.setdefault(k, []).append(v)
        for key, z in zip(("gain_a", "gain_b", "gain_n"), triplet):
            arrays.setdefault(key, []).append(z["gain_eval"])
        for family, ref, probe, analytic in (
            ("isolated_stop_leakage", "connected", "isolated", True),
            ("stub_edge_leakage", "full_route", "stub", True),
            ("rail_connection_sensitivity", "connected", "severed_edges", True),
            ("overlap_diminishing_returns", "single", "near_overlap", False),
            ("spatial_delta_localization", "route", "route", False),
            (
                "rail_connection_sensitivity_stop",
                "connected",
                "severed_rail_stop",
                False,
            ),
        ):
            source_fam = {
                "overlap_diminishing_returns": "amplitude_response",
                "rail_connection_sensitivity_stop": "rail_connection_sensitivity",
            }.get(family, family)
            z = load(i, source_fam, ref)
            good = z["gain_eval"]
            eligible_other = z["present"].copy()
            if analytic:
                bad = np.zeros_like(good)
            elif family == "spatial_delta_localization":
                bad = z["gain_control"]
            else:
                b = load(i, source_fam, probe)
                bad = b["gain_eval"]
                eligible_other &= b["present"]
            canon = (
                "rail_connection_sensitivity" if family.endswith("_stop") else family
            )
            scores = score_relation(canon, good, bad, floor)
            valid = scores["valid"] & eligible_other
            relations.setdefault(family, []).append(
                {
                    "set": i,
                    "eligible": int(eligible_other.sum()),
                    "supported": int(valid.sum()),
                    "passing": int((scores["pass"] & valid).sum()),
                    "reference_check": (
                        "routed reference, analytic zero probe"
                        if analytic
                        else "calculated"
                    ),
                }
            )
    assert not missing, missing
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    np.savez_compressed(a.out / "reference_opportunity.npz", **arrays)
    primary = summary(arrays)
    rows = []
    for name, records_ in relations.items():
        sup = np.array([r["supported"] for r in records_])
        passed = np.array([r["passing"] for r in records_])
        draws = np.random.default_rng(20260912).integers(0, 50, (2000, 50))
        d = sup[draws].sum(axis=1)
        p = passed[draws].sum(axis=1)
        ci = np.quantile(p[d > 0] / d[d > 0], [0.025, 0.975])
        rows.append(
            {
                "relation": name,
                "supported": int(sup.sum()),
                "passing": int(passed.sum()),
                "pooled_pass": float(passed.sum() / sup.sum()),
                "mean_set_pass": float(np.mean(passed[sup > 0] / sup[sup > 0])),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "reference_check": records_[0]["reference_check"],
            }
        )
    rows.append(
        {
            "relation": "poi_reachability_sensitivity",
            "supported": primary["n_supported"],
            "passing": primary["n_pass"],
            "pooled_pass": primary["conditional_pass"],
            "mean_set_pass": float(
                np.mean(arrays["pass"].sum(1) / arrays["supported"].sum(1))
            ),
            "ci_low": primary["ci_low"],
            "ci_high": primary["ci_high"],
            "reference_check": "calculated A/B/N",
        }
    )
    write_csv(a.out / "reference_rates.csv", rows)
    comparisons = []
    for model in manifest["models"]:
        with np.load(a.surrogates / (model["model"] + ".npz")) as z:
            stack = {}
            for i in range(50):
                scores = score_attenuation(
                    z["gain_a"][i],
                    z["gain_b"][i],
                    z["gain_n"][i],
                    floor,
                    eligible=arrays["eligible"][i],
                )
                for k, v in scores.items():
                    stack.setdefault(k, []).append(v)
            comparisons.append(
                {
                    "model": model["model"],
                    **summary({k: np.stack(v) for k, v in stack.items()}),
                }
            )
    write_csv(a.out / "opportunity_common_eligibility.csv", comparisons)
    result = {
        "completed_routing_conditions": len(records),
        "opportunity_only": False,
        "synthetic_sets": 50,
        "known_empty_null_sources": known_empty,
        "missing_authored_sources": missing,
        "excluded_independent_focal_channel": "wrk_emp_total",
        "opportunity": primary,
        "relations": rows,
        "maximum_supported_absolute_residual_ratio": float(
            arrays["magnitude_ratio"][arrays["supported"]].max()
        ),
        "signed_negative_probe_min": float(
            arrays["focal_probe"][arrays["negative_probe"]].min()
        ),
        "per_set_other_relations": relations,
    }
    (a.out / "reference_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "per_set_other_relations"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
