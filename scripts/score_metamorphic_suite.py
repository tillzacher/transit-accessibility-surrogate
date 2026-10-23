"""Score a frozen generated corpus per raw accessibility channel."""

from __future__ import annotations
import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from modules.generated_suite_scoring import (
    load_generated_set,
    score_set_channels,
    score_evaluation_set,
)
from modules.model_io import load_model_checkpoint
from modules.scoring_scenarios import run_model_raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Override the calibration embedded in the corpus",
    )
    parser.add_argument(
        "--opportunity-inputs",
        type=Path,
        help="Background controls for older 13-scene corpora only",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    schema = manifest.get("schema_values")
    if schema is None:
        schema = json.loads(Path(manifest["schema"]["path"]).read_text())
    ckpt = load_model_checkpoint(args.checkpoint, device=args.device)
    calibration = manifest.get("calibration_values")
    if args.calibration is not None:
        calibration = json.loads(args.calibration.read_text())
    if calibration is None:
        calibration = json.loads((REPO / "data/calibration.json").read_text())
    floor = np.asarray(
        calibration["target"]["recommended_dead_reference_floor"], dtype=np.float32
    )
    floor_rule = calibration["target"]["floor_rule"]
    rows: dict[str, list[dict[str, np.ndarray]]] = {}
    for index, entry in enumerate(manifest["sets"]):
        tiles = load_generated_set(args.corpus / entry["directory"])
        opportunity_kwargs = {}
        if args.opportunity_inputs:
            with np.load(
                args.opportunity_inputs / f"set{index:03d}" / "background.npz"
            ) as z:
                opportunity_kwargs = {
                    "opportunity_background": replace(
                        tiles["route_poi_reachable"], poi_raw=z["poi_raw"]
                    ),
                    "opportunity_eligible": z["eligible"],
                }
        predict = lambda tile: run_model_raw(ckpt, tile, device=args.device)
        if opportunity_kwargs:
            scores = score_set_channels(
                tiles, predict, dead_reference_floor=floor, **opportunity_kwargs
            )
        else:
            scores = score_evaluation_set(tiles, predict, dead_reference_floor=floor)
        for name, values in scores.items():
            rows.setdefault(name, []).append(values)
        print(f"scored {entry['directory']}")
    report = {
        "format": "transit-surrogate-metamorphic-scores-v1",
        "checkpoint": str(args.checkpoint),
        "corpus_sha256": manifest["corpus_sha256"],
        "dead_reference_floor": np.asarray(floor).tolist(),
        "dead_reference_floor_rule": floor_rule,
        "opportunity_measurement": "opportunity-attenuation-1",
        "channels": schema["target"],
        "relations": {},
    }
    for name, sets in rows.items():
        scores = np.stack([x["score"] for x in sets])
        valid = np.stack([x["valid"] for x in sets])
        passed = np.stack([x["pass"] for x in sets])
        channel_rows = []
        for j, channel in enumerate(schema["target"]):
            ok = valid[:, j]
            channel_rows.append(
                {
                    "channel": channel,
                    "n_valid": int(ok.sum()),
                    "invalid_rate": float(1 - ok.mean()),
                    "median": float(np.median(scores[ok, j])) if ok.any() else None,
                    "q25": (
                        float(np.quantile(scores[ok, j], 0.25)) if ok.any() else None
                    ),
                    "q75": (
                        float(np.quantile(scores[ok, j], 0.75)) if ok.any() else None
                    ),
                    "pass_rate": float(passed[ok, j].mean()) if ok.any() else None,
                }
            )
        report["relations"][name] = {
            "valid_observations": int(valid.sum()),
            "invalid_rate": float(1 - valid.mean()),
            "pooled_pass_rate": float(passed[valid].mean()) if valid.any() else None,
            "channelwise": channel_rows,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    arrays = {}
    for name, sets in rows.items():
        for field in ("score", "valid", "pass"):
            arrays[name + "__" + field] = np.stack([x[field] for x in sets])
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
