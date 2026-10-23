"""Geographic-holdout test-set evaluation, split by region."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from modules.geo_split import resolve_region_polygon, tile_box_3857
from modules.training_runs import normalize_manifest_tile_records
import scripts.eval_raw_space as ev

DEFAULT_HOLDOUT_REGIONS = [
    {"kind": "radius", "lon": 11.576, "lat": 48.137, "radius_km": 35},
    {"kind": "bbox", "bounds": [12.4, 53.2, 14.4, 54.5]},
]


def _holdout_regions(ckpt_path: Path):
    """Region specs: prefer the run's geo_split.json, else the paper defaults."""
    gs = ckpt_path.with_suffix(".geo_split.json")
    if not gs.exists():
        gs = Path(str(ckpt_path).replace(".pt", ".geo_split.json"))
    if gs.exists():
        d = json.loads(gs.read_text())
        regs = d.get("holdout_region")
        if regs:
            return regs if isinstance(regs, list) else [regs]
    return DEFAULT_HOLDOUT_REGIONS


def _region_name(spec, i):
    if spec.get("kind") == "radius":
        return f"munich_r{int(spec.get('radius_km', 0))}km"
    if spec.get("kind") == "bbox":
        return "mecklenburg_vorpommern"
    return f"region{i}"


def _tile_region_map(dataset_root: Path, test_idx: set, region_specs):
    """Map each test tile_idx -> region name by centroid containment."""
    dm = dataset_root / "manifest.json"
    if not dm.exists():
        cand = sorted(dataset_root.rglob("manifest.json"))
        dm = cand[-1]
    dsm = normalize_manifest_tile_records(dm, json.loads(dm.read_text()))
    polys = [
        (_region_name(s, i), resolve_region_polygon(s))
        for i, s in enumerate(region_specs)
    ]
    out = {}
    for t in dsm.get("tiles") or []:
        tid = t.get("tile_idx")
        if tid not in test_idx:
            continue
        try:
            c = tile_box_3857(t).centroid
        except Exception:
            continue
        for name, poly in polys:
            if poly.contains(c):
                out[tid] = name
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", default="data/training_tiles")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    manifest = json.loads(ckpt.with_suffix(ckpt.suffix + ".manifest.json").read_text())
    test_idx = set((manifest.get("val_split") or {}).get("test_tile_indices") or [])
    if not test_idx:
        raise SystemExit(
            "checkpoint manifest has no test_tile_indices (geographic split required)"
        )
    regions = _holdout_regions(ckpt)
    tmap = _tile_region_map(Path(args.dataset_root), test_idx, regions)
    by_region: dict = {}
    for tid, rname in tmap.items():
        by_region.setdefault(rname, set()).add(tid)
    results = {}
    blocks = list(by_region.items()) + [("combined", test_idx)]
    for rname, ids in blocks:
        if not ids:
            continue
        r = ev.evaluate(
            ckpt, Path(args.dataset_root), args.device, None, split="test", keep_idx=ids
        )
        floor = r["pca_reconstruction_floor"]
        rel = (r["raw_space_mae"] - floor) / floor if floor > 1e-09 else float("nan")
        results[rname] = {
            "tiles": r["tiles_evaluated"],
            "raw_space_mae": round(r["raw_space_mae"], 4),
            "raw_space_rmse": round(r["raw_space_rmse"], 4),
            "pca_space_mae": round(r["pca_space_mae"], 4),
            "pca_reconstruction_floor": round(floor, 4),
            "rel_error_above_floor": round(rel, 4),
        }
        print(
            f"{rname:26s} n={r['tiles_evaluated']:4d}  raw_MAE={r['raw_space_mae']:.4f}  pca_MAE={r['pca_space_mae']:.4f}  floor={floor:.4f}  rel_err={rel:.3f}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
