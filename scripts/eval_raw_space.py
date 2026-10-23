"""Raw-feature-space evaluation for a trained surrogate checkpoint."""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from modules.training_runs import load_tile_arrays, normalize_manifest_tile_records


def _pca_fwd(arr: np.ndarray, ps: Dict) -> np.ndarray:
    """(C,H,W) raw → (K,H,W) PCA space."""
    C, H, W = arr.shape
    comps = np.asarray(ps["components"], dtype=np.float64)
    mean = np.asarray(ps["mean"], dtype=np.float64)
    flat = arr.reshape(C, -1).T
    proj = (flat - mean) @ comps.T
    return proj.T.reshape(-1, H, W).astype(np.float32)


def _pca_inv(arr_k: np.ndarray, ps: Dict) -> np.ndarray:
    """(K,H,W) PCA space → (C,H,W) raw."""
    K, H, W = arr_k.shape
    comps = np.asarray(ps["components"], dtype=np.float64)
    mean = np.asarray(ps["mean"], dtype=np.float64)
    flat = arr_k.reshape(K, -1).T
    raw = flat @ comps + mean
    return raw.T.reshape(-1, H, W).astype(np.float32)


def _norm(arr, stats):
    if stats is None:
        return arr
    m, s = stats
    return (arr - m[: arr.shape[0], None, None]) / (
        s[: arr.shape[0], None, None] + 1e-06
    )


def _denorm(arr, stats):
    if stats is None:
        return arr
    m, s = stats
    return arr * (s[: arr.shape[0], None, None] + 1e-06) + m[: arr.shape[0], None, None]


def _masked(diff: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[float, float]:
    if mask is not None:
        m = (mask if mask.ndim == 2 else mask[0])[None]
        n = max(float(m.sum()) * diff.shape[0], 1.0)
        return (
            float((np.abs(diff) * m).sum() / n),
            float(np.sqrt((diff**2 * m).sum() / n)),
        )
    return (float(np.abs(diff).mean()), float(np.sqrt((diff**2).mean())))


def _build_model(ckpt: Dict, device: str):
    from modules.training_pipeline import build_model, ModelConfig

    mc_dict = ckpt["config"]["model"]
    mc = ModelConfig(
        **{
            k: v
            for k, v in mc_dict.items()
            if k in {f.name for f in fields(ModelConfig)}
        }
    )
    net = build_model(
        mc,
        in_channels=ckpt["in_channels"],
        out_channels=ckpt["out_channels"],
        channel_info=ckpt.get("channel_info"),
    )
    sd = {
        k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k: v
        for k, v in ckpt["model_state"].items()
    }
    net.load_state_dict(sd)
    net.to(device).eval()
    return net


def _norm_map(raw: Dict) -> Dict[str, Optional[Tuple]]:
    out = {}
    for k, v in (raw or {}).items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[k] = (np.asarray(v[0], np.float32), np.asarray(v[1], np.float32))
        else:
            out[k] = None
    return out


def _resolve_tiles(
    manifest_json, dataset_root, limit, split: str = "val", keep_idx=None
):
    """Tiles for the requested split ("val"|"test") if the manifest records them; else the"""
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    val_split = manifest_json.get("val_split")
    data_cfg = manifest_json.get("data_cfg", {})
    out_root = (val_split or {}).get("dataset_output_root") or data_cfg.get(
        "output_root", "training_data_runs_postgis"
    )
    selector = (val_split or {}).get("dataset_run_selector") or data_cfg.get(
        "run_selector", "latest"
    )
    dm = None
    if dataset_root is not None:
        dataset_root = Path(dataset_root)
        dm = (
            dataset_root / "manifest.json"
            if (dataset_root / "manifest.json").exists()
            else dataset_root
        )
        if dm.is_dir():
            cand = sorted(dm.rglob("manifest.json"))
            dm = cand[-1] if cand else None
    if dm is None:
        root = REPO_ROOT / out_root
        dm = root / str(selector) / "manifest.json"
    with open(dm) as fh:
        dsm = normalize_manifest_tile_records(Path(dm), json.load(fh))
    tiles = [(t.get("tile_idx"), Path(t["npz_path"])) for t in dsm.get("tiles") or []]
    idx_key = f"{split}_tile_indices"
    if val_split and val_split.get(idx_key):
        keep = set(val_split[idx_key])
        tiles = [(i, p) for i, p in tiles if i in keep]
    elif split == "test":
        raise SystemExit(
            f"checkpoint manifest has no {idx_key} — this run predates test-index persistence (2026-07-02); re-train or use --split val"
        )
    else:
        print(
            f"  (no val_split — first {limit or len(tiles)} dataset tiles)",
            file=sys.stderr,
        )
    if keep_idx is not None:
        keep_idx = set(keep_idx)
        tiles = [(i, p) for i, p in tiles if i in keep_idx]
    return tiles


@torch.no_grad()
def evaluate(
    checkpoint_path: Path,
    dataset_root: Optional[Path],
    device: str,
    limit: Optional[int],
    split: str = "val",
    keep_idx=None,
) -> Dict:
    manifest_json = json.loads(
        checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".manifest.json"
        ).read_text()
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ps = ckpt.get("pca_state")
    if not ps or "components" not in ps:
        raise SystemExit(
            "checkpoint has no usable pca_state (components/mean) — PCA runs only"
        )
    ci = ckpt.get("channel_info") or {}
    dc = (ckpt.get("config") or {}).get("data") or {}
    norm = _norm_map(ckpt.get("norm_stats") or {})
    n_feat = int(ps.get("n_features_in") or np.asarray(ps["mean"]).shape[0])
    apply_ctx = dc.get("pca_apply_to_context", True)
    apply_tgt = dc.get("pca_apply_to_target", True)
    norm_in = dc.get("normalize_inputs", True)
    norm_tgt = dc.get("normalize_targets", True)
    bus_ch = int(ci.get("bus", 2))
    model = _build_model(ckpt, device)
    _is_total = str((dc or {}).get("target_mode", "delta")).lower() == "total"
    if _is_total:
        print(
            "  target_mode=total → deriving implied delta via a zero-bus second pass",
            file=sys.stderr,
        )
    tiles = _resolve_tiles(
        manifest_json, dataset_root, limit, split=split, keep_idx=keep_idx
    )
    if limit:
        tiles = tiles[:limit]
    raw_mae = raw_rmse = floor_mae = floor_rmse = pca_mae = 0.0
    n = 0
    per_tile: Dict[str, list] = {
        "tile_idx": [],
        "raw_mae": [],
        "raw_rmse": [],
        "floor_mae": [],
        "pca_mae": [],
    }
    for tile_idx, npz_path in tiles:
        try:
            a = load_tile_arrays(npz_path)
            bus = a.get("bus_tensor")[:bus_ch].astype(np.float32)
            rk = next(
                (
                    k
                    for k in ["access_walk_transit_rail", "walk_transit_rail"]
                    if k in a
                ),
                None,
            )
            poik = "poi_tensor" if "poi_tensor" in a else None
            tk = next((k for k in ["access_bus_delta", "bus_delta"] if k in a), None)
            if rk is None or poik is None or tk is None:
                continue
            rail = a[rk][:n_feat].astype(np.float32)
            poi = a[poik][:n_feat].astype(np.float32)
            target_raw = a[tk][:n_feat].astype(np.float32)
            mask_k = next(
                (k for k in ["mask_bus_delta", "bus_mask", "mask_all"] if k in a), None
            )
            mask = a[mask_k].astype(np.float32) if mask_k else None
            rail_m = _pca_fwd(rail, ps) if apply_ctx else rail
            poi_m = _pca_fwd(poi, ps) if apply_ctx else poi
            if norm_in:
                bus_n = _norm(bus, norm.get("bus"))
                rail_m = _norm(rail_m, norm.get("rail"))
                poi_m = _norm(poi_m, norm.get("poi"))
            else:
                bus_n = bus
            inp = np.concatenate([bus_n, rail_m, poi_m], axis=0)
            pred = (
                model(torch.from_numpy(inp[None]).to(device)).squeeze(0).cpu().numpy()
            )
            if norm_tgt:
                pred = _denorm(pred, norm.get("target"))
            pred_raw = _pca_inv(pred, ps) if apply_tgt else pred
            if _is_total:
                bus_zero_n = (
                    _norm(np.zeros_like(bus), norm.get("bus"))
                    if norm_in
                    else np.zeros_like(bus)
                )
                inp0 = np.concatenate([bus_zero_n, rail_m, poi_m], axis=0)
                base = (
                    model(torch.from_numpy(inp0[None]).to(device))
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                if norm_tgt:
                    base = _denorm(base, norm.get("target"))
                base_raw = _pca_inv(base, ps) if apply_tgt else base
                pred_raw = pred_raw - base_raw
                pred = pred - base
            recon = _pca_inv(_pca_fwd(target_raw, ps), ps)
            mae, rmse = _masked(pred_raw - target_raw, mask)
            fmae, frmse = _masked(recon - target_raw, mask)
            tgt_pca = _pca_fwd(target_raw, ps)
            if norm_tgt:
                tgt_pca = _norm(tgt_pca, norm.get("target"))
                pred_pca = _norm(pred, norm.get("target"))
            else:
                pred_pca = pred
            pmae, _ = _masked(pred_pca - tgt_pca, mask)
            raw_mae += mae
            raw_rmse += rmse
            floor_mae += fmae
            floor_rmse += frmse
            pca_mae += pmae
            per_tile["tile_idx"].append(int(tile_idx))
            per_tile["raw_mae"].append(mae)
            per_tile["raw_rmse"].append(rmse)
            per_tile["floor_mae"].append(fmae)
            per_tile["pca_mae"].append(pmae)
            n += 1
        except Exception as exc:
            print(f"  tile {tile_idx} failed: {exc}", file=sys.stderr)
    if n == 0:
        raise SystemExit("no tiles evaluated")
    evr = ps.get("explained_variance_ratio")
    evr_sum = float(np.sum(evr)) if evr is not None else None
    return {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "pca_components": int(ps.get("n_components", 0)),
        "n_features_in": n_feat,
        "explained_variance_ratio_sum": evr_sum,
        "tiles_evaluated": n,
        "raw_space_mae": raw_mae / n,
        "raw_space_rmse": raw_rmse / n,
        "pca_floor_mae": floor_mae / n,
        "pca_floor_rmse": floor_rmse / n,
        "pca_reconstruction_floor": floor_mae / n,
        "pca_space_mae": pca_mae / n,
        "per_tile": per_tile,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Group-D raw-feature-space eval (inverse-PCA)."
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="which held-out split to report raw-space error on",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--save-per-tile",
        default=None,
        help="write per-tile mae/rmse/floor/pca arrays to this .npz path",
    )
    args = ap.parse_args()
    res = evaluate(
        Path(args.checkpoint),
        Path(args.dataset_root) if args.dataset_root else None,
        args.device,
        args.limit,
        split=args.split,
    )
    per_tile = res.pop("per_tile")
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    if args.save_per_tile:
        np.savez_compressed(
            args.save_per_tile, **{k: np.asarray(v) for k, v in per_tile.items()}
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
