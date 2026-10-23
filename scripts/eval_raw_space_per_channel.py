"""Per-channel R^2 on the geographically held-out test tiles for one checkpoint."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from modules.training_runs import load_tile_arrays
from scripts.eval_raw_space import (
    _build_model,
    _denorm,
    _norm,
    _norm_map,
    _pca_fwd,
    _pca_inv,
    _resolve_tiles,
)


def _channel_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    return mask if mask.ndim == 2 else mask[0]


@torch.no_grad()
def evaluate_per_channel(
    checkpoint_path: Path,
    dataset_root: Path,
    schema_path: Path,
    device: str,
    split: str,
    limit: Optional[int],
) -> dict:
    manifest_json = json.loads(
        checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".manifest.json"
        ).read_text()
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ps = ckpt.get("pca_state")
    if not ps or "components" not in ps:
        raise SystemExit("checkpoint has no usable pca_state — PCA runs only")
    ci = ckpt.get("channel_info") or {}
    dc = (ckpt.get("config") or {}).get("data") or {}
    norm = _norm_map(ckpt.get("norm_stats") or {})
    n_feat = int(ps.get("n_features_in") or np.asarray(ps["mean"]).shape[0])
    apply_ctx = dc.get("pca_apply_to_context", True)
    apply_tgt = dc.get("pca_apply_to_target", True)
    norm_in = dc.get("normalize_inputs", True)
    norm_tgt = dc.get("normalize_targets", True)
    bus_ch = int(ci.get("bus", 2))
    is_total = str((dc or {}).get("target_mode", "delta")).lower() == "total"
    schema = json.loads(schema_path.read_text())
    channel_names = [
        n[: -len("_bus_delta")] if n.endswith("_bus_delta") else n
        for n in schema["target"]
    ]
    if len(channel_names) != n_feat:
        raise SystemExit(
            f"schema has {len(channel_names)} target channels, checkpoint expects {n_feat}"
        )
    model = _build_model(ckpt, device)
    tiles = _resolve_tiles(manifest_json, dataset_root, limit, split=split)
    if limit:
        tiles = tiles[:limit]
    if not tiles:
        raise SystemExit("no tiles resolved for this split")
    total = np.zeros(n_feat, dtype=np.float64)
    count = 0.0
    for _, npz_path in tiles:
        a = load_tile_arrays(npz_path)
        t = a.get("access_bus_delta")
        if t is None:
            continue
        t = t[:n_feat].astype(np.float64)
        mask_k = next(
            (k for k in ["mask_bus_delta", "bus_mask", "mask_all"] if k in a), None
        )
        mask = _channel_mask(a[mask_k] if mask_k else None)
        m = mask.astype(np.float64) if mask is not None else np.ones(t.shape[-2:])
        total += (t * m[None]).sum(axis=(1, 2))
        count += float(m.sum())
    channel_mean = total / max(count, 1.0)
    ss_res = np.zeros(n_feat, dtype=np.float64)
    ss_tot = np.zeros(n_feat, dtype=np.float64)
    n_pixels = np.zeros(n_feat, dtype=np.float64)
    n_tiles = 0
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
            target_raw = a[tk][:n_feat].astype(np.float64)
            mask_k = next(
                (k for k in ["mask_bus_delta", "bus_mask", "mask_all"] if k in a), None
            )
            mask = _channel_mask(a[mask_k] if mask_k else None)
            m = (
                mask.astype(np.float64)
                if mask is not None
                else np.ones(target_raw.shape[-2:])
            )
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
            if is_total:
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
            pred_raw = pred_raw.astype(np.float64)
            ss_res += ((pred_raw - target_raw) ** 2 * m[None]).sum(axis=(1, 2))
            ss_tot += ((target_raw - channel_mean[:, None, None]) ** 2 * m[None]).sum(
                axis=(1, 2)
            )
            n_pixels += m.sum() * np.ones(n_feat)
            n_tiles += 1
        except Exception as exc:
            print(f"  tile {tile_idx} failed: {exc}", file=sys.stderr)
    if n_tiles == 0:
        raise SystemExit("no tiles evaluated")
    r2 = np.where(ss_tot > 1e-09, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    return {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "tiles_evaluated": n_tiles,
        "channels": [
            {
                "name": name,
                "r2": float(r) if np.isfinite(r) else None,
                "ss_res": float(sr),
                "ss_tot": float(st),
                "n_pixels": float(np_c),
                "channel_mean": float(cm),
            }
            for name, r, sr, st, np_c, cm in zip(
                channel_names, r2, ss_res, ss_tot, n_pixels, channel_mean
            )
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument(
        "--schema", type=Path, default=REPO_ROOT / "data/channel_schema_105.json"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    res = evaluate_per_channel(
        args.checkpoint,
        args.dataset_root,
        args.schema,
        args.device,
        args.split,
        args.limit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    ranked = sorted(
        (c for c in res["channels"] if c["r2"] is not None), key=lambda c: c["r2"]
    )
    print(
        f"wrote {args.out}: {res['tiles_evaluated']} tiles, {len(ranked)} channels with a defined R^2"
    )
    print("worst 5:", [(c["name"], round(c["r2"], 3)) for c in ranked[:5]])
    print("best 5:", [(c["name"], round(c["r2"], 3)) for c in ranked[-5:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
