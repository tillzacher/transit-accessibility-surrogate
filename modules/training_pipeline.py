from __future__ import annotations
import warnings
import math
import os
import random
import time
import inspect
import json
import gc
import hashlib
import threading
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext
from modules.model_metadata import build_model_io_spec
from modules.training_runs import (
    group_training_runs,
    list_tile_array_keys,
    list_training_runs,
    load_run,
    load_run_lazy,
    load_tile_arrays,
)

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None
from modules.training_config import (
    RunSelector,
    DataConfig,
    ModelConfig,
    TrainConfig,
    TrainResult,
)
from modules.training_losses import (
    _prepare_mask,
    masked_loss,
    masked_mae,
    anisotropic_tv,
)
from modules.unet_models import (
    UNet,
    ResUNetASPP,
    DualStreamResUNet,
    build_model,
)


def _checkpoint_history_path(train_cfg: TrainConfig) -> Optional[Path]:
    if not train_cfg.save_checkpoint:
        return None
    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    if train_cfg.checkpoint_history_name:
        return checkpoint_dir / str(train_cfg.checkpoint_history_name)
    return checkpoint_dir / "unet_latest.pt.history.json"


def _checkpoint_history_paths(
    train_cfg: TrainConfig, latest_checkpoint_path: Optional[Path]
) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    if train_cfg.save_checkpoint:
        for ckpt_path in _checkpoint_output_paths(train_cfg):
            history_path = ckpt_path.with_suffix(ckpt_path.suffix + ".history.json")
            key = str(history_path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(history_path)
    legacy_path = _checkpoint_history_path(train_cfg)
    if legacy_path is not None:
        key = str(legacy_path)
        if key not in seen:
            seen.add(key)
            paths.append(legacy_path)
    if latest_checkpoint_path is not None:
        sidecar_path = latest_checkpoint_path.with_suffix(
            latest_checkpoint_path.suffix + ".history.json"
        )
        key = str(sidecar_path)
        if key not in seen:
            seen.add(key)
            paths.append(sidecar_path)
    return paths


def _checkpoint_output_paths(train_cfg: TrainConfig) -> List[Path]:
    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    names: List[str] = ["unet_latest.pt"]
    config_name = train_cfg.config_checkpoint_name or train_cfg.checkpoint_name
    if config_name:
        names.append(str(config_name))
    unique_paths: List[Path] = []
    seen = set()
    for name in names:
        path = checkpoint_dir / str(name)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def _resume_state_path(train_cfg: TrainConfig) -> Optional[Path]:
    """Path of the per-epoch full training-state file used for resume, or None if this run
    doesn't save checkpoints. Derived from the primary (config-named) checkpoint so two
    concurrent runs don't collide on the shared unet_latest.pt name."""
    paths = _checkpoint_output_paths(train_cfg)
    if not paths:
        return None
    base = paths[-1]
    return base.with_suffix(base.suffix + ".resume.pt")


def data_fingerprint(data_cfg) -> str:
    """Stable hash of the data/split-defining fields of a DataConfig."""
    import hashlib, json as _json

    keys = (
        "run_selector",
        "split",
        "split_seed",
        "split_mode",
        "holdout_region",
        "holdout_buffer_m",
        "holdout_val_fraction",
        "pca_components",
        "target_mode",
        "pca_fit_source",
        "pca_apply_to_context",
        "pca_apply_to_target",
        "normalize_inputs",
        "normalize_targets",
        "normalize_bus_channels",
        "slim_mode",
    )
    payload = {k: getattr(data_cfg, k, None) for k in keys}
    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _save_resume_state(
    path: Path,
    *,
    epoch,
    best_val,
    best_epoch,
    patience_counter,
    global_step,
    model,
    optimizer,
    scheduler,
    scaler,
    history,
    data_fp=None,
) -> None:
    """Atomically write the full training state needed to continue from the next epoch."""
    import numpy as _np

    state = {
        "epoch": int(epoch),
        "best_val": (
            None if best_val is None or not math.isfinite(best_val) else float(best_val)
        ),
        "best_epoch": best_epoch,
        "patience_counter": int(patience_counter),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy_rng": _np.random.get_state(),
        "data_fingerprint": data_fp,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, tmp)
    tmp.replace(path)


def _write_checkpoint_history(
    train_cfg: TrainConfig,
    history: Dict[str, List[float]],
    *,
    epoch: int,
    total_epochs: int,
    best_val: Optional[float],
    best_epoch: Optional[int],
    latest_val_loss: Optional[float],
    latest_checkpoint_path: Optional[Path],
    info_text: Optional[str] = None,
) -> Optional[Path]:
    history_paths = _checkpoint_history_paths(train_cfg, latest_checkpoint_path)
    if not history_paths:
        return None
    payload = {
        "version": 1,
        "epoch": int(epoch),
        "total_epochs": int(total_epochs),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_val_loss": (
            None if best_val is None or not math.isfinite(best_val) else float(best_val)
        ),
        "latest_val_loss": None if latest_val_loss is None else float(latest_val_loss),
        "checkpoint_path": (
            None if latest_checkpoint_path is None else str(latest_checkpoint_path)
        ),
        "history": _json_safe(history),
        "plot": {
            "log_y": bool(train_cfg.loss_plot_log_scale),
            "show_mae": bool(train_cfg.loss_plot_show_mae),
        },
        "info_text": info_text,
    }
    first_path: Optional[Path] = None
    for history_path in history_paths:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(payload, indent=2))
        if first_path is None:
            first_path = history_path
    return first_path


def _pick_key(
    arrays: Dict[str, np.ndarray],
    preferred: Sequence[str],
    fallback_prefix: Optional[str] = None,
) -> Optional[str]:
    for key in preferred:
        if key and key in arrays:
            return key
    if fallback_prefix:
        for key in arrays.keys():
            if key.startswith(fallback_prefix):
                return key
    return None


def _pick_name(
    names: Sequence[str],
    preferred: Sequence[str],
    fallback_prefix: Optional[str] = None,
) -> Optional[str]:
    name_set = set(names)
    for key in preferred:
        if key and key in name_set:
            return key
    if fallback_prefix:
        for key in names:
            if key.startswith(fallback_prefix):
                return key
    return None


def _target_key_preferences(
    target_mode: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    mode = str(target_mode or "delta").lower()
    if mode == "delta":
        return (("access_bus_delta", "bus_delta"), ("mask_bus_delta", "bus_mask"))
    if mode == "total":
        return (
            ("access_walk_transit_total", "access_total"),
            ("mask_walk_transit_total", "mask_total", "mask_all"),
        )
    raise ValueError(
        f"Unsupported target_mode '{target_mode}' (expected 'delta' or 'total')"
    )


def _pick_target_name(names: Sequence[str], target_mode: str) -> Optional[str]:
    target_names, _ = _target_key_preferences(target_mode)
    return _pick_name(names, target_names)


def _pick_target_mask_name(names: Sequence[str], target_mode: str) -> Optional[str]:
    _, mask_names = _target_key_preferences(target_mode)
    return _pick_name(names, mask_names)


def _pick_target_key(arrays: Dict[str, np.ndarray], target_mode: str) -> Optional[str]:
    target_names, _ = _target_key_preferences(target_mode)
    return _pick_key(arrays, target_names)


def _pick_target_mask_key(
    arrays: Dict[str, np.ndarray], target_mode: str
) -> Optional[str]:
    _, mask_names = _target_key_preferences(target_mode)
    return _pick_key(arrays, mask_names)


def _normalize_split(split: Sequence[float]) -> Tuple[float, float, float]:
    if len(split) != 3:
        raise ValueError("split must be a 3-tuple (train, val, test)")
    total = float(sum(split))
    if total <= 0:
        raise ValueError("split fractions must sum to a positive value")
    if not math.isclose(total, 1.0):
        split = tuple((x / total for x in split))
    return tuple(split)


def _resolve_run_paths(output_root: Path, selector: RunSelector) -> List[Path]:
    if isinstance(selector, (int, str, Path)):
        selectors: List[Union[int, str, Path]] = [selector]
    else:
        selectors = list(selector)
    paths: List[Path] = []
    runs_df = (
        list_training_runs(output_root)
        .reset_index()
        .rename(columns={"index": "run_idx"})
    )
    for sel in selectors:
        if isinstance(sel, Path):
            sel_path = Path(sel)
            if sel_path.is_dir():
                manifest_path = sel_path / "manifest.json"
                if manifest_path.exists():
                    paths.append(manifest_path)
                    continue
            paths.append(sel_path)
            continue
        if isinstance(sel, str):
            sel_path = Path(sel)
            if sel.endswith(".json") and sel_path.exists():
                paths.append(sel_path)
                continue
            if sel_path.is_dir():
                manifest_path = sel_path / "manifest.json"
                if manifest_path.exists():
                    paths.append(manifest_path)
                    continue
            candidate_dir = output_root / sel
            if candidate_dir.is_dir():
                manifest_path = candidate_dir / "manifest.json"
                if manifest_path.exists():
                    paths.append(manifest_path)
                    continue
        if isinstance(sel, Path) or (
            isinstance(sel, str) and sel.endswith(".json") and Path(sel).exists()
        ):
            paths.append(Path(sel))
            continue
        if isinstance(sel, int):
            if runs_df.empty:
                raise RuntimeError(f"No training runs found in {output_root}")
            if sel < 0 or sel >= len(runs_df):
                raise ValueError(
                    f"run_idx {sel} not found (available runs: 0..{max(len(runs_df) - 1, 0)})"
                )
            paths.append(Path(runs_df.iloc[sel]["path"]))
            continue
        if isinstance(sel, str) and sel.startswith("run_idx:"):
            try:
                idx = int(sel.split(":", 1)[1])
            except Exception:
                raise ValueError(f"Invalid run_idx selector '{sel}'")
            if runs_df.empty:
                raise RuntimeError(f"No training runs found in {output_root}")
            if idx < 0 or idx >= len(runs_df):
                raise ValueError(
                    f"run_idx {idx} not found (available runs: 0..{max(len(runs_df) - 1, 0)})"
                )
            paths.append(Path(runs_df.iloc[idx]["path"]))
            continue
        if isinstance(sel, str) and sel.startswith("group_idx:"):
            try:
                idx = int(sel.split(":", 1)[1])
            except Exception:
                raise ValueError(f"Invalid group_idx selector '{sel}'")
            groups = group_training_runs(output_root)
            if groups.empty or idx < 0 or idx >= len(groups):
                raise ValueError(
                    f"group_idx {idx} not found (available groups: 0..{max(len(groups) - 1, 0)})"
                )
            row = groups.iloc[idx]
            run_paths = row.get("run_paths")
            if isinstance(run_paths, list) and run_paths:
                paths.extend((Path(p) for p in run_paths))
            else:
                group_runs = row["run_ids"]
                paths.extend(_resolve_run_paths(output_root, group_runs))
            continue
        if isinstance(sel, str) and sel.startswith("group_sig:"):
            sig = sel.split(":", 1)[1]
            groups = group_training_runs(output_root)
            match = groups[groups["signature"] == sig]
            if match.empty:
                raise ValueError(f"group_sig '{sig}' not found")
            row = match.iloc[0]
            run_paths = row.get("run_paths")
            if isinstance(run_paths, list) and run_paths:
                paths.extend((Path(p) for p in run_paths))
            else:
                group_runs = row["run_ids"]
                paths.extend(_resolve_run_paths(output_root, group_runs))
            continue
        if isinstance(sel, str) and sel.lower() == "latest":
            if runs_df.empty:
                raise RuntimeError(f"No training runs found in {output_root}")
            paths.append(Path(runs_df.iloc[0]["path"]))
            continue
        if isinstance(sel, str) and sel.lower() == "all":
            if runs_df.empty:
                raise RuntimeError(f"No training runs found in {output_root}")
            paths.extend((Path(p) for p in runs_df["path"].tolist()))
            continue
        run_id_matches = runs_df[runs_df["run_id"] == sel]
        if len(run_id_matches) == 1:
            paths.append(Path(run_id_matches.iloc[0]["path"]))
            continue
        if len(run_id_matches) > 1:
            match_indices = run_id_matches["run_idx"].tolist()
            raise ValueError(
                f"Ambiguous run_id '{sel}' matched multiple runs at run_idx={match_indices}. Use run_idx:<n> (or integer run_idx) instead."
            )
        raise ValueError(f"Could not resolve run selector '{sel}'")
    if not paths:
        raise RuntimeError(f"No manifests resolved for selector {selector}")
    return paths


def _load_tiles(
    output_root: Path,
    selector: RunSelector,
    lazy: bool = False,
    limit: Optional[int] = None,
):
    manifests = _resolve_run_paths(output_root, selector)
    all_tiles = []
    first_manifest = None
    run_ids = []
    loaded_manifests = []
    for manifest_path in manifests:
        remaining = None
        if limit is not None:
            remaining = max(int(limit) - len(all_tiles), 0)
            if remaining <= 0:
                break
        if lazy:
            manifest, tiles = load_run_lazy(manifest_path, limit=remaining)
        else:
            manifest, tiles = load_run(manifest_path, limit=remaining)
        manifest = dict(manifest)
        manifest["manifest_path"] = str(manifest_path)
        for tile in tiles:
            tile["manifest_path"] = str(manifest_path)
        all_tiles.extend(tiles)
        run_ids.append(manifest.get("run_id"))
        if first_manifest is None:
            first_manifest = manifest
        loaded_manifests.append((Path(manifest_path), manifest))
        if limit is not None and len(all_tiles) >= int(limit):
            break
    if first_manifest is None:
        raise RuntimeError("No manifests loaded.")
    _validate_manifest_tensor_schemas(loaded_manifests)
    return (first_manifest, all_tiles, run_ids)


def _normalize_tensor_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": int(schema.get("version", 0)),
        "bus": list(schema.get("bus") or []),
        "rail": list(schema.get("rail") or []),
        "poi": list(schema.get("poi") or []),
        "target": list(schema.get("target") or []),
    }


def _infer_manifest_tensor_schema(manifest: Dict[str, Any]) -> Dict[str, Any]:
    schema = manifest.get("tensor_schema")
    if isinstance(schema, dict):
        normalized = _normalize_tensor_schema(schema)
        if any((normalized[key] for key in ("bus", "rail", "poi", "target"))):
            return normalized
    feature_groups = manifest.get("feature_groups") or {}
    return _normalize_tensor_schema(
        {
            "version": 0,
            "bus": manifest.get("bus_channels") or [],
            "rail": feature_groups.get("walk_transit_rail") or [],
            "poi": manifest.get("poi_layers") or [],
            "target": feature_groups.get("bus_delta") or [],
        }
    )


def _validate_manifest_tensor_schemas(
    loaded_manifests: Sequence[Tuple[Path, Dict[str, Any]]],
) -> None:
    if not loaded_manifests:
        return
    ref_path, ref_manifest = loaded_manifests[0]
    ref_schema = _infer_manifest_tensor_schema(ref_manifest)
    ref_manifest["tensor_schema"] = ref_schema
    mismatches = []
    for manifest_path, manifest in loaded_manifests[1:]:
        schema = _infer_manifest_tensor_schema(manifest)
        manifest["tensor_schema"] = schema
        for key in ("bus", "rail", "poi", "target"):
            if schema.get(key) != ref_schema.get(key):
                mismatches.append(
                    f"{manifest_path} -> {key}={schema.get(key)} | reference {ref_path} -> {key}={ref_schema.get(key)}"
                )
    if mismatches:
        detail = "\n".join(mismatches[:8])
        raise ValueError(
            f"Refusing to merge training runs with incompatible tensor schemas. Rail/POI/target/bus channel order must match exactly across manifests.\n{detail}"
        )


def _get_tile_meta(tile: Dict) -> Optional[Dict]:
    meta = tile.get("meta")
    if isinstance(meta, dict):
        return meta
    meta_path = tile.get("meta_path")
    if not meta_path:
        return None
    try:
        with Path(meta_path).open() as f:
            meta = json.load(f)
        tile["meta"] = meta
        return meta
    except Exception:
        return None


def _tile_source_array_path(tile: Dict) -> Path:
    return Path(tile["npz_path"])


def _hash_ndarray(digest: "hashlib._Hash", arr: np.ndarray) -> None:
    arr_c = np.ascontiguousarray(arr)
    digest.update(str(arr_c.dtype).encode("utf-8"))
    digest.update(str(arr_c.shape).encode("utf-8"))
    digest.update(arr_c.tobytes())


def _sample_cache_signature(
    tile: Dict,
    sample_keys: Sequence[str],
    pca_map: Dict[str, Optional[PCA]],
    norm_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    channel_trim: Dict[str, int],
    *,
    pca_apply_to_context: bool = True,
    pca_apply_to_target: bool = True,
    slim_mode: bool = False,
    slim_channel_idx: Optional[int] = None,
    slim_expand_channels: bool = False,
    target_mode: str = "delta",
) -> str:
    source_path = _tile_source_array_path(tile).resolve()
    stat = source_path.stat()
    digest = hashlib.sha1()
    digest.update(b"sample-cache-v1")
    digest.update(str(source_path).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(",".join(sorted(sample_keys)).encode("utf-8"))
    digest.update(json.dumps(channel_trim, sort_keys=True).encode("utf-8"))
    digest.update(
        json.dumps(
            {
                "pca_apply_to_context": bool(pca_apply_to_context),
                "pca_apply_to_target": bool(pca_apply_to_target),
                "slim_mode": bool(slim_mode),
                "slim_channel_idx": slim_channel_idx,
                "slim_expand_channels": bool(slim_expand_channels),
                "target_mode": str(target_mode),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    shared_pca = pca_map.get("shared")
    if shared_pca is not None:
        digest.update(b"pca:shared")
        _hash_ndarray(digest, shared_pca.components_.astype(np.float32))
        _hash_ndarray(digest, shared_pca.mean_.astype(np.float32))
    for name in sorted(norm_stats):
        mean, std = norm_stats[name]
        digest.update(name.encode("utf-8"))
        _hash_ndarray(digest, mean.astype(np.float32))
        _hash_ndarray(digest, std.astype(np.float32))
    return digest.hexdigest()[:12]


def _tile_precomputed_sample_cache_path(
    tile: Dict,
    sample_keys: Sequence[str],
    pca_map: Dict[str, Optional[PCA]],
    norm_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    channel_trim: Dict[str, int],
    cache_root: Optional[Path] = None,
    *,
    pca_apply_to_context: bool = True,
    pca_apply_to_target: bool = True,
    slim_mode: bool = False,
    slim_channel_idx: Optional[int] = None,
    slim_expand_channels: bool = False,
    target_mode: str = "delta",
) -> Path:
    source_path = _tile_source_array_path(tile).resolve()
    signature = _sample_cache_signature(
        tile,
        sample_keys,
        pca_map,
        norm_stats,
        channel_trim,
        pca_apply_to_context=pca_apply_to_context,
        pca_apply_to_target=pca_apply_to_target,
        slim_mode=slim_mode,
        slim_channel_idx=slim_channel_idx,
        slim_expand_channels=slim_expand_channels,
        target_mode=target_mode,
    )
    if cache_root is None:
        base_dir = source_path.parent.parent / "tile_sample_cache"
    else:
        base_dir = Path(cache_root).expanduser().resolve()
    return (
        base_dir
        / source_path.parent.parent.name
        / f"{source_path.stem}.{signature}.npz"
    )


def _materialize_tile_sample(
    tile: Dict,
    *,
    lazy: bool,
    sample_keys: Sequence[str],
    key_map: Dict[str, Optional[str]],
    pca_map: Dict[str, Optional[PCA]],
    norm_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    channel_trim: Dict[str, int],
    pca_apply_to_context: bool = True,
    pca_apply_to_target: bool = True,
    slim_mode: bool = False,
    slim_channel_idx: Optional[int] = None,
    slim_expand_channels: bool = False,
    return_components: bool = False,
) -> Dict[str, np.ndarray]:
    arrays = _get_tile_arrays(tile, lazy=lazy, keys=sample_keys)
    bus = arrays[key_map["bus"]].astype(np.float32)
    rail = arrays[key_map["rail"]].astype(np.float32)
    poi = arrays[key_map["poi"]].astype(np.float32)
    target = arrays[key_map["target"]].astype(np.float32)
    mask_key = key_map.get("mask")
    mask = arrays.get(mask_key) if mask_key else None
    if channel_trim:
        if channel_trim.get("bus") and bus.shape[0] > channel_trim["bus"]:
            bus = bus[: channel_trim["bus"]]
        if channel_trim.get("rail") and rail.shape[0] > channel_trim["rail"]:
            rail = rail[: channel_trim["rail"]]
        if channel_trim.get("poi") and poi.shape[0] > channel_trim["poi"]:
            poi = poi[: channel_trim["poi"]]
        if channel_trim.get("target") and target.shape[0] > channel_trim["target"]:
            target = target[: channel_trim["target"]]
        if (
            mask is not None
            and mask.ndim == 3
            and channel_trim.get("target")
            and (mask.shape[0] > channel_trim["target"])
        ):
            mask = mask[: channel_trim["target"]]
    shared_pca = pca_map.get("shared")
    context_pca = shared_pca if pca_apply_to_context else None
    target_pca = shared_pca if pca_apply_to_target else None
    rail = _apply_pca(rail, context_pca)
    poi = _apply_pca(poi, context_pca)
    target = _apply_pca(target, target_pca)
    eps = 1e-06
    if "bus" in norm_stats:
        mean, std = norm_stats["bus"]
        bus = (bus - mean[:, None, None]) / (std[:, None, None] + eps)
    if "rail" in norm_stats:
        mean, std = norm_stats["rail"]
        rail = (rail - mean[:, None, None]) / (std[:, None, None] + eps)
    if "poi" in norm_stats:
        mean, std = norm_stats["poi"]
        poi = (poi - mean[:, None, None]) / (std[:, None, None] + eps)
    if "target" in norm_stats:
        mean, std = norm_stats["target"]
        target = (target - mean[:, None, None]) / (std[:, None, None] + eps)
    if return_components:
        payload = {
            "bus": bus.astype(np.float32, copy=False),
            "rail": rail.astype(np.float32, copy=False),
            "poi": poi.astype(np.float32, copy=False),
            "target": target.astype(np.float32, copy=False),
        }
        if mask is not None:
            mask_arr = mask.astype(np.float32, copy=False)
            if mask_arr.ndim == 2:
                mask_arr = mask_arr[None, ...]
            payload["mask"] = mask_arr
        return payload
    if slim_mode:
        if slim_channel_idx is None:
            if slim_expand_channels:
                raise ValueError(
                    "Expanded slim samples must be sliced by TileDataset.__getitem__; use return_components=True."
                )
            raise ValueError(
                "slim_channel_idx is required when slim_mode=True and slim_expand_channels=False."
            )
        max_common = min(rail.shape[0], poi.shape[0], target.shape[0])
        if slim_channel_idx < 0 or slim_channel_idx >= max_common:
            raise IndexError(
                f"slim_channel_idx={slim_channel_idx} out of range for common channel count {max_common}"
            )
        inputs = np.concatenate(
            [
                bus,
                rail[slim_channel_idx : slim_channel_idx + 1],
                poi[slim_channel_idx : slim_channel_idx + 1],
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        target = target[slim_channel_idx : slim_channel_idx + 1]
        if mask is not None and mask.ndim == 3 and (mask.shape[0] > 1):
            mask = mask[slim_channel_idx : slim_channel_idx + 1]
    else:
        inputs = np.concatenate([bus, rail, poi], axis=0).astype(np.float32, copy=False)
    payload = {"inputs": inputs, "target": target.astype(np.float32, copy=False)}
    if mask is not None:
        mask_arr = mask.astype(np.float32, copy=False)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[None, ...]
        payload["mask"] = mask_arr
    return payload


def _ensure_precomputed_sample_cache(
    tile: Dict,
    *,
    lazy: bool,
    sample_keys: Sequence[str],
    key_map: Dict[str, Optional[str]],
    pca_map: Dict[str, Optional[PCA]],
    norm_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    channel_trim: Dict[str, int],
    cache_root: Optional[Path] = None,
    force: bool = False,
    pca_apply_to_context: bool = True,
    pca_apply_to_target: bool = True,
    slim_mode: bool = False,
    slim_channel_idx: Optional[int] = None,
    slim_expand_channels: bool = False,
) -> Path:
    dest = _tile_precomputed_sample_cache_path(
        tile,
        sample_keys,
        pca_map,
        norm_stats,
        channel_trim,
        cache_root=cache_root,
        pca_apply_to_context=pca_apply_to_context,
        pca_apply_to_target=pca_apply_to_target,
        slim_mode=slim_mode,
        slim_channel_idx=slim_channel_idx,
        slim_expand_channels=slim_expand_channels,
        target_mode=str(key_map.get("target_mode") or "delta"),
    )
    if dest.exists() and (not force):
        tile["sample_cache_path"] = str(dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = _materialize_tile_sample(
        tile,
        lazy=lazy,
        sample_keys=sample_keys,
        key_map=key_map,
        pca_map=pca_map,
        norm_stats=norm_stats,
        channel_trim=channel_trim,
        pca_apply_to_context=pca_apply_to_context,
        pca_apply_to_target=pca_apply_to_target,
        slim_mode=slim_mode,
        slim_channel_idx=slim_channel_idx,
        slim_expand_channels=slim_expand_channels,
        return_components=bool(slim_mode and slim_expand_channels),
    )
    tmp = dest.with_suffix(
        f".tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.npz"
    )
    try:
        np.savez(tmp, **payload)
        if force or not dest.exists():
            tmp.replace(dest)
        else:
            tmp.unlink(missing_ok=True)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    tile["sample_cache_path"] = str(dest)
    return dest


def _prepare_precomputed_sample_cache(
    tiles: Sequence[Dict],
    *,
    enabled: bool,
    lazy: bool,
    sample_keys: Sequence[str],
    key_map: Dict[str, Optional[str]],
    pca_map: Dict[str, Optional[PCA]],
    norm_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    channel_trim: Dict[str, int],
    cache_root: Optional[Path],
    force: bool,
    workers: int,
    pca_apply_to_context: bool = True,
    pca_apply_to_target: bool = True,
    slim_mode: bool = False,
    slim_channel_idx: Optional[int] = None,
    slim_expand_channels: bool = False,
) -> None:
    if not enabled or not tiles:
        return
    if workers > 1:
        worker_args = [
            (
                tile,
                lazy,
                tuple(sample_keys),
                key_map,
                pca_map,
                norm_stats,
                channel_trim,
                cache_root,
                force,
                pca_apply_to_context,
                pca_apply_to_target,
                slim_mode,
                slim_channel_idx,
                slim_expand_channels,
            )
            for tile in tiles
        ]
        chunksize = max(1, min(32, len(worker_args) // max(workers * 4, 1)))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for idx, cache_path in enumerate(
                ex.map(
                    _precompute_sample_cache_worker, worker_args, chunksize=chunksize
                ),
                1,
            ):
                tiles[idx - 1]["sample_cache_path"] = str(cache_path)
                if (
                    idx == 1
                    or idx == len(tiles)
                    or idx % max(1, min(50, len(tiles) // 10 or 1)) == 0
                ):
                    pass
    else:
        for idx, tile in enumerate(tiles, 1):
            cache_path = _ensure_precomputed_sample_cache(
                tile,
                lazy=lazy,
                sample_keys=sample_keys,
                key_map=key_map,
                pca_map=pca_map,
                norm_stats=norm_stats,
                channel_trim=channel_trim,
                cache_root=cache_root,
                force=force,
                pca_apply_to_context=pca_apply_to_context,
                pca_apply_to_target=pca_apply_to_target,
                slim_mode=slim_mode,
                slim_channel_idx=slim_channel_idx,
                slim_expand_channels=slim_expand_channels,
            )
            if (
                idx == 1
                or idx == len(tiles)
                or idx % max(1, min(50, len(tiles) // 10 or 1)) == 0
            ):
                pass


def _precompute_sample_cache_worker(args) -> Path:
    (
        tile,
        lazy,
        sample_keys,
        key_map,
        pca_map,
        norm_stats,
        channel_trim,
        cache_root,
        force,
        pca_apply_to_context,
        pca_apply_to_target,
        slim_mode,
        slim_channel_idx,
        slim_expand_channels,
    ) = args
    return _ensure_precomputed_sample_cache(
        tile,
        lazy=lazy,
        sample_keys=sample_keys,
        key_map=key_map,
        pca_map=pca_map,
        norm_stats=norm_stats,
        channel_trim=channel_trim,
        cache_root=cache_root,
        force=force,
        pca_apply_to_context=pca_apply_to_context,
        pca_apply_to_target=pca_apply_to_target,
        slim_mode=slim_mode,
        slim_channel_idx=slim_channel_idx,
        slim_expand_channels=slim_expand_channels,
    )


def _preload_precomputed_samples(
    tiles: Sequence[Dict], *, enabled: bool, max_gb: Optional[float]
) -> int:
    if not enabled or not tiles:
        return 0
    sample_tiles = [tile for tile in tiles if tile.get("sample_cache_path")]
    if not sample_tiles:
        warnings.warn(
            "Precomputed sample preload requested but no sample cache paths are available.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0
    byte_budget = None if max_gb is None else max(0, int(float(max_gb) * 1024**3))
    loaded_bytes = 0
    loaded_tiles = 0
    for idx, tile in enumerate(sample_tiles, 1):
        path = Path(tile["sample_cache_path"])
        size_bytes = path.stat().st_size
        if (
            byte_budget is not None
            and loaded_tiles > 0
            and (loaded_bytes + size_bytes > byte_budget)
        ):
            break
        tile["sample_cache_arrays"] = load_tile_arrays(
            path, keys=["inputs", "bus", "rail", "poi", "target", "mask"]
        )
        loaded_bytes += size_bytes
        loaded_tiles += 1
    return loaded_tiles


def _preload_tiles_to_ram(
    tiles: Sequence[Dict],
    *,
    enabled: bool,
    lazy: bool,
    sample_keys: Sequence[str],
    key_map: Dict[str, Optional[str]],
    pca_map: Dict,
    norm_stats: Dict,
    channel_trim: Dict[str, int],
    workers: int,
    pca_apply_to_context: bool = True,
    pca_apply_to_target: bool = True,
    slim_mode: bool = False,
    slim_channel_idx: Optional[int] = None,
    slim_expand_channels: bool = False,
) -> None:
    """Materialize every tile (PCA projection + normalization) and store in tile["ram_sample"]."""
    if not enabled or not tiles:
        return

    def _load_one(tile: Dict) -> Dict[str, np.ndarray]:
        return _materialize_tile_sample(
            tile,
            lazy=lazy,
            sample_keys=sample_keys,
            key_map=key_map,
            pca_map=pca_map,
            norm_stats=norm_stats,
            channel_trim=channel_trim,
            pca_apply_to_context=pca_apply_to_context,
            pca_apply_to_target=pca_apply_to_target,
            slim_mode=slim_mode,
            slim_channel_idx=slim_channel_idx,
            slim_expand_channels=slim_expand_channels,
            return_components=bool(slim_mode and slim_expand_channels),
        )

    effective_workers = max(1, workers)
    if effective_workers > 1:
        with ThreadPoolExecutor(max_workers=effective_workers) as ex:
            for idx, (tile, payload) in enumerate(
                zip(tiles, ex.map(_load_one, tiles)), 1
            ):
                tile["ram_sample"] = payload
    else:
        for idx, tile in enumerate(tiles, 1):
            tile["ram_sample"] = _load_one(tile)


def _get_tile_arrays(
    tile: Dict, lazy: bool = False, keys: Optional[Sequence[str]] = None
) -> Dict:
    """Return arrays for a tile, loading from disk if lazy is enabled."""
    if not lazy or "arrays" in tile:
        if keys is None:
            return tile["arrays"]
        return {key: tile["arrays"][key] for key in keys if key in tile["arrays"]}
    return load_tile_arrays(_tile_source_array_path(tile), keys=keys)


def _tile_grid_shape(
    tile: Dict, lazy: bool = False, target_mode: str = "delta"
) -> Optional[Tuple[int, int]]:
    meta = _get_tile_meta(tile)
    if meta:
        grid = meta.get("grid")
        if not grid:
            for key in (
                "meta_bus_delta",
                "meta_walk_transit_rail",
                "meta_poi",
                "meta_bus",
            ):
                entry = meta.get(key) or {}
                if isinstance(entry, dict) and entry.get("grid"):
                    grid = entry.get("grid")
                    break
        if grid:
            h = grid.get("height")
            w = grid.get("width")
            if h is not None and w is not None:
                return (int(h), int(w))
    source_names = (
        list_tile_array_keys(_tile_source_array_path(tile))
        if lazy
        else list(tile["arrays"].keys())
    )
    key = _pick_target_name(source_names, target_mode)
    if key is None:
        return None
    arr = _get_tile_arrays(tile, lazy=lazy, keys=[key])
    h, w = arr[key].shape[1:]
    return (h, w)


def _tile_meta_counts(tile: Dict) -> Optional[Dict[str, int]]:
    meta = _get_tile_meta(tile)
    if not meta:
        return None

    def _bands(entry: Optional[Dict]) -> Optional[int]:
        if not entry:
            return None
        arr = entry.get("array") if isinstance(entry, dict) else None
        if not isinstance(arr, dict):
            return None
        bands = arr.get("bands")
        return int(bands) if bands is not None else None

    bus = _bands(meta.get("meta_bus"))
    rail = _bands(meta.get("meta_walk_transit_rail"))
    target = _bands(meta.get("meta_bus_delta"))
    poi_entry = meta.get("meta_poi")
    poi = None
    if isinstance(poi_entry, dict):
        layers = poi_entry.get("layers")
        if layers is not None:
            poi = len(layers)
        else:
            poi = _bands(poi_entry)
    if bus is None or rail is None or poi is None or (target is None):
        return None
    return {"bus": bus, "rail": rail, "poi": poi, "target": target}


def _tile_array_counts(
    tile: Dict, lazy: bool = False, target_mode: str = "delta"
) -> Dict[str, int]:
    source_names = (
        list_tile_array_keys(_tile_source_array_path(tile))
        if lazy
        else list(tile["arrays"].keys())
    )
    bus_key = _pick_name(source_names, ["bus_tensor"])
    rail_key = _pick_name(
        source_names,
        ["access_walk_transit_rail", "walk_transit_rail"],
        fallback_prefix="access_",
    )
    poi_key = _pick_name(source_names, ["poi_tensor"])
    target_key = _pick_target_name(source_names, target_mode)
    if bus_key is None or rail_key is None or poi_key is None or (target_key is None):
        raise KeyError("Tile missing required tensors for channel alignment.")
    arrays = _get_tile_arrays(
        tile, lazy=lazy, keys=[bus_key, rail_key, poi_key, target_key]
    )
    return {
        "bus": arrays[bus_key].shape[0],
        "rail": arrays[rail_key].shape[0],
        "poi": arrays[poi_key].shape[0],
        "target": arrays[target_key].shape[0],
    }


def _fit_shared_pca(
    tiles: List[Dict],
    keys: Sequence[str],
    n_components: int,
    max_samples: int,
    random_state: int,
    lazy: bool = False,
    channel_overrides: Optional[Dict[str, int]] = None,
    workers: int = 0,
    progress_label: Optional[str] = None,
) -> PCA:
    if PCA is None:
        raise ImportError(
            "scikit-learn is required for PCA. Please install scikit-learn or disable PCA."
        )
    rng = np.random.default_rng(random_state)
    samples = []
    total_tiles = len(tiles)
    shared_override = None
    if channel_overrides:
        shared_override = min(channel_overrides.values())
    stream_count = max(1, len(keys))
    per_stream_tile_budget = None
    if max_samples and total_tiles > 0:
        per_stream_tile_budget = max(
            1, int(math.ceil(max_samples / (total_tiles * stream_count)))
        )
    if workers > 1:

        def _load_arrays(tile: Dict) -> Dict[str, Optional[np.ndarray]]:
            arrays = _get_tile_arrays(tile, lazy=lazy, keys=keys)
            return {key: arrays.get(key) for key in keys}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for idx, arrays in enumerate(ex.map(_load_arrays, tiles), 1):
                for key in keys:
                    arr = arrays.get(key)
                    if arr is None:
                        continue
                    target_ch = None
                    if channel_overrides:
                        target_ch = channel_overrides.get(key, shared_override)
                    if target_ch is not None and arr.shape[0] > target_ch:
                        arr = arr[:target_ch]
                    flat = arr.reshape(arr.shape[0], -1).T
                    if target_ch is not None and flat.shape[1] > target_ch:
                        flat = flat[:, :target_ch]
                    if (
                        per_stream_tile_budget
                        and flat.shape[0] > per_stream_tile_budget
                    ):
                        sample_idx = rng.choice(
                            flat.shape[0], size=per_stream_tile_budget, replace=False
                        )
                        flat = flat[sample_idx]
                    samples.append(flat.astype(np.float32))
    else:
        for idx, tile in enumerate(tiles, 1):
            arrays = _get_tile_arrays(tile, lazy=lazy, keys=keys)
            for key in keys:
                arr = arrays.get(key)
                if arr is None:
                    continue
                target_ch = None
                if channel_overrides:
                    target_ch = channel_overrides.get(key, shared_override)
                if target_ch is not None and arr.shape[0] > target_ch:
                    arr = arr[:target_ch]
                flat = arr.reshape(arr.shape[0], -1).T
                if target_ch is not None and flat.shape[1] > target_ch:
                    flat = flat[:, :target_ch]
                if per_stream_tile_budget and flat.shape[0] > per_stream_tile_budget:
                    sample_idx = rng.choice(
                        flat.shape[0], size=per_stream_tile_budget, replace=False
                    )
                    flat = flat[sample_idx]
                samples.append(flat.astype(np.float32))
    if not samples:
        raise ValueError(f"No arrays found for keys {keys} to fit PCA.")
    data = np.concatenate(samples, axis=0)
    if max_samples and data.shape[0] > max_samples:
        sample_idx = rng.choice(data.shape[0], size=max_samples, replace=False)
        data = data[sample_idx]
    n_comp = min(n_components, data.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    pca.fit(data)
    return pca


def _apply_pca(arr: np.ndarray, pca: Optional[PCA]) -> np.ndarray:
    if pca is None:
        return arr
    shp = arr.shape
    flat = arr.reshape(shp[0], -1).T
    transformed = pca.transform(flat).T
    new_ch = transformed.shape[0]
    return transformed.reshape(new_ch, shp[1], shp[2]).astype(np.float32)


def _compute_norm_stats(
    tiles: List[Dict],
    key: str,
    pca: Optional[PCA] = None,
    lazy: bool = False,
    target_channels: Optional[int] = None,
    workers: int = 0,
    progress_label: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std for a tensor key (optionally after PCA)."""
    sum_ch = None
    sumsq_ch = None
    count = 0

    def _stats_for_tile(tile: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
        arr = _get_tile_arrays(tile, lazy=lazy, keys=[key])[key].astype(np.float64)
        if target_channels is not None and arr.shape[0] > target_channels:
            arr = arr[:target_channels]
        if pca is not None:
            arr = _apply_pca(arr.astype(np.float32), pca).astype(np.float64)
        flat = arr.reshape(arr.shape[0], -1)
        return (flat.sum(axis=1), (flat**2).sum(axis=1), flat.shape[1])

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for idx, (sum_tile, sumsq_tile, count_tile) in enumerate(
                ex.map(_stats_for_tile, tiles), 1
            ):
                if sum_ch is None:
                    sum_ch = np.zeros(sum_tile.shape[0], dtype=np.float64)
                    sumsq_ch = np.zeros(sum_tile.shape[0], dtype=np.float64)
                sum_ch += sum_tile
                sumsq_ch += sumsq_tile
                count += count_tile
    else:
        for idx, tile in enumerate(tiles, 1):
            sum_tile, sumsq_tile, count_tile = _stats_for_tile(tile)
            if sum_ch is None:
                sum_ch = np.zeros(sum_tile.shape[0], dtype=np.float64)
                sumsq_ch = np.zeros(sum_tile.shape[0], dtype=np.float64)
            sum_ch += sum_tile
            sumsq_ch += sumsq_tile
            count += count_tile
    if sum_ch is None or count == 0:
        raise ValueError(f"No data to compute norm stats for key {key}")
    mean = sum_ch / count
    var = sumsq_ch / count - mean**2
    std = np.sqrt(np.maximum(var, 1e-12))
    return (mean.astype(np.float32), std.astype(np.float32))


def _apply_output_activation(x: torch.Tensor, name: Optional[str]) -> torch.Tensor:
    if name is None:
        return x
    n = name.lower()
    if n == "sigmoid":
        return torch.sigmoid(x)
    if n == "tanh":
        return torch.tanh(x)
    if n == "clamp0_1":
        return torch.clamp(x, 0.0, 1.0)
    raise ValueError(f"Unsupported output_activation '{name}'")


def _forward_pred(model, inputs, uncertainty_on: bool):
    """Run the model, returning (mu, logvar). logvar is None unless the uncertainty head is"""
    if uncertainty_on:
        mu, logvar = model(inputs, return_uncertainty=True)
        return (mu, logvar)
    return (model(inputs), None)


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _extract_pca_state(pca: Optional[PCA]) -> Optional[Dict[str, Any]]:
    if pca is None:
        return None
    state = {
        "components": pca.components_.astype(np.float32),
        "mean": pca.mean_.astype(np.float32),
        "n_components": int(getattr(pca, "n_components_", pca.components_.shape[0])),
        "n_features_in": int(getattr(pca, "n_features_in_", pca.components_.shape[1])),
    }
    if hasattr(pca, "explained_variance_ratio_"):
        state["explained_variance_ratio"] = pca.explained_variance_ratio_.astype(
            np.float32
        )
    if hasattr(pca, "explained_variance_"):
        state["explained_variance"] = pca.explained_variance_.astype(np.float32)
    return state


def _align_and_trim_channels(
    tiles: List[Dict], lazy: bool = False, target_mode: str = "delta"
) -> Dict[str, int]:
    """Ensure consistent tensor keys and channel counts across tiles."""
    if not tiles:
        raise ValueError("No tiles available for alignment check.")
    counts_by_run: Dict[str, Dict[str, int]] = {}
    min_counts: Dict[str, int] = {
        "bus": math.inf,
        "rail": math.inf,
        "poi": math.inf,
        "target": math.inf,
    }
    for tile in tiles:
        run_id = tile.get("run_id", "<unknown>")
        counts = (
            _tile_meta_counts(tile)
            if lazy and str(target_mode).lower() == "delta"
            else None
        )
        if counts is None:
            counts = _tile_array_counts(tile, lazy=lazy, target_mode=target_mode)
        counts_by_run[run_id] = counts
        for name, ch in counts.items():
            min_counts[name] = min(min_counts[name], ch)
    for name in ("bus", "rail", "poi", "target"):
        per_run = {rid: c[name] for rid, c in counts_by_run.items()}
        unique_counts = set(per_run.values())
        if len(unique_counts) > 1:
            warnings.warn(
                "Channel mismatch for %s across runs; trimming to min=%s | per-run=%s"
                % (name, min_counts[name], per_run),
                RuntimeWarning,
                stacklevel=2,
            )
        if min_counts[name] == math.inf or min_counts[name] <= 0:
            raise ValueError(f"Invalid channel count for {name}: {min_counts[name]}")
    if not lazy:
        ref = _get_tile_arrays(tiles[0], lazy=lazy)
        ref_run = tiles[0].get("run_id", "<unknown>")
        ref_keys = {
            "bus": _pick_key(ref, ["bus_tensor"]),
            "rail": _pick_key(
                ref,
                ["access_walk_transit_rail", "walk_transit_rail"],
                fallback_prefix="access_",
            ),
            "poi": _pick_key(ref, ["poi_tensor"]),
            "target": _pick_target_key(ref, target_mode),
            "mask": _pick_target_mask_key(ref, target_mode),
        }
        for name, key in ref_keys.items():
            if key is None and name != "mask":
                raise KeyError(
                    f"Reference tile missing required tensor for {name} (run={ref_run})"
                )
        for tile in tiles:
            arrays = tile["arrays"]
            for name, key in ref_keys.items():
                if name == "mask":
                    continue
                cur_key = _pick_key(
                    arrays,
                    [key] if key else [],
                    fallback_prefix="access_" if name == "rail" else None,
                )
                arr = arrays[cur_key]
                target_ch = min_counts[name] if name in min_counts else arr.shape[0]
                if arr.shape[0] > target_ch:
                    arrays[cur_key] = arr[:target_ch]
            if ref_keys["mask"]:
                m_key = _pick_key(arrays, [ref_keys["mask"]], fallback_prefix="mask_")
                if m_key:
                    mask_arr = arrays[m_key]
                    if mask_arr.ndim == 3 and mask_arr.shape[0] > min_counts["target"]:
                        arrays[m_key] = mask_arr[: min_counts["target"]]
    return min_counts


class TileDataset(Dataset):

    def __init__(
        self,
        tiles: List[Dict],
        pca_map: Dict[str, Optional[PCA]],
        norm_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        limit: Optional[int] = None,
        lazy: bool = False,
        channel_trim: Optional[Dict[str, int]] = None,
        use_precomputed_samples: bool = False,
        pca_apply_to_context: bool = True,
        pca_apply_to_target: bool = True,
        target_mode: str = "delta",
        slim_mode: bool = False,
        slim_channel_idx: Optional[int] = None,
        slim_expand_channels: bool = False,
        augment_flip: bool = False,
        augment_rot90: bool = False,
        augment_bus_activation_scale: bool = False,
        augment_bus_activation_scale_prob: float = 1.0,
        augment_bus_activation_scale_min: float = 0.05,
        augment_bus_activation_scale_max: float = 1.0,
        augment_bus_activation_scale_distribution: str = "log_uniform",
        augment_bus_activation_scale_keep_full_prob: float = 0.25,
        augment_bus_activation_target_power: float = 1.0,
        augment_isolated_stop: bool = False,
        augment_isolated_stop_prob: float = 0.15,
        augment_stub_edge: bool = False,
        augment_stub_edge_prob: float = 0.15,
        augment_stub_edge_min_len_px: int = 5,
        augment_stub_edge_max_len_px: int = 30,
        augment_orphan_stop_overlay: bool = False,
        augment_orphan_stop_overlay_prob: float = 0.15,
        augment_orphan_stop_overlay_min_shift_px: int = 12,
        augment_orphan_stop_overlay_max_shift_px: int = 80,
        augment_orphan_stop_overlay_edge_dilate_px: int = 5,
        augment_orphan_stop_overlay_scale_min: float = 0.5,
        augment_orphan_stop_overlay_scale_max: float = 1.0,
    ):
        self.tiles = tiles[:limit] if limit else tiles
        self.pca_map = pca_map
        self.norm_stats = norm_stats or {}
        self.lazy = lazy
        self.channel_trim = channel_trim or {}
        self.use_precomputed_samples = bool(use_precomputed_samples)
        self.pca_apply_to_context = bool(pca_apply_to_context)
        self.pca_apply_to_target = bool(pca_apply_to_target)
        self.target_mode = str(target_mode).lower()
        self.slim_mode = bool(slim_mode)
        self.slim_channel_idx = slim_channel_idx
        self.slim_expand_channels = bool(slim_expand_channels)
        self.slim_channels = 1
        self.augment_flip = augment_flip
        self.augment_rot90 = augment_rot90
        self.augment_bus_activation_scale = bool(augment_bus_activation_scale)
        self.augment_bus_activation_scale_prob = float(
            augment_bus_activation_scale_prob
        )
        self.augment_bus_activation_scale_min = float(augment_bus_activation_scale_min)
        self.augment_bus_activation_scale_max = float(augment_bus_activation_scale_max)
        self.augment_bus_activation_scale_distribution = str(
            augment_bus_activation_scale_distribution
        ).lower()
        self.augment_bus_activation_scale_keep_full_prob = float(
            augment_bus_activation_scale_keep_full_prob
        )
        self.augment_bus_activation_target_power = float(
            augment_bus_activation_target_power
        )
        self.augment_isolated_stop = bool(augment_isolated_stop)
        self.augment_isolated_stop_prob = float(augment_isolated_stop_prob)
        self.augment_stub_edge = bool(augment_stub_edge)
        self.augment_stub_edge_prob = float(augment_stub_edge_prob)
        self.augment_stub_edge_min_len_px = int(augment_stub_edge_min_len_px)
        self.augment_stub_edge_max_len_px = int(augment_stub_edge_max_len_px)
        self.augment_orphan_stop_overlay = bool(augment_orphan_stop_overlay)
        self.augment_orphan_stop_overlay_prob = float(augment_orphan_stop_overlay_prob)
        self.augment_orphan_stop_overlay_min_shift_px = int(
            augment_orphan_stop_overlay_min_shift_px
        )
        self.augment_orphan_stop_overlay_max_shift_px = int(
            augment_orphan_stop_overlay_max_shift_px
        )
        self.augment_orphan_stop_overlay_edge_dilate_px = int(
            augment_orphan_stop_overlay_edge_dilate_px
        )
        self.augment_orphan_stop_overlay_scale_min = float(
            augment_orphan_stop_overlay_scale_min
        )
        self.augment_orphan_stop_overlay_scale_max = float(
            augment_orphan_stop_overlay_scale_max
        )
        self._target_pca_mean_projection: Optional[torch.Tensor] = None
        shared_pca = pca_map.get("shared")
        if shared_pca is not None and self.pca_apply_to_target:
            self._target_pca_mean_projection = torch.from_numpy(
                np.asarray(shared_pca.components_, dtype=np.float32)
                @ np.asarray(shared_pca.mean_, dtype=np.float32)
            )
        if self.augment_bus_activation_scale:
            if not 0.0 <= self.augment_bus_activation_scale_prob <= 1.0:
                raise ValueError("augment_bus_activation_scale_prob must be in [0, 1]")
            if not 0.0 <= self.augment_bus_activation_scale_keep_full_prob <= 1.0:
                raise ValueError(
                    "augment_bus_activation_scale_keep_full_prob must be in [0, 1]"
                )
            if (
                self.augment_bus_activation_scale_min <= 0
                or self.augment_bus_activation_scale_max <= 0
            ):
                raise ValueError("bus activation scale bounds must be positive")
            if (
                self.augment_bus_activation_scale_min
                > self.augment_bus_activation_scale_max
            ):
                raise ValueError("augment_bus_activation_scale_min must be <= max")
            if self.augment_bus_activation_scale_distribution not in {
                "log_uniform",
                "uniform",
            }:
                raise ValueError(
                    "augment_bus_activation_scale_distribution must be 'log_uniform' or 'uniform'"
                )
        if self.augment_orphan_stop_overlay:
            if not 0.0 <= self.augment_orphan_stop_overlay_prob <= 1.0:
                raise ValueError("augment_orphan_stop_overlay_prob must be in [0, 1]")
            if self.augment_orphan_stop_overlay_min_shift_px < 0:
                raise ValueError(
                    "augment_orphan_stop_overlay_min_shift_px must be non-negative"
                )
            if (
                self.augment_orphan_stop_overlay_max_shift_px
                < self.augment_orphan_stop_overlay_min_shift_px
            ):
                raise ValueError(
                    "augment_orphan_stop_overlay_max_shift_px must be >= min_shift_px"
                )
            if self.augment_orphan_stop_overlay_edge_dilate_px < 0:
                raise ValueError(
                    "augment_orphan_stop_overlay_edge_dilate_px must be non-negative"
                )
            if (
                self.augment_orphan_stop_overlay_scale_min < 0
                or self.augment_orphan_stop_overlay_scale_max < 0
            ):
                raise ValueError("orphan stop overlay scales must be non-negative")
            if (
                self.augment_orphan_stop_overlay_scale_min
                > self.augment_orphan_stop_overlay_scale_max
            ):
                raise ValueError(
                    "augment_orphan_stop_overlay_scale_min must be <= scale_max"
                )
        if self.augment_stub_edge:
            if not 0.0 <= self.augment_stub_edge_prob <= 1.0:
                raise ValueError("augment_stub_edge_prob must be in [0, 1]")
            if self.augment_stub_edge_min_len_px < 1:
                raise ValueError("augment_stub_edge_min_len_px must be >= 1")
            if self.augment_stub_edge_max_len_px < self.augment_stub_edge_min_len_px:
                raise ValueError("augment_stub_edge_max_len_px must be >= min_len_px")
        if not self.tiles:
            raise ValueError("Dataset received no tiles.")
        if self.target_mode not in {"delta", "total"}:
            raise ValueError("target_mode must be 'delta' or 'total'")
        if self.slim_mode:
            if self.slim_channel_idx is not None and int(self.slim_channel_idx) < 0:
                raise ValueError("slim_channel_idx must be non-negative")
            if self.slim_channel_idx is None and (not self.slim_expand_channels):
                raise ValueError(
                    "slim_channel_idx is required when slim_mode=True unless slim_expand_channels=True"
                )
        if (
            self.use_precomputed_samples
            and self.tiles[0].get("sample_cache_arrays") is not None
        ):
            cached0 = self.tiles[0]["sample_cache_arrays"]
            self.bus_key = self.rail_key = self.poi_key = self.target_key = (
                self.mask_key
            ) = None
            self.height = cached0["target"].shape[1]
            self.width = cached0["target"].shape[2]
            source_inputs = cached0.get("inputs")
            if self.slim_mode and source_inputs is None:
                base_bus = cached0["bus"].shape[0]
                rail_ch = cached0["rail"].shape[0]
                poi_ch = cached0["poi"].shape[0]
                target_ch = cached0["target"].shape[0]
                self.slim_channels = min(rail_ch, poi_ch, target_ch)
                if self.slim_channels <= 0:
                    raise ValueError(
                        "Slim precomputed sample cache has no common channels."
                    )
                if (
                    self.slim_channel_idx is not None
                    and int(self.slim_channel_idx) >= self.slim_channels
                ):
                    raise ValueError(
                        f"slim_channel_idx={self.slim_channel_idx} out of range for {self.slim_channels} channel(s)"
                    )
                self.target_channels = 1
                self.bus_channels = int(base_bus)
                self.rail_channels = 1
                self.poi_channels = 1
                return
            self.target_channels = cached0["target"].shape[0]
            self.bus_channels = None
            self.rail_channels = None
            self.poi_channels = None
            source_inputs = cached0["inputs"]
            pca_shared = pca_map.get("shared")
            base_bus = self.channel_trim.get("bus")
            base_rail = self.channel_trim.get("rail")
            base_poi = self.channel_trim.get("poi")
            if base_bus is None or base_rail is None or base_poi is None:
                raise ValueError(
                    "Channel trim must be resolved before using precomputed sample cache."
                )
            self.bus_channels = int(base_bus)
            self.rail_channels = (
                int(pca_shared.n_components_) if pca_shared else int(base_rail)
            )
            self.poi_channels = (
                source_inputs.shape[0] - self.bus_channels - self.rail_channels
            )
            if self.poi_channels < 0:
                raise ValueError("Precomputed sample cache channel layout is invalid.")
            return
        source_names = (
            list_tile_array_keys(_tile_source_array_path(self.tiles[0]))
            if self.lazy
            else list(self.tiles[0]["arrays"].keys())
        )
        self.bus_key = _pick_name(source_names, ["bus_tensor"])
        self.rail_key = _pick_name(
            source_names,
            ["access_walk_transit_rail", "walk_transit_rail"],
            fallback_prefix="access_",
        )
        self.poi_key = _pick_name(source_names, ["poi_tensor"])
        self.target_key = _pick_target_name(source_names, self.target_mode)
        self.mask_key = _pick_target_mask_name(source_names, self.target_mode)
        if (
            self.bus_key is None
            or self.rail_key is None
            or self.poi_key is None
            or (self.target_key is None)
        ):
            raise KeyError(
                f"Missing required tensors (bus={self.bus_key}, rail={self.rail_key}, poi={self.poi_key}, target={self.target_key})"
            )
        init_keys = [self.bus_key, self.rail_key, self.poi_key, self.target_key]
        if self.mask_key:
            init_keys.append(self.mask_key)
        arrays0 = _get_tile_arrays(self.tiles[0], lazy=self.lazy, keys=init_keys)
        target_ch = self.channel_trim.get("target", arrays0[self.target_key].shape[0])
        self.height = arrays0[self.target_key].shape[1]
        self.width = arrays0[self.target_key].shape[2]
        self.bus_channels = min(
            arrays0[self.bus_key].shape[0],
            self.channel_trim.get("bus", arrays0[self.bus_key].shape[0]),
        )
        pca_shared = pca_map.get("shared")
        rail_base = min(
            arrays0[self.rail_key].shape[0],
            self.channel_trim.get("rail", arrays0[self.rail_key].shape[0]),
        )
        poi_base = min(
            arrays0[self.poi_key].shape[0],
            self.channel_trim.get("poi", arrays0[self.poi_key].shape[0]),
        )
        target_base = min(arrays0[self.target_key].shape[0], target_ch)
        self.rail_channels = (
            pca_shared.n_components_
            if pca_shared and self.pca_apply_to_context
            else rail_base
        )
        self.poi_channels = (
            pca_shared.n_components_
            if pca_shared and self.pca_apply_to_context
            else poi_base
        )
        self.target_channels = (
            pca_shared.n_components_
            if pca_shared and self.pca_apply_to_target
            else target_base
        )
        if self.slim_mode:
            self.slim_channels = min(
                int(self.rail_channels),
                int(self.poi_channels),
                int(self.target_channels),
            )
            if self.slim_channels <= 0:
                raise ValueError(
                    "Slim mode requires at least one common rail/POI/target channel."
                )
            if (
                self.slim_channel_idx is not None
                and int(self.slim_channel_idx) >= self.slim_channels
            ):
                raise ValueError(
                    f"slim_channel_idx={self.slim_channel_idx} out of range for {self.slim_channels} channel(s)"
                )
            self.rail_channels = 1
            self.poi_channels = 1
            self.target_channels = 1

    def __len__(self):
        if (
            self.slim_mode
            and self.slim_expand_channels
            and (self.slim_channel_idx is None)
        ):
            return len(self.tiles) * int(self.slim_channels)
        return len(self.tiles)

    def _resolve_tile_and_channel(self, idx: int) -> Tuple[Dict, Optional[int]]:
        if (
            self.slim_mode
            and self.slim_expand_channels
            and (self.slim_channel_idx is None)
        ):
            tile_idx = idx // int(self.slim_channels)
            channel_idx = idx % int(self.slim_channels)
            return (self.tiles[tile_idx], channel_idx)
        return (self.tiles[idx], self.slim_channel_idx if self.slim_mode else None)

    def _slim_from_components(
        self, comp: Dict[str, np.ndarray], channel_idx: Optional[int]
    ):
        if channel_idx is None:
            raise ValueError("Slim component sample requires a channel index.")
        bus = comp["bus"].astype(np.float32, copy=False)
        rail = comp["rail"].astype(np.float32, copy=False)
        poi = comp["poi"].astype(np.float32, copy=False)
        target = comp["target"].astype(np.float32, copy=False)
        max_common = min(rail.shape[0], poi.shape[0], target.shape[0])
        if channel_idx < 0 or channel_idx >= max_common:
            raise IndexError(
                f"slim channel index {channel_idx} out of range for common channel count {max_common}"
            )
        inputs = np.concatenate(
            [
                bus,
                rail[channel_idx : channel_idx + 1],
                poi[channel_idx : channel_idx + 1],
            ],
            axis=0,
        )
        target = target[channel_idx : channel_idx + 1]
        mask = comp.get("mask")
        if mask is not None:
            mask = mask.astype(np.float32, copy=False)
            if mask.ndim == 2:
                mask = mask[None, ...]
            elif mask.ndim == 3 and mask.shape[0] > 1:
                mask = mask[channel_idx : channel_idx + 1]
        inputs_t = torch.from_numpy(inputs.astype(np.float32, copy=False))
        target_t = torch.from_numpy(target.astype(np.float32, copy=False))
        mask_t = (
            None
            if mask is None
            else torch.from_numpy(mask.astype(np.float32, copy=False))
        )
        return (inputs_t, target_t, mask_t)

    def __getitem__(self, idx):
        tile, channel_idx = self._resolve_tile_and_channel(idx)
        ram = tile.get("ram_sample")
        if ram is not None:
            if self.slim_mode and "inputs" not in ram:
                inputs, target, mask_t = self._slim_from_components(ram, channel_idx)
                return self._maybe_augment(
                    inputs, target, mask_t, channel_idx=channel_idx
                )
            mask = ram.get("mask")
            inputs = torch.from_numpy(ram["inputs"])
            target = torch.from_numpy(ram["target"])
            mask_t = None if mask is None else torch.from_numpy(mask)
            return self._maybe_augment(inputs, target, mask_t, channel_idx=channel_idx)
        if self.use_precomputed_samples and tile.get("sample_cache_arrays") is not None:
            cached = tile["sample_cache_arrays"]
            if self.slim_mode and "inputs" not in cached:
                inputs, target, mask_t = self._slim_from_components(cached, channel_idx)
                return self._maybe_augment(
                    inputs, target, mask_t, channel_idx=channel_idx
                )
            mask = cached.get("mask")
            inputs = torch.from_numpy(cached["inputs"].astype(np.float32, copy=False))
            target = torch.from_numpy(cached["target"].astype(np.float32, copy=False))
            mask_t = (
                None
                if mask is None
                else torch.from_numpy(mask.astype(np.float32, copy=False))
            )
            return self._maybe_augment(inputs, target, mask_t, channel_idx=channel_idx)
        if self.use_precomputed_samples and tile.get("sample_cache_path"):
            cache_keys = (
                ["inputs", "target", "mask"]
                if not (
                    self.slim_mode
                    and self.slim_expand_channels
                    and (self.slim_channel_idx is None)
                )
                else ["bus", "rail", "poi", "target", "mask"]
            )
            cached = load_tile_arrays(Path(tile["sample_cache_path"]), keys=cache_keys)
            if self.slim_mode and "inputs" not in cached:
                inputs, target, mask_t = self._slim_from_components(cached, channel_idx)
                return self._maybe_augment(
                    inputs, target, mask_t, channel_idx=channel_idx
                )
            mask = cached.get("mask")
            inputs = torch.from_numpy(cached["inputs"].astype(np.float32, copy=False))
            target = torch.from_numpy(cached["target"].astype(np.float32, copy=False))
            mask_t = (
                None
                if mask is None
                else torch.from_numpy(mask.astype(np.float32, copy=False))
            )
            return self._maybe_augment(inputs, target, mask_t, channel_idx=channel_idx)
        sample_keys = [self.bus_key, self.rail_key, self.poi_key, self.target_key]
        if self.mask_key:
            sample_keys.append(self.mask_key)
        payload = _materialize_tile_sample(
            tile,
            lazy=self.lazy,
            sample_keys=sample_keys,
            key_map={
                "bus": self.bus_key,
                "rail": self.rail_key,
                "poi": self.poi_key,
                "target": self.target_key,
                "mask": self.mask_key,
            },
            pca_map=self.pca_map,
            norm_stats=self.norm_stats,
            channel_trim=self.channel_trim,
            pca_apply_to_context=self.pca_apply_to_context,
            pca_apply_to_target=self.pca_apply_to_target,
            slim_mode=self.slim_mode,
            slim_channel_idx=channel_idx,
            slim_expand_channels=False,
        )
        inputs = torch.from_numpy(payload["inputs"])
        target = torch.from_numpy(payload["target"])
        mask_t = None if "mask" not in payload else torch.from_numpy(payload["mask"])
        return self._maybe_augment(inputs, target, mask_t, channel_idx=channel_idx)

    def _sample_bus_activation_scale(self) -> float:
        if (
            not self.augment_bus_activation_scale
            or self.bus_channels is None
            or self.bus_channels <= 0
            or (random.random() > self.augment_bus_activation_scale_prob)
        ):
            return 1.0
        if random.random() < self.augment_bus_activation_scale_keep_full_prob:
            return 1.0
        lo = max(self.augment_bus_activation_scale_min, 1e-08)
        hi = max(self.augment_bus_activation_scale_max, lo)
        if self.augment_bus_activation_scale_distribution == "log_uniform":
            return float(math.exp(random.uniform(math.log(lo), math.log(hi))))
        return float(random.uniform(lo, hi))

    def _scale_normalized_channels(
        self,
        tensor: torch.Tensor,
        *,
        scale: float,
        norm_key: str,
        channel_slice: slice,
        pca_mean_projection: Optional[torch.Tensor] = None,
        stats_channel_idx: Optional[int] = None,
    ) -> torch.Tensor:
        out = tensor.clone()
        values = out[channel_slice]
        if norm_key in self.norm_stats:
            mean_np, std_np = self.norm_stats[norm_key]
            mean = torch.as_tensor(mean_np, dtype=values.dtype, device=values.device)
            std = torch.as_tensor(std_np, dtype=values.dtype, device=values.device)
            if pca_mean_projection is not None:
                offset = pca_mean_projection.to(
                    dtype=values.dtype, device=values.device
                )
            else:
                offset = torch.zeros_like(mean)
            if values.shape[0] == 1 and mean.numel() != 1:
                idx = int(stats_channel_idx or 0)
                mean = mean[idx : idx + 1]
                std = std[idx : idx + 1]
                offset = offset[idx : idx + 1]
            reshape = (slice(None),) + (None,) * (values.ndim - 1)
            raw_projected = values * std[reshape] + mean[reshape]
            scaled_projected = scale * raw_projected + (scale - 1.0) * offset[reshape]
            values = (scaled_projected - mean[reshape]) / (std[reshape] + 1e-06)
        else:
            values = values * scale
            if pca_mean_projection is not None:
                offset = pca_mean_projection.to(
                    dtype=values.dtype, device=values.device
                )
                reshape = (slice(None),) + (None,) * (values.ndim - 1)
                values = values + (scale - 1.0) * offset[reshape]
        out[channel_slice] = values
        return out

    def _maybe_scale_bus_activation(
        self, inputs, target, *, channel_idx: Optional[int] = None
    ):
        """Expose the surrogate to soft bus activations used by inverse synthesis."""
        bus_scale = self._sample_bus_activation_scale()
        if bus_scale == 1.0:
            return (inputs, target)
        target_scale = float(bus_scale**self.augment_bus_activation_target_power)
        inputs = self._scale_normalized_channels(
            inputs,
            scale=bus_scale,
            norm_key="bus",
            channel_slice=slice(0, int(self.bus_channels or 0)),
        )
        target = self._scale_normalized_channels(
            target,
            scale=target_scale,
            norm_key="target",
            channel_slice=slice(0, target.shape[0]),
            pca_mean_projection=self._target_pca_mean_projection,
            stats_channel_idx=channel_idx if self.slim_mode else None,
        )
        return (inputs, target)

    def _maybe_augment_isolated_stop(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        *,
        channel_idx: Optional[int] = None,
    ):
        """Simulate isolated (unconnected) bus stops: zero edge channel + target."""
        if (
            not self.augment_isolated_stop
            or self.bus_channels is None
            or self.bus_channels < 2
            or (random.random() > self.augment_isolated_stop_prob)
        ):
            return (inputs, target)
        out_inputs = inputs.clone()
        out_target = target.clone()
        edge_ch_abs = 1
        if "bus" in self.norm_stats:
            mean_np, std_np = self.norm_stats["bus"]
            edge_norm_zero = float(-mean_np[1] / (std_np[1] + 1e-06))
            out_inputs[edge_ch_abs] = edge_norm_zero
        else:
            out_inputs[edge_ch_abs] = 0.0
        if "target" in self.norm_stats:
            mean_np, std_np = self.norm_stats["target"]
            for c in range(out_target.shape[0]):
                stat_idx = int(
                    channel_idx
                    if self.slim_mode
                    and out_target.shape[0] == 1
                    and (channel_idx is not None)
                    else c
                )
                out_target[c] = float(-mean_np[stat_idx] / (std_np[stat_idx] + 1e-06))
        else:
            out_target.zero_()
        return (out_inputs, out_target)

    def _maybe_augment_stub_edge(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        *,
        channel_idx: Optional[int] = None,
    ):
        """Replace bus edge channel with a short synthetic disconnected stub and zero the target."""
        if (
            not self.augment_stub_edge
            or self.bus_channels is None
            or self.bus_channels < 2
            or (random.random() > self.augment_stub_edge_prob)
        ):
            return (inputs, target)
        import math
        import numpy as np

        out_inputs = inputs.clone()
        out_target = target.clone()
        H, W = (out_inputs.shape[-2], out_inputs.shape[-1])
        flat_idx = int(torch.argmax(out_inputs[0].reshape(-1)).item())
        start_row = flat_idx // W
        start_col = flat_idx % W
        angle_rad = random.uniform(0.0, 2.0 * math.pi)
        min_len = max(1, self.augment_stub_edge_min_len_px)
        max_len = max(min_len, self.augment_stub_edge_max_len_px)
        length_px = random.randint(min_len, max_len)
        stub_raw = np.zeros((H, W), dtype=np.float32)
        sigma = 1.5
        n_steps = max(length_px, 1)
        rad = int(math.ceil(3.0 * sigma))
        for step in range(n_steps + 1):
            frac = step / n_steps
            r = start_row + frac * length_px * math.sin(angle_rad)
            c = start_col + frac * length_px * math.cos(angle_rad)
            ri, ci = (int(round(r)), int(round(c)))
            if not (0 <= ri < H and 0 <= ci < W):
                continue
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    nr, nc = (ri + dr, ci + dc)
                    if 0 <= nr < H and 0 <= nc < W:
                        val = math.exp(-(dr * dr + dc * dc) / (2.0 * sigma * sigma))
                        if val > stub_raw[nr, nc]:
                            stub_raw[nr, nc] = val
        if "bus" in self.norm_stats:
            mean_np, std_np = self.norm_stats["bus"]
            stub_norm = (stub_raw - float(mean_np[1])) / (float(std_np[1]) + 1e-06)
        else:
            stub_norm = stub_raw
        out_inputs[1] = torch.from_numpy(stub_norm).to(out_inputs.device)
        if "target" in self.norm_stats:
            mean_np, std_np = self.norm_stats["target"]
            for c in range(out_target.shape[0]):
                stat_idx = int(
                    channel_idx
                    if self.slim_mode
                    and out_target.shape[0] == 1
                    and (channel_idx is not None)
                    else c
                )
                out_target[c] = float(-mean_np[stat_idx] / (std_np[stat_idx] + 1e-06))
        else:
            out_target.zero_()
        return (out_inputs, out_target)

    def _maybe_augment_orphan_stop_overlay(
        self, inputs: torch.Tensor, target: torch.Tensor
    ):
        """Add isolated stop blobs while preserving route edges and labels."""
        if (
            not self.augment_orphan_stop_overlay
            or self.bus_channels is None
            or self.bus_channels < 2
            or (random.random() > self.augment_orphan_stop_overlay_prob)
        ):
            return (inputs, target)
        stop_norm = inputs[0]
        edge_norm = inputs[1]
        if "bus" in self.norm_stats:
            mean_np, std_np = self.norm_stats["bus"]
            stop_raw = stop_norm * float(std_np[0]) + float(mean_np[0])
            edge_raw = edge_norm * float(std_np[1]) + float(mean_np[1])
        else:
            stop_raw = stop_norm
            edge_raw = edge_norm
        if float(torch.clamp(stop_raw, min=0).sum().item()) <= 0.0:
            return (inputs, target)
        h, w = (stop_raw.shape[-2], stop_raw.shape[-1])
        max_shift = min(self.augment_orphan_stop_overlay_max_shift_px, max(h, w) - 1)
        min_shift = min(self.augment_orphan_stop_overlay_min_shift_px, max_shift)
        if max_shift <= 0:
            return (inputs, target)

        def _sample_shift() -> int:
            mag = (
                random.randint(min_shift, max_shift)
                if max_shift >= min_shift
                else max_shift
            )
            return mag if random.random() < 0.5 else -mag

        dy = _sample_shift()
        dx = _sample_shift()
        src = torch.clamp(stop_raw, min=0)
        shifted = torch.zeros_like(src)
        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(h, h + dy)
        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(w, w + dx)
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            return (inputs, target)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = src[src_y0:src_y1, src_x0:src_x1]
        edge_active = (edge_raw > 1e-06).to(dtype=shifted.dtype)
        radius = self.augment_orphan_stop_overlay_edge_dilate_px
        if radius > 0 and float(edge_active.sum().item()) > 0.0:
            edge_active = torch.nn.functional.max_pool2d(
                edge_active[None, None],
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )[0, 0]
        orphan_mask = edge_active <= 0.0
        scale = random.uniform(
            self.augment_orphan_stop_overlay_scale_min,
            self.augment_orphan_stop_overlay_scale_max,
        )
        orphan_raw = shifted * float(scale) * orphan_mask.to(dtype=shifted.dtype)
        if float(orphan_raw.sum().item()) <= 0.0:
            return (inputs, target)
        out_inputs = inputs.clone()
        stop_aug = torch.maximum(torch.clamp(stop_raw, min=0), orphan_raw)
        if "bus" in self.norm_stats:
            mean_np, std_np = self.norm_stats["bus"]
            out_inputs[0] = (stop_aug - float(mean_np[0])) / (float(std_np[0]) + 1e-06)
        else:
            out_inputs[0] = stop_aug
        return (out_inputs, target)

    def _maybe_augment(
        self, inputs, target, mask, *, channel_idx: Optional[int] = None
    ):
        """Apply random spatial augmentations consistently to inputs, target, and mask."""
        inputs, target = self._maybe_scale_bus_activation(
            inputs, target, channel_idx=channel_idx
        )
        inputs, target = self._maybe_augment_orphan_stop_overlay(inputs, target)
        inputs, target = self._maybe_augment_isolated_stop(
            inputs, target, channel_idx=channel_idx
        )
        inputs, target = self._maybe_augment_stub_edge(
            inputs, target, channel_idx=channel_idx
        )
        if not self.augment_flip and (not self.augment_rot90):
            return (inputs, target, mask)
        k = random.randint(0, 3) if self.augment_rot90 else 0
        hflip = self.augment_flip and random.random() < 0.5
        vflip = self.augment_flip and random.random() < 0.5
        if k == 0 and (not hflip) and (not vflip):
            return (inputs, target, mask)

        def _apply(t):
            if k:
                t = torch.rot90(t, k, dims=[1, 2])
            if hflip:
                t = torch.flip(t, dims=[2])
            if vflip:
                t = torch.flip(t, dims=[1])
            return t

        inputs = _apply(inputs)
        target = _apply(target)
        if mask is not None:
            mask = _apply(mask)
        return (inputs, target, mask)


def _split_tiles(
    tiles: List[Dict], split: Tuple[float, float, float], seed: int, shuffle: bool
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    n = len(tiles)
    split = _normalize_split(split)
    idxs = list(range(n))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idxs)
    n_train = int(round(split[0] * n))
    n_val = int(round(split[1] * n))
    n_train = min(max(n_train, 1), n)
    n_val = min(max(n_val, 0), n - n_train)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train : n_train + n_val]
    test_idx = idxs[n_train + n_val :]
    return (
        [tiles[i] for i in train_idx],
        [tiles[i] for i in val_idx],
        [tiles[i] for i in test_idx],
    )


def _estimate_batch_mb(
    sample: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], batch_size: int
) -> float:
    inputs, target, mask = sample
    total_elems = inputs.numel() + target.numel()
    if mask is not None:
        total_elems += mask.numel()
    bytes_per_elem = 4
    return total_elems * batch_size * bytes_per_elem / 1000000.0


def _shm_free_mb() -> Optional[float]:
    shm_path = Path("/dev/shm")
    if os.name != "posix" or not shm_path.exists():
        return None
    try:
        stats = os.statvfs(shm_path)
    except OSError:
        return None
    return stats.f_bavail * stats.f_frsize / 1000000.0


def _resolve_dataloader_runtime(
    cfg: DataConfig, est_batch_mb: Optional[float], loader_name: str
) -> Dict[str, Optional[int] | bool]:
    num_workers = max(0, int(cfg.num_workers))
    if num_workers == 0:
        return {"num_workers": 0, "persistent_workers": False, "prefetch_factor": None}
    if cfg.prefetch_factor is not None:
        effective_prefetch = int(cfg.prefetch_factor)
    elif cfg.preload_precomputed_samples:
        effective_prefetch = 4
    elif cfg.precomputed_sample_cache:
        effective_prefetch = 3
    else:
        effective_prefetch = 2
    effective_prefetch = max(1, effective_prefetch)
    requested_workers = num_workers
    requested_prefetch = effective_prefetch
    persistent_workers = bool(
        cfg.persistent_workers or (num_workers > 0 and cfg.precomputed_sample_cache)
    )
    shm_free_mb = _shm_free_mb()
    if est_batch_mb is not None and shm_free_mb is not None and (est_batch_mb > 0):
        shm_budget_mb = shm_free_mb * 0.35
        projected_mb = est_batch_mb * num_workers * effective_prefetch
        adjusted = False
        if projected_mb > shm_budget_mb and effective_prefetch > 1:
            effective_prefetch = 1
            projected_mb = est_batch_mb * num_workers * effective_prefetch
            adjusted = True
        if projected_mb > shm_budget_mb:
            max_workers = int(
                shm_budget_mb // max(est_batch_mb * effective_prefetch, 1e-06)
            )
            num_workers = max(0, min(num_workers, max_workers))
            projected_mb = est_batch_mb * num_workers * effective_prefetch
            adjusted = True
        if adjusted:
            warnings.warn(
                "%s DataLoader adjusted for /dev/shm budget: workers %s -> %s, prefetch %s -> %s | est_batch=%.2f MB | projected_inflight=%.2f MB | shm_free=%.2f MB | budget=%.2f MB"
                % (
                    loader_name,
                    requested_workers,
                    num_workers,
                    requested_prefetch,
                    effective_prefetch,
                    est_batch_mb,
                    projected_mb,
                    shm_free_mb,
                    shm_budget_mb,
                ),
                RuntimeWarning,
                stacklevel=2,
            )
    if num_workers <= 0:
        warnings.warn(
            "%s DataLoader falling back to num_workers=0 to avoid shared-memory worker crashes."
            % (loader_name,),
            RuntimeWarning,
            stacklevel=2,
        )
        return {"num_workers": 0, "persistent_workers": False, "prefetch_factor": None}
    prefetch_to_pass = None
    if cfg.prefetch_factor is not None or effective_prefetch != 2:
        prefetch_to_pass = effective_prefetch
    return {
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_to_pass,
    }


def _resolve_precompute_workers(cfg: DataConfig) -> int:
    if cfg.precompute_workers is None:
        if cfg.num_workers:
            return max(0, min(8, int(cfg.num_workers)))
        return 0
    return max(0, int(cfg.precompute_workers))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values)) / max(len(values), 1)


def _check_spatial_consistency(
    tiles: List[Dict], lazy: bool = False, target_mode: str = "delta"
):
    heights = set()
    widths = set()
    for t in tiles:
        shape = _tile_grid_shape(t, lazy=lazy, target_mode=target_mode)
        if shape is None:
            continue
        h, w = shape
        heights.add(h)
        widths.add(w)
    if len(heights) > 1 or len(widths) > 1:
        raise ValueError(
            f"Tiles have inconsistent spatial shapes. Heights={heights}, Widths={widths}"
        )


def _resolve_tile_grid(tile: Dict) -> Optional[Dict]:
    """Return the grid meta dict for a tile (with the same meta_* fallback as _tile_grid_shape),
    or None if unavailable."""
    meta = _get_tile_meta(tile)
    if not meta:
        return None
    grid = meta.get("grid")
    if not grid:
        for key in ("meta_bus_delta", "meta_walk_transit_rail", "meta_poi", "meta_bus"):
            entry = meta.get(key) or {}
            if isinstance(entry, dict) and entry.get("grid"):
                grid = entry.get("grid")
                break
    return grid if isinstance(grid, dict) else None


def _check_geo_consistency(
    tiles: List[Dict], lazy: bool = False, target_mode: str = "delta"
):
    """Assert CRS and pixel size agree across a run_selector mix."""
    crs_seen = set()
    pixel_m_seen = set()
    for t in tiles:
        grid = _resolve_tile_grid(t)
        if not grid:
            continue
        crs = grid.get("crs")
        if crs is not None:
            crs_seen.add(str(crs))
        bounds = grid.get("bounds")
        width = grid.get("width")
        if bounds and len(bounds) == 4 and width:
            pixel_m_seen.add(
                round((float(bounds[2]) - float(bounds[0])) / float(width), 1)
            )
    if len(crs_seen) > 1:
        raise ValueError(f"Tiles mix coordinate reference systems: {sorted(crs_seen)}")
    if len(pixel_m_seen) > 1:
        raise ValueError(
            f"Tiles mix ground pixel sizes (m): {sorted(pixel_m_seen)} — a same-H×W mix of different resolutions would corrupt training. Check the run_selector mix."
        )


def _build_dataloader(
    dataset: Dataset,
    cfg: DataConfig,
    shuffle: bool,
    *,
    est_batch_mb: Optional[float] = None,
    loader_name: str = "loader",
) -> DataLoader:
    runtime = _resolve_dataloader_runtime(cfg, est_batch_mb, loader_name)
    kwargs = {
        "dataset": dataset,
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": runtime["num_workers"],
        "pin_memory": bool(cfg.pin_memory),
        "persistent_workers": runtime["persistent_workers"],
        "drop_last": cfg.drop_last,
    }
    if runtime["num_workers"] > 0 and runtime["prefetch_factor"] is not None:
        kwargs["prefetch_factor"] = runtime["prefetch_factor"]
    return DataLoader(**kwargs)


def train_unet_pipeline(
    data_cfg: DataConfig = DataConfig(),
    model_cfg: ModelConfig = ModelConfig(),
    train_cfg: TrainConfig = TrainConfig(),
) -> TrainResult:
    """Train a configurable U-Net on bus/rail/POI tensors to predict bus_delta access."""
    init_seed = (
        data_cfg.init_seed if data_cfg.init_seed is not None else data_cfg.split_seed
    )
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)
    if train_cfg.torch_threads:
        torch.set_num_threads(train_cfg.torch_threads)
    manual_device = train_cfg.device is not None
    if manual_device:
        requested = str(train_cfg.device)
        if requested.lower().startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError(
                f"Requested device '{requested}' but CUDA is not available."
            )
        train_device = requested
    else:
        train_device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg.device = train_device
    manifest, tiles, run_ids = _load_tiles(
        data_cfg.output_root,
        data_cfg.run_selector,
        lazy=data_cfg.lazy_load,
        limit=data_cfg.limit_samples,
    )
    precompute_workers = _resolve_precompute_workers(data_cfg)
    if tiles:
        if data_cfg.lazy_load:
            first_tile_names = list_tile_array_keys(_tile_source_array_path(tiles[0]))
        else:
            first_tile_names = list(tiles[0]["arrays"].keys())
    else:
        first_tile_names = []
    bus_key = _pick_name(first_tile_names, ["bus_tensor"])
    rail_key = _pick_name(
        first_tile_names,
        ["access_walk_transit_rail", "walk_transit_rail"],
        fallback_prefix="access_",
    )
    poi_key = _pick_name(first_tile_names, ["poi_tensor"])
    target_key = _pick_target_name(first_tile_names, data_cfg.target_mode)
    _check_spatial_consistency(
        tiles, lazy=data_cfg.lazy_load, target_mode=data_cfg.target_mode
    )
    _check_geo_consistency(
        tiles, lazy=data_cfg.lazy_load, target_mode=data_cfg.target_mode
    )
    min_counts = _align_and_trim_channels(
        tiles, lazy=data_cfg.lazy_load, target_mode=data_cfg.target_mode
    )
    if str(getattr(data_cfg, "split_mode", "random")).lower() == "geographic":
        from modules.geo_split import split_tiles_geographic

        if not data_cfg.holdout_region:
            raise ValueError(
                "split_mode='geographic' requires data_cfg.holdout_region to be set"
            )
        _ckpt_stem = (
            Path(train_cfg.checkpoint_name).stem
            if train_cfg.checkpoint_name
            else "geo_holdout"
        )
        _geo_dump = str(Path(train_cfg.checkpoint_dir) / f"{_ckpt_stem}.geo_split.json")
        train_tiles, val_tiles, test_tiles, _geo_summary = split_tiles_geographic(
            tiles,
            data_cfg.holdout_region,
            buffer_m=data_cfg.holdout_buffer_m,
            val_fraction=data_cfg.holdout_val_fraction,
            seed=data_cfg.split_seed,
            dump_path=_geo_dump,
        )
    else:
        train_tiles, val_tiles, test_tiles = _split_tiles(
            tiles, data_cfg.split, data_cfg.split_seed, data_cfg.shuffle
        )
    _tr = {t.get("tile_idx") for t in train_tiles}
    _va = {t.get("tile_idx") for t in val_tiles}
    _te = {t.get("tile_idx") for t in test_tiles}
    _overlap = _tr & _va | _tr & _te | _va & _te
    if _overlap:
        raise ValueError(
            f"Split leakage: {len(_overlap)} tile(s) appear in more than one of train/val/test: {sorted(_overlap)[:10]}"
        )
    if precompute_workers > 1:
        pass
    pca_map: Dict[str, Optional[PCA]] = {"shared": None}
    if data_cfg.pca_components:
        pca_init_keys = [key for key in (rail_key, poi_key, target_key) if key]
        arrays0 = _get_tile_arrays(
            tiles[0], lazy=data_cfg.lazy_load, keys=pca_init_keys
        )
        rail_key = _pick_key(
            arrays0,
            ["access_walk_transit_rail", "walk_transit_rail"],
            fallback_prefix="access_",
        )
        poi_key = _pick_key(arrays0, ["poi_tensor"])
        target_key = _pick_target_key(arrays0, data_cfg.target_mode)
        if not rail_key or not poi_key or (not target_key):
            raise ValueError(
                "PCA enabled requires rail, poi, and target tensors to be present across runs."
            )
        rail_ch = min(arrays0[rail_key].shape[0], min_counts["rail"])
        poi_ch = min(arrays0[poi_key].shape[0], min_counts["poi"])
        target_ch = min(arrays0[target_key].shape[0], min_counts["target"])
        pca_channel_counts = []
        if data_cfg.pca_apply_to_context:
            pca_channel_counts.extend([rail_ch, poi_ch])
        if data_cfg.pca_apply_to_target:
            pca_channel_counts.append(target_ch)
        if len(set(pca_channel_counts)) > 1:
            raise ValueError(
                f"PCA enabled requires matching channel counts across tensors it is applied to; got rail={rail_ch} (apply={data_cfg.pca_apply_to_context}), poi={poi_ch} (apply={data_cfg.pca_apply_to_context}), target={target_ch} (apply={data_cfg.pca_apply_to_target})"
            )
        fit_source = data_cfg.pca_fit_source.lower()
        if fit_source == "combined":
            fit_keys = []
            if data_cfg.pca_apply_to_context:
                fit_keys.extend([rail_key, poi_key])
            if data_cfg.pca_apply_to_target:
                fit_keys.append(target_key)
        elif fit_source == "poi":
            if not data_cfg.pca_apply_to_context:
                raise ValueError(
                    "pca_fit_source='poi' requires pca_apply_to_context=True"
                )
            fit_keys = [poi_key]
        elif fit_source == "rail":
            if not data_cfg.pca_apply_to_context:
                raise ValueError(
                    "pca_fit_source='rail' requires pca_apply_to_context=True"
                )
            fit_keys = [rail_key]
        elif fit_source == "target":
            if not data_cfg.pca_apply_to_target:
                raise ValueError(
                    "pca_fit_source='target' requires pca_apply_to_target=True"
                )
            fit_keys = [target_key]
        else:
            raise ValueError(f"Unsupported pca_fit_source '{data_cfg.pca_fit_source}'")
        if not fit_keys:
            raise ValueError(
                "pca_components is set but PCA is disabled for both context and target tensors."
            )
        channel_overrides = {}
        if rail_key:
            channel_overrides[rail_key] = min_counts["rail"]
        if poi_key:
            channel_overrides[poi_key] = min_counts["poi"]
        if target_key:
            channel_overrides[target_key] = min_counts["target"]
        pca_map["shared"] = _fit_shared_pca(
            train_tiles,
            keys=fit_keys,
            n_components=data_cfg.pca_components,
            max_samples=data_cfg.pca_max_samples,
            random_state=data_cfg.pca_random_state,
            lazy=data_cfg.lazy_load,
            channel_overrides=channel_overrides,
            workers=precompute_workers,
            progress_label="pca:shared",
        )
    norm_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if data_cfg.normalize_inputs or data_cfg.normalize_targets:
        norm_init_keys = [
            key for key in (bus_key, rail_key, poi_key, target_key) if key
        ]
        arrays0 = _get_tile_arrays(
            train_tiles[0], lazy=data_cfg.lazy_load, keys=norm_init_keys
        )
        bus_key = _pick_key(arrays0, ["bus_tensor"])
        rail_key = _pick_key(
            arrays0,
            ["access_walk_transit_rail", "walk_transit_rail"],
            fallback_prefix="access_",
        )
        poi_key = _pick_key(arrays0, ["poi_tensor"])
        target_key = _pick_target_key(arrays0, data_cfg.target_mode)
        if data_cfg.normalize_inputs and bus_key and data_cfg.normalize_bus_channels:
            norm_stats["bus"] = _compute_norm_stats(
                train_tiles,
                bus_key,
                pca=None,
                lazy=data_cfg.lazy_load,
                target_channels=min_counts["bus"],
                workers=precompute_workers,
                progress_label="norm:bus",
            )
        elif (
            data_cfg.normalize_inputs
            and bus_key
            and (not data_cfg.normalize_bus_channels)
        ):
            pass
        if data_cfg.normalize_inputs and rail_key:
            norm_stats["rail"] = _compute_norm_stats(
                train_tiles,
                rail_key,
                pca_map["shared"] if data_cfg.pca_apply_to_context else None,
                lazy=data_cfg.lazy_load,
                target_channels=min_counts["rail"],
                workers=precompute_workers,
                progress_label="norm:rail",
            )
        if data_cfg.normalize_inputs and poi_key:
            norm_stats["poi"] = _compute_norm_stats(
                train_tiles,
                poi_key,
                pca_map["shared"] if data_cfg.pca_apply_to_context else None,
                lazy=data_cfg.lazy_load,
                target_channels=min_counts["poi"],
                workers=precompute_workers,
                progress_label="norm:poi",
            )
        if data_cfg.normalize_targets and target_key:
            norm_stats["target"] = _compute_norm_stats(
                train_tiles,
                target_key,
                pca_map["shared"] if data_cfg.pca_apply_to_target else None,
                lazy=data_cfg.lazy_load,
                target_channels=min_counts["target"],
                workers=precompute_workers,
                progress_label="norm:target",
            )
    sample_keys = [key for key in (bus_key, rail_key, poi_key, target_key) if key]
    if train_tiles:
        sample_name_candidates = (
            list_tile_array_keys(_tile_source_array_path(train_tiles[0]))
            if data_cfg.lazy_load
            else list(train_tiles[0]["arrays"].keys())
        )
    else:
        sample_name_candidates = []
    mask_key = _pick_target_mask_name(sample_name_candidates, data_cfg.target_mode)
    if mask_key:
        sample_keys.append(mask_key)
    key_map = {
        "bus": bus_key,
        "rail": rail_key,
        "poi": poi_key,
        "target": target_key,
        "mask": mask_key,
        "target_mode": data_cfg.target_mode,
    }
    _prepare_precomputed_sample_cache(
        tiles,
        enabled=bool(data_cfg.precomputed_sample_cache),
        lazy=data_cfg.lazy_load,
        sample_keys=sample_keys,
        key_map=key_map,
        pca_map=pca_map,
        norm_stats=norm_stats,
        channel_trim=min_counts,
        cache_root=data_cfg.precomputed_sample_cache_dir,
        force=bool(data_cfg.precomputed_sample_cache_force_rebuild),
        workers=precompute_workers,
        pca_apply_to_context=bool(data_cfg.pca_apply_to_context),
        pca_apply_to_target=bool(data_cfg.pca_apply_to_target),
        slim_mode=bool(data_cfg.slim_mode),
        slim_channel_idx=data_cfg.slim_channel_idx,
        slim_expand_channels=bool(data_cfg.slim_expand_channels),
    )
    _preload_precomputed_samples(
        tiles,
        enabled=bool(
            data_cfg.precomputed_sample_cache and data_cfg.preload_precomputed_samples
        ),
        max_gb=data_cfg.preload_precomputed_samples_max_gb,
    )
    _preload_tiles_to_ram(
        tiles,
        enabled=bool(data_cfg.preload_to_ram),
        lazy=data_cfg.lazy_load,
        sample_keys=sample_keys,
        key_map=key_map,
        pca_map=pca_map,
        norm_stats=norm_stats,
        channel_trim=min_counts,
        workers=precompute_workers,
        pca_apply_to_context=bool(data_cfg.pca_apply_to_context),
        pca_apply_to_target=bool(data_cfg.pca_apply_to_target),
        slim_mode=bool(data_cfg.slim_mode),
        slim_channel_idx=data_cfg.slim_channel_idx,
        slim_expand_channels=bool(data_cfg.slim_expand_channels),
    )
    train_ds = TileDataset(
        train_tiles,
        pca_map=pca_map,
        norm_stats=norm_stats,
        limit=data_cfg.limit_samples,
        lazy=data_cfg.lazy_load,
        channel_trim=min_counts,
        use_precomputed_samples=bool(data_cfg.precomputed_sample_cache),
        pca_apply_to_context=data_cfg.pca_apply_to_context,
        pca_apply_to_target=data_cfg.pca_apply_to_target,
        target_mode=data_cfg.target_mode,
        slim_mode=data_cfg.slim_mode,
        slim_channel_idx=data_cfg.slim_channel_idx,
        slim_expand_channels=data_cfg.slim_expand_channels,
        augment_flip=data_cfg.augment_flip,
        augment_rot90=data_cfg.augment_rot90,
        augment_bus_activation_scale=data_cfg.augment_bus_activation_scale,
        augment_bus_activation_scale_prob=data_cfg.augment_bus_activation_scale_prob,
        augment_bus_activation_scale_min=data_cfg.augment_bus_activation_scale_min,
        augment_bus_activation_scale_max=data_cfg.augment_bus_activation_scale_max,
        augment_bus_activation_scale_distribution=data_cfg.augment_bus_activation_scale_distribution,
        augment_bus_activation_scale_keep_full_prob=data_cfg.augment_bus_activation_scale_keep_full_prob,
        augment_bus_activation_target_power=data_cfg.augment_bus_activation_target_power,
        augment_isolated_stop=data_cfg.augment_isolated_stop,
        augment_isolated_stop_prob=data_cfg.augment_isolated_stop_prob,
        augment_stub_edge=data_cfg.augment_stub_edge,
        augment_stub_edge_prob=data_cfg.augment_stub_edge_prob,
        augment_stub_edge_min_len_px=data_cfg.augment_stub_edge_min_len_px,
        augment_stub_edge_max_len_px=data_cfg.augment_stub_edge_max_len_px,
        augment_orphan_stop_overlay=data_cfg.augment_orphan_stop_overlay,
        augment_orphan_stop_overlay_prob=data_cfg.augment_orphan_stop_overlay_prob,
        augment_orphan_stop_overlay_min_shift_px=data_cfg.augment_orphan_stop_overlay_min_shift_px,
        augment_orphan_stop_overlay_max_shift_px=data_cfg.augment_orphan_stop_overlay_max_shift_px,
        augment_orphan_stop_overlay_edge_dilate_px=data_cfg.augment_orphan_stop_overlay_edge_dilate_px,
        augment_orphan_stop_overlay_scale_min=data_cfg.augment_orphan_stop_overlay_scale_min,
        augment_orphan_stop_overlay_scale_max=data_cfg.augment_orphan_stop_overlay_scale_max,
    )
    val_ds = (
        TileDataset(
            val_tiles,
            pca_map=pca_map,
            norm_stats=norm_stats,
            lazy=data_cfg.lazy_load,
            channel_trim=min_counts,
            use_precomputed_samples=bool(data_cfg.precomputed_sample_cache),
            pca_apply_to_context=data_cfg.pca_apply_to_context,
            pca_apply_to_target=data_cfg.pca_apply_to_target,
            target_mode=data_cfg.target_mode,
            slim_mode=data_cfg.slim_mode,
            slim_channel_idx=data_cfg.slim_channel_idx,
            slim_expand_channels=data_cfg.slim_expand_channels,
        )
        if val_tiles
        else None
    )
    test_ds = (
        TileDataset(
            test_tiles,
            pca_map=pca_map,
            norm_stats=norm_stats,
            lazy=data_cfg.lazy_load,
            channel_trim=min_counts,
            use_precomputed_samples=bool(data_cfg.precomputed_sample_cache),
            pca_apply_to_context=data_cfg.pca_apply_to_context,
            pca_apply_to_target=data_cfg.pca_apply_to_target,
            target_mode=data_cfg.target_mode,
            slim_mode=data_cfg.slim_mode,
            slim_channel_idx=data_cfg.slim_channel_idx,
            slim_expand_channels=data_cfg.slim_expand_channels,
        )
        if test_tiles
        else None
    )
    total_in_ch = train_ds.bus_channels + train_ds.rail_channels + train_ds.poi_channels
    sample_in, sample_out, sample_mask = train_ds[0]
    resolved_in_ch = model_cfg.in_channels or sample_in.shape[0]
    resolved_out_ch = model_cfg.out_channels or sample_out.shape[0]
    model_cfg_resolved = asdict(model_cfg)
    model_cfg_resolved["in_channels"] = resolved_in_ch
    model_cfg_resolved["out_channels"] = resolved_out_ch
    channel_info = {
        "bus_channels": train_ds.bus_channels,
        "rail_channels": train_ds.rail_channels,
        "poi_channels": train_ds.poi_channels,
        "target_channels": train_ds.target_channels,
        "input_channels": total_in_ch,
        "height": train_ds.height,
        "width": train_ds.width,
        "target_mode": data_cfg.target_mode,
        "slim_mode": bool(data_cfg.slim_mode),
        "slim_channel_idx": data_cfg.slim_channel_idx,
        "slim_expand_channels": bool(data_cfg.slim_expand_channels),
        "slim_channels": int(getattr(train_ds, "slim_channels", 1)),
        "pca_apply_to_context": bool(data_cfg.pca_apply_to_context),
        "pca_apply_to_target": bool(data_cfg.pca_apply_to_target),
        "tensor_schema": _json_safe(_infer_manifest_tensor_schema(manifest)),
    }
    pca_state = _extract_pca_state(pca_map.get("shared"))
    pca_meta = None
    if pca_map.get("shared") is not None:
        pca_obj = pca_map["shared"]
        try:
            explained = float(np.sum(pca_obj.explained_variance_ratio_))
        except Exception:
            explained = None
        pca_meta = {
            "enabled": True,
            "n_components": int(
                getattr(pca_obj, "n_components_", pca_obj.components_.shape[0])
            ),
            "n_features_in": int(
                getattr(pca_obj, "n_features_in_", pca_obj.components_.shape[1])
            ),
            "fit_source": data_cfg.pca_fit_source,
            "apply_to_context": bool(data_cfg.pca_apply_to_context),
            "apply_to_target": bool(data_cfg.pca_apply_to_target),
            "explained_variance_ratio_sum": explained,
        }
    norm_stats_meta = {
        key: {"mean_shape": list(mean.shape), "std_shape": list(std.shape)}
        for key, (mean, std) in norm_stats.items()
    }

    def _run_training_attempt(batch_size: int) -> TrainResult:
        data_cfg.batch_size = batch_size
        est_mb = _estimate_batch_mb(
            (sample_in, sample_out, sample_mask), data_cfg.batch_size
        )
        in_ch = resolved_in_ch
        out_ch = resolved_out_ch
        model = build_model(model_cfg, in_ch, out_ch, channel_info=channel_info).to(
            train_device
        )
        uncertainty_on = bool(getattr(model_cfg, "output_uncertainty", False))
        _loss_is_nll = str(train_cfg.loss_type).lower() in {"gaussian_nll", "nll"}
        if uncertainty_on and (not _loss_is_nll):
            raise ValueError(
                f"model_cfg.output_uncertainty=True requires train_cfg.loss_type='gaussian_nll' (got '{train_cfg.loss_type}')."
            )
        if _loss_is_nll and (not uncertainty_on):
            raise ValueError(
                "train_cfg.loss_type='gaussian_nll' requires model_cfg.output_uncertainty=True so the head emits (μ, logσ²)."
            )
        param_count = sum((p.numel() for p in model.parameters()))
        if train_cfg.torch_compile:
            model = torch.compile(model, mode="reduce-overhead")
        model_figure_paths = []
        opt_name = train_cfg.optimizer.lower()
        if opt_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_cfg.lr,
                weight_decay=train_cfg.weight_decay,
                betas=(train_cfg.beta1, train_cfg.beta2),
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg.lr,
                weight_decay=train_cfg.weight_decay,
                betas=(train_cfg.beta1, train_cfg.beta2),
            )
        else:
            raise ValueError(f"Unsupported optimizer '{train_cfg.optimizer}'")
        scheduler = None
        if train_cfg.lr_scheduler is not None:
            if train_cfg.lr_scheduler == "reduce_on_plateau":
                sched_kwargs = dict(
                    optimizer=optimizer,
                    mode="min",
                    factor=train_cfg.lr_scheduler_factor,
                    patience=train_cfg.lr_scheduler_patience,
                )
                if (
                    "verbose"
                    in inspect.signature(
                        torch.optim.lr_scheduler.ReduceLROnPlateau.__init__
                    ).parameters
                ):
                    sched_kwargs["verbose"] = True
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(**sched_kwargs)
            elif train_cfg.lr_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=train_cfg.epochs, eta_min=0.0
                )
            else:
                raise ValueError(f"Unsupported lr_scheduler '{train_cfg.lr_scheduler}'")
        train_loader = _build_dataloader(
            train_ds, data_cfg, shuffle=True, est_batch_mb=est_mb, loader_name="train"
        )
        val_loader = (
            _build_dataloader(
                val_ds, data_cfg, shuffle=False, est_batch_mb=est_mb, loader_name="val"
            )
            if val_ds
            else None
        )
        test_loader = (
            _build_dataloader(
                test_ds,
                data_cfg,
                shuffle=False,
                est_batch_mb=est_mb,
                loader_name="test",
            )
            if test_ds
            else None
        )
        history: Dict[str, List] = {
            "train_loss": [],
            "train_mae": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "lr": [],
            "rel_gap": [],
            "epoch_sec": [],
            "grad_norm": [],
        }
        best_val = math.inf
        best_epoch = None
        stopped_early = False
        best_path = None
        patience_counter = 0
        device_type = "cuda" if str(train_device).lower().startswith("cuda") else "cpu"
        copy_non_blocking = bool(data_cfg.pin_memory and device_type == "cuda")
        scaler_enabled = train_cfg.use_amp and device_type == "cuda"
        scaler_kwargs = {"enabled": scaler_enabled}
        if _GRADSCALER_SUPPORTS_DEVICE_TYPE:
            scaler_kwargs["device_type"] = device_type
        scaler = _GradScaler(**scaler_kwargs)
        train_start = time.time()
        epochs_completed = 0
        base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        global_step = 0
        batch_losses = deque(maxlen=5)
        train_ma_window = deque(maxlen=5)
        val_ma_window = deque(maxlen=5)
        start_epoch = 1
        resume_path = (
            _resume_state_path(train_cfg)
            if train_cfg.save_checkpoint and getattr(train_cfg, "resume", True)
            else None
        )
        _cur_fp = data_fingerprint(data_cfg)
        if resume_path is not None and resume_path.exists():
            _rs_fp = object()
            try:
                _rs_fp = torch.load(
                    resume_path, map_location="cpu", weights_only=False
                ).get("data_fingerprint")
            except Exception as exc:
                warnings.warn(
                    "RESUME: could not read %s to verify data fingerprint (%s)"
                    % (resume_path, exc),
                    RuntimeWarning,
                    stacklevel=2,
                )
                _rs_fp = None
            _legacy = _rs_fp is None
            _allow_legacy = os.environ.get("SURROGATE_ALLOW_LEGACY_RESUME") == "1"
            if _legacy and _allow_legacy:
                warnings.warn(
                    "RESUME: %s carries no data fingerprint (legacy file). Proceeding ONLY because SURROGATE_ALLOW_LEGACY_RESUME=1 was set explicitly — the caller asserts this state was produced by the current data/split config."
                    % (resume_path,),
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif _legacy or _rs_fp != _cur_fp:
                quarantined = resume_path.with_suffix(
                    resume_path.suffix + ".incompatible"
                )
                try:
                    resume_path.replace(quarantined)
                except Exception:
                    pass
                warnings.warn(
                    "RESUME REFUSED: resume-file data fingerprint %r != current config %r%s. Moved it to %s and starting FRESH from epoch 1 — resuming would have mixed two different experiments (see data_fingerprint docstring)."
                    % (
                        _rs_fp,
                        _cur_fp,
                        (
                            " [legacy file, no fingerprint; set SURROGATE_ALLOW_LEGACY_RESUME=1 to accept]"
                            if _legacy
                            else ""
                        ),
                        quarantined,
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                resume_path = _resume_state_path(train_cfg)
        if resume_path is not None and resume_path.exists():
            try:
                rs = torch.load(
                    resume_path, map_location=train_device, weights_only=False
                )
                model.load_state_dict(rs["model_state"])
                optimizer.load_state_dict(rs["optimizer_state"])
                if scheduler is not None and rs.get("scheduler_state") is not None:
                    scheduler.load_state_dict(rs["scheduler_state"])
                if rs.get("scaler_state") is not None:
                    scaler.load_state_dict(rs["scaler_state"])
                best_val = (
                    rs.get("best_val") if rs.get("best_val") is not None else math.inf
                )
                best_epoch = rs.get("best_epoch")
                patience_counter = int(rs.get("patience_counter", 0))
                global_step = int(rs.get("global_step", 0))
                if isinstance(rs.get("history"), dict):
                    history = rs["history"]
                try:
                    torch.set_rng_state(rs["torch_rng"])
                    if rs.get("cuda_rng") is not None and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(rs["cuda_rng"])
                    import numpy as _np

                    if rs.get("numpy_rng") is not None:
                        _np.random.set_state(rs["numpy_rng"])
                except Exception:
                    pass
                start_epoch = int(rs["epoch"]) + 1
            except Exception as exc:
                warnings.warn(
                    "RESUME: failed to load %s (%s) — starting from epoch 1"
                    % (resume_path, exc),
                    RuntimeWarning,
                    stacklevel=2,
                )
                start_epoch = 1
        if start_epoch > train_cfg.epochs:
            existing_ckpts = [
                p for p in _checkpoint_output_paths(train_cfg) if p.exists()
            ]
            complete_path = existing_ckpts[0] if existing_ckpts else None
            if complete_path is not None:
                try:
                    _done_payload = torch.load(
                        complete_path, map_location=train_device, weights_only=False
                    )
                    model.load_state_dict(_done_payload["model_state"])
                except Exception as exc:
                    warnings.warn(
                        "RESUME: could not load completed checkpoint %s (%s)"
                        % (complete_path, exc),
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if resume_path is not None and resume_path.exists():
                try:
                    resume_path.unlink()
                except OSError:
                    pass
            return TrainResult(
                history=history,
                final_train_metrics={},
                final_val_metrics=None,
                final_test_metrics=None,
                checkpoint_path=complete_path,
                figure_path=None,
                model=model,
                config={
                    "data": asdict(data_cfg),
                    "model": model_cfg_resolved,
                    "train": asdict(train_cfg),
                    "manifest": manifest,
                    "channel_info": _json_safe(channel_info),
                    "pca_meta": (
                        _json_safe(pca_meta) if pca_meta else {"enabled": False}
                    ),
                    "norm_stats_meta": _json_safe(norm_stats_meta),
                },
                param_count=param_count,
                train_samples=len(train_tiles),
                train_time_sec=0.0,
                sec_per_epoch=0.0,
                val_samples=len(val_tiles),
                test_samples=len(test_tiles),
                already_complete=True,
            )
        for epoch in range(start_epoch, train_cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            running_mae = 0.0
            running_count = 0.0
            optimizer.zero_grad(set_to_none=train_cfg.zero_grad_set_to_none)
            epoch_start = time.perf_counter()
            history["lr"].append(optimizer.param_groups[0]["lr"])
            epoch_grad_norms: List[float] = []
            for batch_idx, (inputs, targets, masks) in enumerate(train_loader, 1):
                inputs = inputs.to(train_device, non_blocking=copy_non_blocking)
                targets = targets.to(train_device, non_blocking=copy_non_blocking)
                masks = (
                    masks.to(train_device, non_blocking=copy_non_blocking)
                    if masks is not None
                    else None
                )
                masks = _prepare_mask(masks, targets)
                take_step = (
                    batch_idx % train_cfg.grad_accum_steps == 0
                    or batch_idx == len(train_loader)
                )
                if train_cfg.warmup_steps > 0 and global_step < train_cfg.warmup_steps:
                    scale_lr = float(global_step + 1) / float(train_cfg.warmup_steps)
                    for base_lr, pg in zip(base_lrs, optimizer.param_groups):
                        pg["lr"] = base_lr * scale_lr
                if scaler.is_enabled():
                    if _AUTOCAST_SUPPORTS_DEVICE_TYPE:
                        amp_ctx = lambda: _autocast(
                            device_type=device_type, enabled=True
                        )
                    else:
                        amp_ctx = lambda: _autocast(enabled=True)
                else:
                    amp_ctx = nullcontext
                with amp_ctx():
                    preds, logvar_out = _forward_pred(model, inputs, uncertainty_on)
                    preds = _apply_output_activation(preds, train_cfg.output_activation)
                    loss_raw = masked_loss(
                        preds,
                        targets,
                        masks,
                        loss_type=train_cfg.loss_type,
                        smooth_l1_beta=train_cfg.smooth_l1_beta,
                        peak_weight_alpha=train_cfg.peak_weight_alpha,
                        peak_weight_power=train_cfg.peak_weight_power,
                        grad_loss_weight=train_cfg.grad_loss_weight,
                        logvar=logvar_out,
                    )
                    loss = loss_raw / float(train_cfg.grad_accum_steps)
                    if train_cfg.tv_weight > 0:
                        loss = loss + train_cfg.tv_weight * anisotropic_tv(
                            preds
                        ) / float(train_cfg.grad_accum_steps)
                batch_losses.append(loss_raw.item())
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if take_step:
                        if train_cfg.grad_clip:
                            scaler.unscale_(optimizer)
                            gn = nn.utils.clip_grad_norm_(
                                model.parameters(), train_cfg.grad_clip
                            )
                            epoch_grad_norms.append(float(gn))
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=train_cfg.zero_grad_set_to_none)
                        global_step += 1
                else:
                    loss.backward()
                    if take_step:
                        if train_cfg.grad_clip:
                            gn = nn.utils.clip_grad_norm_(
                                model.parameters(), train_cfg.grad_clip
                            )
                            epoch_grad_norms.append(float(gn))
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=train_cfg.zero_grad_set_to_none)
                        global_step += 1
                with torch.no_grad():
                    mae = masked_mae(preds, targets, masks)
                    elem_count = (
                        masks.sum().item() if masks is not None else preds.numel()
                    )
                    running_loss += loss_raw.item() * elem_count
                    running_mae += mae.item() * elem_count
                    running_count += elem_count
            epoch_train_loss = running_loss / max(running_count, 1e-06)
            epoch_train_mae = running_mae / max(running_count, 1e-06)
            train_ma_window.append(epoch_train_loss)
            train_loss_ma = _mean(train_ma_window)
            history["train_loss"].append(epoch_train_loss)
            history["train_mae"].append(epoch_train_mae)
            val_loss = None
            val_mae = None
            if val_loader and epoch % train_cfg.val_interval == 0:
                model.eval()
                v_loss_sum = 0.0
                v_mse_sum = 0.0
                v_mae_sum = 0.0
                v_count = 0.0
                with torch.no_grad():
                    for inputs, targets, masks in val_loader:
                        inputs = inputs.to(train_device, non_blocking=copy_non_blocking)
                        targets = targets.to(
                            train_device, non_blocking=copy_non_blocking
                        )
                        masks = (
                            masks.to(train_device, non_blocking=copy_non_blocking)
                            if masks is not None
                            else None
                        )
                        masks = _prepare_mask(masks, targets)
                        preds, logvar_out = _forward_pred(model, inputs, uncertainty_on)
                        preds = _apply_output_activation(
                            preds, train_cfg.output_activation
                        )
                        l = masked_loss(
                            preds,
                            targets,
                            masks,
                            loss_type=train_cfg.loss_type,
                            smooth_l1_beta=train_cfg.smooth_l1_beta,
                            peak_weight_alpha=train_cfg.peak_weight_alpha,
                            peak_weight_power=train_cfg.peak_weight_power,
                            grad_loss_weight=train_cfg.grad_loss_weight,
                            logvar=logvar_out,
                        )
                        l_mse = masked_loss(
                            preds,
                            targets,
                            masks,
                            loss_type="mse",
                            smooth_l1_beta=train_cfg.smooth_l1_beta,
                            peak_weight_alpha=train_cfg.peak_weight_alpha,
                            peak_weight_power=train_cfg.peak_weight_power,
                            grad_loss_weight=train_cfg.grad_loss_weight,
                        )
                        m = masked_mae(preds, targets, masks)
                        elem_count = (
                            masks.sum().item() if masks is not None else preds.numel()
                        )
                        v_loss_sum += l.item() * elem_count
                        v_mse_sum += l_mse.item() * elem_count
                        v_mae_sum += m.item() * elem_count
                        v_count += elem_count
                val_loss = v_loss_sum / max(v_count, 1e-06)
                val_mae = v_mae_sum / max(v_count, 1e-06)
                val_rmse = math.sqrt(max(v_mse_sum / max(v_count, 1e-06), 0.0))
                history["val_loss"].append(val_loss)
                history["val_mae"].append(val_mae)
                val_ma_window.append(val_loss)
                val_loss_ma = _mean(val_ma_window)
                rel_gap = (val_loss_ma - train_loss_ma) / max(train_loss_ma, 1e-08)
                history["val_rmse"].append(val_rmse)
                history["rel_gap"].append(rel_gap)
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    if train_cfg.save_checkpoint:
                        train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_paths = _checkpoint_output_paths(train_cfg)
                        best_path = checkpoint_paths[0] if checkpoint_paths else None
                        manifest_payload = {
                            "checkpoint_path": (
                                str(best_path) if best_path is not None else None
                            ),
                            "model_figure_path": (
                                str(model_figure_paths[0])
                                if model_figure_paths
                                else None
                            ),
                            "model_figure_paths": [
                                str(path) for path in model_figure_paths
                            ],
                            "epoch": int(epoch),
                            "val_loss": float(val_loss),
                            "val_rmse": float(val_rmse),
                            "data_cfg": _json_safe(asdict(data_cfg)),
                            "model_cfg": _json_safe(model_cfg_resolved),
                            "architecture": {
                                "family": str(model_cfg.architecture).lower(),
                                "residual_blocks": bool(model_cfg.residual_blocks),
                                "skip_attention": model_cfg.skip_attention,
                                "bottleneck_attention": model_cfg.bottleneck_attention,
                                "decoder_attention": model_cfg.decoder_attention,
                                "attention_reduction": int(
                                    model_cfg.attention_reduction
                                ),
                                "input_projection_channels": model_cfg.input_projection_channels,
                                "aspp_dilations": list(model_cfg.aspp_dilations),
                                "aspp_dropout": float(model_cfg.aspp_dropout),
                                "dual_stream_fusion": model_cfg.dual_stream_fusion,
                                "bus_input_channels": model_cfg.bus_input_channels,
                                "context_input_channels": model_cfg.context_input_channels,
                                "dual_stream_share_encoder": bool(
                                    model_cfg.dual_stream_share_encoder
                                ),
                                "triple_stream_fusion": model_cfg.triple_stream_fusion,
                                "triple_stream_share_rail_poi": bool(
                                    model_cfg.triple_stream_share_rail_poi
                                ),
                                "rail_input_channels": model_cfg.rail_input_channels,
                                "poi_input_channels": model_cfg.poi_input_channels,
                            },
                            "train_cfg": _json_safe(asdict(train_cfg)),
                            "channel_info": _json_safe(channel_info),
                            "pca": (
                                _json_safe(pca_meta) if pca_meta else {"enabled": False}
                            ),
                            "norm_stats": _json_safe(norm_stats_meta),
                        }
                        manifest_payload["model_io_spec"] = build_model_io_spec(
                            data_cfg=_json_safe(asdict(data_cfg)),
                            model_cfg=_json_safe(model_cfg_resolved),
                            train_cfg=_json_safe(asdict(train_cfg)),
                            channel_info=_json_safe(channel_info),
                            pca_meta=(
                                _json_safe(pca_meta) if pca_meta else {"enabled": False}
                            ),
                            norm_stats_meta=_json_safe(norm_stats_meta),
                            checkpoint_path=(
                                str(best_path) if best_path is not None else None
                            ),
                            history_json_path=(
                                str(
                                    best_path.with_suffix(
                                        best_path.suffix + ".history.json"
                                    )
                                )
                                if best_path is not None
                                else None
                            ),
                            model_figure_path=(
                                str(model_figure_paths[0])
                                if model_figure_paths
                                else None
                            ),
                            model_figure_paths=[
                                str(path) for path in model_figure_paths
                            ],
                            epoch_history=_json_safe(history),
                        )
                        synthesis_bindings = {
                            "version": 1,
                            "pca_enabled": bool(pca_state is not None),
                            "pca_components": (
                                None
                                if pca_state is None
                                else int(pca_state.get("n_components", 0))
                            ),
                            "pca_features_in": (
                                None
                                if pca_state is None
                                else int(pca_state.get("n_features_in", 0))
                            ),
                            "normalize_inputs": bool(data_cfg.normalize_inputs),
                            "normalize_targets": bool(data_cfg.normalize_targets),
                            "output_activation": train_cfg.output_activation,
                            "channel_info": _json_safe(channel_info),
                            "policy_layer_names": _json_safe(
                                (_infer_manifest_tensor_schema(manifest) or {}).get(
                                    "rail"
                                )
                                or []
                            ),
                        }
                        manifest_payload["synthesis_bindings"] = _json_safe(
                            synthesis_bindings
                        )
                        manifest_payload["val_split"] = {
                            "seed": data_cfg.split_seed,
                            "mode": data_cfg.split_mode,
                            "fractions": list(data_cfg.split),
                            "val_tile_indices": sorted(
                                [t["tile_idx"] for t in val_tiles]
                            ),
                            "train_tile_indices": sorted(
                                [t["tile_idx"] for t in train_tiles]
                            ),
                            "test_tile_indices": sorted(
                                [t["tile_idx"] for t in test_tiles]
                            ),
                            "val_tile_count": len(val_tiles),
                            "train_tile_count": len(train_tiles),
                            "test_tile_count": len(test_tiles),
                            "total_tile_count": len(train_tiles)
                            + len(val_tiles)
                            + len(test_tiles),
                            "dataset_run_selector": str(data_cfg.run_selector),
                            "dataset_output_root": str(data_cfg.output_root),
                        }
                        ckpt_payload = {
                            "model_state": model.state_dict(),
                            "config": {
                                "data": asdict(data_cfg),
                                "model": model_cfg_resolved,
                                "train": asdict(train_cfg),
                            },
                            "manifest": manifest,
                            "model_manifest": manifest_payload,
                            "channel_info": channel_info,
                            "pca_state": pca_state,
                            "norm_stats": norm_stats,
                            "synthesis_bindings": synthesis_bindings,
                            "in_channels": in_ch,
                            "out_channels": out_ch,
                        }
                        for ckpt_path in checkpoint_paths:
                            manifest_payload_one = dict(manifest_payload)
                            manifest_payload_one["checkpoint_path"] = str(ckpt_path)
                            torch.save(ckpt_payload, ckpt_path)
                            manifest_path = ckpt_path.with_suffix(
                                ckpt_path.suffix + ".manifest.json"
                            )
                            manifest_path.write_text(
                                json.dumps(manifest_payload_one, indent=2)
                            )
                if val_loss >= best_val:
                    patience_counter += 1
                    if (
                        train_cfg.early_stop_patience
                        and patience_counter >= train_cfg.early_stop_patience
                    ):
                        epochs_completed = epoch
                        stopped_early = True
                        break
                if scheduler:
                    if (
                        train_cfg.lr_scheduler == "reduce_on_plateau"
                        and val_loss is not None
                    ):
                        scheduler.step(val_loss)
                    elif train_cfg.lr_scheduler == "cosine":
                        scheduler.step()
            else:
                history["val_loss"].append(None)
                history["val_mae"].append(None)
                history["val_rmse"].append(None)
                history["rel_gap"].append(None)
            history["epoch_sec"].append(time.perf_counter() - epoch_start)
            history["grad_norm"].append(
                float(sum(epoch_grad_norms) / len(epoch_grad_norms))
                if epoch_grad_norms
                else None
            )
            if scheduler and train_cfg.lr_scheduler == "cosine":
                scheduler.step()
            epochs_completed = epoch
            _write_checkpoint_history(
                train_cfg,
                history,
                epoch=epoch,
                total_epochs=train_cfg.epochs,
                best_val=best_val if math.isfinite(best_val) else None,
                best_epoch=best_epoch,
                latest_val_loss=val_loss,
                latest_checkpoint_path=best_path,
                info_text=_format_training_info(train_cfg, optimizer),
            )
            if resume_path is not None:
                try:
                    _save_resume_state(
                        resume_path,
                        epoch=epoch,
                        best_val=best_val,
                        best_epoch=best_epoch,
                        patience_counter=patience_counter,
                        global_step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        history=history,
                        data_fp=_cur_fp,
                    )
                except Exception as exc:
                    warnings.warn(
                        "RESUME: failed to write %s (%s)" % (resume_path, exc),
                        RuntimeWarning,
                        stacklevel=2,
                    )
        if resume_path is not None and resume_path.exists():
            try:
                resume_path.unlink()
            except OSError:
                pass
        train_time_sec = time.time() - train_start
        sec_per_epoch = train_time_sec / max(epochs_completed, 1)
        if (
            best_path is not None
            and Path(best_path).exists()
            and (best_epoch is not None)
        ):
            try:
                _best_payload = torch.load(
                    best_path, map_location=train_device, weights_only=False
                )
                model.load_state_dict(_best_payload["model_state"])
            except Exception as exc:
                warnings.warn(
                    "Could not restore best checkpoint %s for final eval (%s) — final metrics reflect last-epoch weights"
                    % (best_path, exc),
                    RuntimeWarning,
                    stacklevel=2,
                )

        def _eval_loader(loader: Optional[DataLoader], name: str):
            if loader is None:
                return None
            model.eval()
            loss_sum = 0.0
            mse_sum = 0.0
            mae_sum = 0.0
            count = 0.0
            with torch.no_grad():
                for inputs, targets, masks in loader:
                    inputs = inputs.to(train_device, non_blocking=copy_non_blocking)
                    targets = targets.to(train_device, non_blocking=copy_non_blocking)
                    masks = (
                        masks.to(train_device, non_blocking=copy_non_blocking)
                        if masks is not None
                        else None
                    )
                    masks = _prepare_mask(masks, targets)
                    preds, logvar_out = _forward_pred(model, inputs, uncertainty_on)
                    preds = _apply_output_activation(preds, train_cfg.output_activation)
                    l = masked_loss(
                        preds,
                        targets,
                        masks,
                        loss_type=train_cfg.loss_type,
                        smooth_l1_beta=train_cfg.smooth_l1_beta,
                        peak_weight_alpha=train_cfg.peak_weight_alpha,
                        peak_weight_power=train_cfg.peak_weight_power,
                        grad_loss_weight=train_cfg.grad_loss_weight,
                        logvar=logvar_out,
                    )
                    l_mse = masked_loss(
                        preds,
                        targets,
                        masks,
                        loss_type="mse",
                        smooth_l1_beta=train_cfg.smooth_l1_beta,
                        peak_weight_alpha=train_cfg.peak_weight_alpha,
                        peak_weight_power=train_cfg.peak_weight_power,
                        grad_loss_weight=train_cfg.grad_loss_weight,
                    )
                    m = masked_mae(preds, targets, masks)
                    elem_count = (
                        masks.sum().item() if masks is not None else preds.numel()
                    )
                    loss_sum += l.item() * elem_count
                    mse_sum += l_mse.item() * elem_count
                    mae_sum += m.item() * elem_count
                    count += elem_count
            loss_avg = loss_sum / max(count, 1e-06)
            mse_avg = mse_sum / max(count, 1e-06)
            mae_avg = mae_sum / max(count, 1e-06)
            metrics = {
                "loss": loss_avg,
                "mse": mse_avg,
                "mae": mae_avg,
                "rmse": math.sqrt(max(mse_avg, 0)),
            }
            return metrics

        final_train_metrics = _eval_loader(train_loader, "train_final")
        final_val_metrics = _eval_loader(val_loader, "val_final")
        final_test_metrics = _eval_loader(test_loader, "test_final")
        fig_path = None
        _write_checkpoint_history(
            train_cfg,
            history,
            epoch=epochs_completed,
            total_epochs=train_cfg.epochs,
            best_val=best_val if math.isfinite(best_val) else None,
            best_epoch=best_epoch,
            latest_val_loss=next(
                (v for v in reversed(history["val_loss"]) if v is not None), None
            ),
            latest_checkpoint_path=best_path,
            info_text=_format_training_info(train_cfg, optimizer),
        )
        return TrainResult(
            history=history,
            final_train_metrics=final_train_metrics or {},
            final_val_metrics=final_val_metrics,
            final_test_metrics=final_test_metrics,
            checkpoint_path=best_path,
            figure_path=fig_path,
            model=model,
            config={
                "data": asdict(data_cfg),
                "model": model_cfg_resolved,
                "train": asdict(train_cfg),
                "manifest": manifest,
                "channel_info": _json_safe(channel_info),
                "pca_meta": _json_safe(pca_meta) if pca_meta else {"enabled": False},
                "norm_stats_meta": _json_safe(norm_stats_meta),
            },
            param_count=param_count,
            train_samples=len(train_tiles),
            train_time_sec=train_time_sec,
            sec_per_epoch=sec_per_epoch,
            val_samples=len(val_tiles),
            test_samples=len(test_tiles),
            best_epoch=best_epoch,
            best_val_loss=best_val if math.isfinite(best_val) else None,
            total_epochs_run=epochs_completed,
            stopped_early=stopped_early,
        )

    batch_size = data_cfg.batch_size
    max_retries = max(0, train_cfg.oom_retries)
    for attempt in range(max_retries + 1):
        try:
            return _run_training_attempt(batch_size)
        except torch.cuda.OutOfMemoryError:
            if not str(train_device).lower().startswith("cuda"):
                raise
            if batch_size <= 1 or attempt >= max_retries:
                raise
            new_batch = max(1, batch_size // 2)
            warnings.warn(
                "CUDA OOM during training (batch_size=%s). Retrying with batch_size=%s (attempt %s/%s)."
                % (batch_size, new_batch, attempt + 1, max_retries),
                RuntimeWarning,
                stacklevel=2,
            )
            batch_size = new_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    raise RuntimeError("OOM retry loop exited unexpectedly")


def _format_training_info(train_cfg: TrainConfig, optimizer) -> str:
    opt_name = train_cfg.optimizer
    lr = getattr(train_cfg, "lr", None)
    wd = getattr(train_cfg, "weight_decay", None)
    sched = train_cfg.lr_scheduler or "none"
    sched_extra = ""
    if sched == "reduce_on_plateau":
        sched_extra = f" (patience={train_cfg.lr_scheduler_patience}, factor={train_cfg.lr_scheduler_factor})"
    elif sched == "cosine":
        sched_extra = " (cosine)"
    parts = [
        f"opt={opt_name}",
        f"lr={lr:.2e}" if lr is not None else "lr=?",
        f"wd={wd:.1e}" if wd is not None else "wd=?",
        f"sched={sched}{sched_extra}",
        f"grad_accum={train_cfg.grad_accum_steps}",
        f"warmup={train_cfg.warmup_steps}",
        f"tv={train_cfg.tv_weight}",
        f"clip={train_cfg.grad_clip}",
        f"amp={train_cfg.use_amp}",
    ]
    return " | ".join(parts)


__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "TrainResult",
    "UNet",
    "ResUNetASPP",
    "DualStreamResUNet",
    "build_model",
    "train_unet_pipeline",
]
try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast

    _GRADSCALER_SUPPORTS_DEVICE_TYPE = (
        "device_type" in inspect.signature(_GradScaler).parameters
    )
    _AUTOCAST_SUPPORTS_DEVICE_TYPE = (
        "device_type" in inspect.signature(_autocast).parameters
    )
except Exception:
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast

    _GRADSCALER_SUPPORTS_DEVICE_TYPE = False
    _AUTOCAST_SUPPORTS_DEVICE_TYPE = False
