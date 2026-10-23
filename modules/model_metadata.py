from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            value = float(obj)
            return None if math.isnan(value) or math.isinf(value) else value
    except Exception:
        pass
    return obj


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _history_values(history: Dict[str, Any], key: str) -> List[float]:
    values = history.get(key)
    if not isinstance(values, list):
        return []
    out: List[float] = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def convergence_summary(
    history: Optional[Dict[str, Any]], *, total_epochs_planned: Optional[int] = None
) -> Dict[str, Any]:
    history = _as_dict(history)
    train_loss = _history_values(history, "train_loss")
    train_mae = _history_values(history, "train_mae")
    val_loss = _history_values(history, "val_loss")
    val_mae = _history_values(history, "val_mae")
    val_rmse = _history_values(history, "val_rmse")
    rel_gap = _history_values(history, "rel_gap")
    epoch_sec = _history_values(history, "epoch_sec")
    best_val_loss = min(val_loss) if val_loss else None
    best_val_epoch = None
    if best_val_loss is not None:
        for idx, value in enumerate(history.get("val_loss", [])):
            try:
                if float(value) == best_val_loss:
                    best_val_epoch = idx + 1
                    break
            except Exception:
                continue

    def at_best(values: List[float]) -> Optional[float]:
        if best_val_epoch is None or best_val_epoch < 1 or best_val_epoch > len(values):
            return None
        return values[best_val_epoch - 1]

    total_epochs_run = max(len(train_loss), len(val_loss), len(val_mae), len(val_rmse))
    return {
        "total_epochs_planned": total_epochs_planned,
        "total_epochs_run": total_epochs_run,
        "early_stopped": bool(
            total_epochs_planned
            and total_epochs_run
            and (total_epochs_run < int(total_epochs_planned))
        ),
        "best_val_epoch": best_val_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mae": at_best(val_mae),
        "best_val_rmse": at_best(val_rmse),
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_train_mae": train_mae[-1] if train_mae else None,
        "final_val_loss": val_loss[-1] if val_loss else None,
        "final_val_mae": val_mae[-1] if val_mae else None,
        "final_val_rmse": val_rmse[-1] if val_rmse else None,
        "final_rel_gap": rel_gap[-1] if rel_gap else None,
        "min_rel_gap": min(rel_gap) if rel_gap else None,
        "max_rel_gap": max(rel_gap) if rel_gap else None,
        "sec_per_epoch_avg": sum(epoch_sec) / len(epoch_sec) if epoch_sec else None,
        "history_keys": sorted(
            (k for k, v in history.items() if isinstance(v, list) and v)
        ),
    }


def build_model_io_spec(
    *,
    data_cfg: Optional[Dict[str, Any]] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
    train_cfg: Optional[Dict[str, Any]] = None,
    channel_info: Optional[Dict[str, Any]] = None,
    pca_meta: Optional[Dict[str, Any]] = None,
    norm_stats_meta: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
    manifest_json_path: Optional[str] = None,
    history_json_path: Optional[str] = None,
    model_figure_path: Optional[str] = None,
    model_figure_paths: Optional[List[str]] = None,
    epoch_history: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    data_cfg = _as_dict(data_cfg)
    model_cfg = _as_dict(model_cfg)
    train_cfg = _as_dict(train_cfg)
    channel_info = _as_dict(channel_info)
    pca_meta = _as_dict(pca_meta)
    norm_stats_meta = _as_dict(norm_stats_meta)
    pca_enabled = bool(pca_meta.get("enabled") or data_cfg.get("pca_components"))
    model_channels = int(
        pca_meta.get("n_components")
        or data_cfg.get("pca_components")
        or channel_info.get("target_channels")
        or channel_info.get("rail_channels")
        or model_cfg.get("out_channels")
        or 0
    )
    raw_channels = int(
        pca_meta.get("n_features_in")
        or channel_info.get("raw_channels")
        or channel_info.get("target_channels")
        or channel_info.get("rail_channels")
        or model_channels
        or 0
    )
    slim_mode = bool(channel_info.get("slim_mode", data_cfg.get("slim_mode", False)))
    slim_expand = bool(
        channel_info.get(
            "slim_expand_channels", data_cfg.get("slim_expand_channels", False)
        )
    )
    slim_idx = channel_info.get("slim_channel_idx", data_cfg.get("slim_channel_idx"))
    figure_paths = [str(path) for path in model_figure_paths or [] if path]
    if model_figure_path and model_figure_path not in figure_paths:
        figure_paths.insert(0, str(model_figure_path))
    if not slim_mode:
        model_kind = "dense"
        inference_strategy = "single_pass"
        output_channels_per_pass = model_channels
        full_model_output_available = True
    elif slim_expand:
        model_kind = "slim_shared"
        inference_strategy = "loop_channels_same_checkpoint"
        output_channels_per_pass = 1
        full_model_output_available = model_channels > 0
    elif slim_idx is not None:
        model_kind = "slim_fixed"
        inference_strategy = "single_channel_only"
        output_channels_per_pass = 1
        full_model_output_available = False
    else:
        model_kind = "slim_shared"
        inference_strategy = "loop_channels_same_checkpoint"
        output_channels_per_pass = 1
        full_model_output_available = model_channels > 0
    full_raw_output_available = bool(
        full_model_output_available
        and (
            not pca_enabled
            or model_channels == int(pca_meta.get("n_components") or model_channels)
        )
    )
    history_summary = convergence_summary(
        epoch_history, total_epochs_planned=train_cfg.get("epochs")
    )
    schema = _as_dict(channel_info.get("tensor_schema"))
    spec = {
        "version": 1,
        "run_name": run_name,
        "model_kind": model_kind,
        "target_mode": data_cfg.get("target_mode")
        or channel_info.get("target_mode")
        or "delta",
        "feature_space": "pca" if pca_enabled else "raw",
        "inference_strategy": inference_strategy,
        "input_contract": {
            "bus_channels": int(
                channel_info.get("bus_channels")
                or model_cfg.get("bus_input_channels")
                or 2
            ),
            "rail_channels_per_pass": 1 if slim_mode else model_channels,
            "poi_channels_per_pass": 1 if slim_mode else model_channels,
            "input_channels_per_pass": int(
                model_cfg.get("in_channels")
                or channel_info.get("input_channels")
                or (4 if slim_mode else 2 + 2 * model_channels)
            ),
            "normalized_inputs": bool(
                data_cfg.get("normalize_inputs", bool(norm_stats_meta))
            ),
            "pca_apply_to_context": bool(
                data_cfg.get(
                    "pca_apply_to_context",
                    channel_info.get("pca_apply_to_context", pca_enabled),
                )
            ),
        },
        "output_contract": {
            "output_channels_per_pass": output_channels_per_pass,
            "model_space_channels": model_channels,
            "raw_space_channels": raw_channels,
            "full_model_output_available": full_model_output_available,
            "full_raw_output_available": full_raw_output_available,
            "fixed_channel_idx": None if slim_idx is None else int(slim_idx),
            "normalized_targets": bool(
                data_cfg.get("normalize_targets", bool(norm_stats_meta))
            ),
            "pca_apply_to_target": bool(
                data_cfg.get(
                    "pca_apply_to_target",
                    channel_info.get("pca_apply_to_target", pca_enabled),
                )
            ),
        },
        "channel_axis": {
            "model_count": model_channels,
            "raw_count": raw_channels,
            "raw_names": schema.get("target") or schema.get("rail") or [],
            "pca_components": (
                int(pca_meta.get("n_components") or 0) if pca_enabled else 0
            ),
        },
        "artifacts": {
            "checkpoint_path": checkpoint_path,
            "manifest_json_path": manifest_json_path,
            "history_json_path": history_json_path,
            "model_figure_path": figure_paths[0] if figure_paths else model_figure_path,
            "model_figure_paths": figure_paths,
        },
        "convergence_summary": history_summary,
    }
    return json_safe(spec)


