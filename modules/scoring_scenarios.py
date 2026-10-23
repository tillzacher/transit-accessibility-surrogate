"""Scenario format, I/O helpers, and model-space adapter for behavioral scoring."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np


@dataclass
class ScenarioTile:
    """One authored scenario tile, loaded from a .npz file."""

    bus_stops: np.ndarray
    bus_edges: np.ndarray
    rail_raw: np.ndarray
    poi_raw: np.ndarray
    eval_mask: np.ndarray
    control_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_scenario(path: Union[str, Path]) -> ScenarioTile:
    """Load a scenario from a .npz file and return a ScenarioTile."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    npz = np.load(path, allow_pickle=False)
    bus_stops = npz["bus_stops"].astype(np.float32)
    bus_edges = npz["bus_edges"].astype(np.float32)
    rail_raw = npz["rail_raw"].astype(np.float32)
    poi_raw = npz["poi_raw"].astype(np.float32)
    eval_mask = npz["eval_mask"].astype(bool)
    control_mask = (
        npz["control_mask"].astype(bool) if "control_mask" in npz.files else None
    )
    meta_raw = npz["metadata"]
    if meta_raw.ndim == 0:
        metadata = json.loads(str(meta_raw))
    else:
        metadata = json.loads(str(meta_raw.flat[0]))
    H, W = bus_stops.shape
    assert bus_edges.shape == (
        H,
        W,
    ), f"bus_edges shape mismatch: {bus_edges.shape} vs ({H},{W})"
    assert rail_raw.ndim == 3 and rail_raw.shape[1:] == (
        H,
        W,
    ), f"rail_raw must be (C, H, W), got {rail_raw.shape}"
    assert poi_raw.ndim == 3 and poi_raw.shape[1:] == (
        H,
        W,
    ), f"poi_raw must be (C, H, W), got {poi_raw.shape}"
    assert eval_mask.shape == (
        H,
        W,
    ), f"eval_mask shape mismatch: {eval_mask.shape} vs ({H},{W})"
    if control_mask is not None:
        assert control_mask.shape == (
            H,
            W,
        ), f"control_mask shape mismatch: {control_mask.shape} vs ({H},{W})"
    return ScenarioTile(
        bus_stops=bus_stops,
        bus_edges=bus_edges,
        rail_raw=rail_raw,
        poi_raw=poi_raw,
        eval_mask=eval_mask,
        control_mask=control_mask,
        metadata=metadata,
    )


def save_scenario(
    tile: ScenarioTile, path: Union[str, Path], *, compressed: bool = False
) -> None:
    """Save a ScenarioTile to a .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        bus_stops=tile.bus_stops.astype(np.float32),
        bus_edges=tile.bus_edges.astype(np.float32),
        rail_raw=tile.rail_raw.astype(np.float32),
        poi_raw=tile.poi_raw.astype(np.float32),
        eval_mask=tile.eval_mask.astype(bool),
        metadata=np.array(json.dumps(tile.metadata)),
    )
    if tile.control_mask is not None:
        arrays["control_mask"] = tile.control_mask.astype(bool)
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **arrays)


def _require_full_channels(
    arr: np.ndarray, target_channels: int, stream: str
) -> np.ndarray:
    """Validate and return one complete raw-channel scenario tensor."""
    if arr.ndim != 3:
        raise ValueError(f"scenario {stream} must be (C,H,W), got {arr.shape}")
    if arr.shape[0] != target_channels:
        raise ValueError(
            f"scenario {stream} requires {target_channels} raw channels; got shape {arr.shape}"
        )
    return arr.astype(np.float32, copy=False)


def _is_total_target(ckpt: Any) -> bool:
    """True if this checkpoint predicts total accessibility rather than the bus-induced delta."""
    cfg = getattr(ckpt, "config", None) or getattr(ckpt, "cfg", None) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    data = cfg.get("data") or {}
    mode = data.get("target_mode")
    if mode is None:
        raw = getattr(ckpt, "raw", None)
        if isinstance(raw, dict):
            mode = ((raw.get("config") or {}).get("data") or {}).get("target_mode")
    return str(mode).lower() == "total"


def run_model_raw(ckpt: Any, scenario: ScenarioTile, device: str = "cpu") -> np.ndarray:
    """Run a loaded checkpoint on a drawn scenario tile."""
    from modules.surrogate_inference import (
        prepare_tensor_context,
        SurrogateInferenceAdapter,
    )

    if ckpt.pca is not None:
        rail_raw_count = int(ckpt.pca.n_features_in_)
        poi_raw_count = rail_raw_count
    else:
        norm_stats = ckpt.norm_stats or {}

        def _get_raw_channel_count(key: str) -> int:
            entry = norm_stats.get(key)
            if isinstance(entry, dict):
                return len(np.asarray(entry["mean"]))
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                return len(np.asarray(entry[0]))
            raise RuntimeError(
                f"Cannot determine raw {key} width for non-PCA checkpoint: norm_stats['{key}'] format not recognised. Expected dict with 'mean' key or 2-tuple (mean, std)."
            )

        rail_raw_count = _get_raw_channel_count("rail")
        poi_raw_count = _get_raw_channel_count("poi")
    tile_ctx = {
        "rail": _require_full_channels(scenario.rail_raw, rail_raw_count, "rail"),
        "poi": _require_full_channels(scenario.poi_raw, poi_raw_count, "poi"),
    }
    ctx = prepare_tensor_context(ckpt, tile_ctx)
    bus = np.stack([scenario.bus_stops, scenario.bus_edges], axis=0)
    adapter = SurrogateInferenceAdapter(ckpt, device=device)
    if adapter.model_kind == "slim_fixed":
        fixed = adapter.spec.get("output_contract", {}).get("fixed_channel_idx")
        raise ValueError(
            f"Behavioral scoring is undefined for a per-channel slim model (model_kind='slim_fixed', fixed_channel_idx={fixed}): it predicts only one PCA component, not a full delta. Behavioral scoring supports 'dense' and 'slim_shared' models. To score the factorized family, use the shared (slim_expand_channels=True) model or ensemble the full per-channel set."
        )
    pred_model = adapter.predict_model_space(ctx, bus)
    pred_raw = adapter.display_space(pred_model, "raw")
    if _is_total_target(ckpt):
        bus_zero = np.zeros_like(bus)
        base_model = adapter.predict_model_space(ctx, bus_zero)
        base_raw = adapter.display_space(base_model, "raw")
        pred_raw = pred_raw - base_raw
    return pred_raw.astype(np.float32)


__all__ = [
    "ScenarioTile",
    "load_scenario",
    "save_scenario",
    "_require_full_channels",
    "run_model_raw",
]
