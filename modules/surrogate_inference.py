from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import torch
from modules.model_io import LoadedCheckpoint
from modules.model_metadata import build_model_io_spec


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


def apply_norm_np(
    arr: np.ndarray,
    stats: Optional[Dict[str, Any]],
    key: str,
    channel_idx: Optional[int] = None,
) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    if not stats or key not in stats:
        return arr
    mean, std = stats[key]
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if channel_idx is not None and arr.shape[0] == 1 and (mean.shape[0] > 1):
        idx = int(np.clip(channel_idx, 0, mean.shape[0] - 1))
        mean = mean[idx : idx + 1]
        std = std[idx : idx + 1]
    if arr.shape[0] != mean.shape[0] or arr.shape[0] != std.shape[0]:
        raise ValueError(
            f"Cannot normalize {key}: tensor has {arr.shape[0]} channels but stats have {mean.shape[0]}/{std.shape[0]}"
        )
    return ((arr - mean[:, None, None]) / (std[:, None, None] + 1e-06)).astype(
        np.float32
    )


def undo_norm_np(
    arr: np.ndarray,
    stats: Optional[Dict[str, Any]],
    key: str,
    channel_idx: Optional[int] = None,
) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    if not stats or key not in stats:
        return arr
    mean, std = stats[key]
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if channel_idx is not None and arr.shape[0] == 1 and (mean.shape[0] > 1):
        idx = int(np.clip(channel_idx, 0, mean.shape[0] - 1))
        mean = mean[idx : idx + 1]
        std = std[idx : idx + 1]
    if arr.shape[0] != mean.shape[0] or arr.shape[0] != std.shape[0]:
        raise ValueError(
            f"Cannot denormalize {key}: tensor has {arr.shape[0]} channels but stats have {mean.shape[0]}/{std.shape[0]}"
        )
    return (arr * (std[:, None, None] + 1e-06) + mean[:, None, None]).astype(np.float32)


def project_pca(arr: np.ndarray, pca: Any) -> np.ndarray:
    if pca is None:
        return arr.astype(np.float32, copy=False)
    C, H, W = arr.shape
    flat = arr.reshape(C, -1).T
    return pca.transform(flat).T.reshape(-1, H, W).astype(np.float32)


def inverse_project_pca(arr: np.ndarray, pca: Any) -> np.ndarray:
    if pca is None:
        return arr.astype(np.float32, copy=False)
    C, H, W = arr.shape
    flat = arr.reshape(C, -1).T
    return pca.inverse_transform(flat).T.reshape(-1, H, W).astype(np.float32)


@dataclass
class SurrogateTensorContext:
    rail_raw: np.ndarray
    poi_raw: np.ndarray
    rail_model: np.ndarray
    poi_model: np.ndarray
    raw_channel_count: int
    model_channel_count: int


def model_io_spec_from_checkpoint(ckpt: LoadedCheckpoint) -> Dict[str, Any]:
    manifest = ckpt.manifest if isinstance(ckpt.manifest, dict) else {}
    embedded = manifest.get("model_io_spec")
    if isinstance(embedded, dict):
        return embedded
    cfg = ckpt.config if isinstance(ckpt.config, dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    channel_info = manifest.get("channel_info") if isinstance(manifest, dict) else None
    if not isinstance(channel_info, dict):
        channel_info = (
            getattr(ckpt, "config", {}).get("channel_info", {})
            if isinstance(getattr(ckpt, "config", {}), dict)
            else {}
        )
    pca_meta = manifest.get("pca") if isinstance(manifest.get("pca"), dict) else {}
    return build_model_io_spec(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        channel_info=channel_info,
        pca_meta=pca_meta,
        norm_stats_meta=(
            manifest.get("norm_stats")
            if isinstance(manifest.get("norm_stats"), dict)
            else {}
        ),
        checkpoint_path=str(ckpt.checkpoint_path),
    )


def prepare_tensor_context(
    ckpt: LoadedCheckpoint, tile_ctx: Dict[str, Any]
) -> SurrogateTensorContext:
    spec = model_io_spec_from_checkpoint(ckpt)
    raw_count = int(spec.get("channel_axis", {}).get("raw_count") or 0)
    if raw_count <= 0 and ckpt.pca is not None:
        raw_count = int(getattr(ckpt.pca, "n_features_in_", 0) or 0)
    if raw_count <= 0:
        raw_count = int(spec.get("channel_axis", {}).get("model_count") or 0)
    if raw_count <= 0:
        raise RuntimeError(
            "Could not infer raw rail/POI/target channel count from checkpoint."
        )
    rail = np.asarray(tile_ctx["rail"], dtype=np.float32)[:raw_count]
    poi = np.asarray(tile_ctx["poi"], dtype=np.float32)[:raw_count]
    if rail.shape[0] < raw_count or poi.shape[0] < raw_count:
        raise ValueError(
            f"Tile has too few channels: rail={rail.shape[0]} poi={poi.shape[0]} required={raw_count}"
        )
    rail_model = project_pca(rail, ckpt.pca)
    poi_model = project_pca(poi, ckpt.pca)
    return SurrogateTensorContext(
        rail_raw=rail,
        poi_raw=poi,
        rail_model=rail_model,
        poi_model=poi_model,
        raw_channel_count=raw_count,
        model_channel_count=int(rail_model.shape[0]),
    )


class SurrogateInferenceAdapter:

    def __init__(self, ckpt: LoadedCheckpoint, *, device: Optional[str] = None):
        self.ckpt = ckpt
        self.device = device or "cpu"
        self.spec = model_io_spec_from_checkpoint(ckpt)
        self.output_activation = (
            ckpt.config.get("train", {}).get("output_activation")
            if isinstance(ckpt.config, dict)
            else None
        )

    @property
    def model_kind(self) -> str:
        return str(self.spec.get("model_kind") or "dense")

    def predict_model_space(
        self, ctx: SurrogateTensorContext, bus: np.ndarray
    ) -> np.ndarray:
        if self.model_kind == "dense":
            return self._predict_dense(ctx, bus)
        if self.model_kind == "slim_shared":
            return self._predict_slim_shared(ctx, bus)
        if self.model_kind == "slim_fixed":
            fixed = self.spec.get("output_contract", {}).get("fixed_channel_idx")
            if fixed is None:
                raise ValueError(
                    "slim_fixed checkpoint is missing fixed_channel_idx in model_io_spec."
                )
            return self._predict_slim_channel(ctx, bus, int(fixed))
        raise ValueError(f"Unsupported surrogate model_kind '{self.model_kind}'")

    def predict_all_components(
        self, ctx: SurrogateTensorContext, bus: np.ndarray
    ) -> np.ndarray:
        """Full per-component model output as (K, H, W), regardless of model_kind."""
        if self.model_kind == "dense":
            return self._predict_dense(ctx, bus)
        return self._predict_slim_shared(ctx, bus)

    def display_space(self, pred_model: np.ndarray, space: str) -> np.ndarray:
        if str(space or "raw") == "model":
            return pred_model.astype(np.float32, copy=False)
        if self.model_kind == "slim_fixed":
            fixed = self.spec.get("output_contract", {}).get("fixed_channel_idx")
            denorm = undo_norm_np(
                pred_model,
                self.ckpt.norm_stats,
                "target",
                int(fixed) if fixed is not None else None,
            )
        else:
            denorm = undo_norm_np(pred_model, self.ckpt.norm_stats, "target")
        if self.ckpt.pca is not None and denorm.shape[0] != int(
            getattr(self.ckpt.pca, "n_components_", denorm.shape[0])
        ):
            raise ValueError(
                "Cannot inverse-project partial slim output to raw space; load a full-output slim model or use model space."
            )
        return inverse_project_pca(denorm, self.ckpt.pca)

    def rail_display_space(self, ctx: SurrogateTensorContext, space: str) -> np.ndarray:
        if str(space or "raw") == "model":
            return (
                apply_norm_np(ctx.rail_model, self.ckpt.norm_stats, "rail")
                if self.ckpt.norm_stats
                else ctx.rail_model
            )
        return ctx.rail_raw.astype(np.float32, copy=False)

    def poi_display_space(self, ctx: SurrogateTensorContext, space: str) -> np.ndarray:
        if str(space or "raw") == "model":
            return ctx.poi_model.astype(np.float32, copy=False)
        return ctx.poi_raw.astype(np.float32, copy=False)

    def _forward_np(self, x_np: np.ndarray) -> np.ndarray:
        x = (
            torch.from_numpy(x_np.astype(np.float32, copy=False))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            pred = self.ckpt.model(x)
            pred = _apply_output_activation(pred, self.output_activation)
        return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _predict_dense(
        self, ctx: SurrogateTensorContext, bus: np.ndarray
    ) -> np.ndarray:
        bus_norm = apply_norm_np(bus, self.ckpt.norm_stats, "bus")
        rail_norm = apply_norm_np(ctx.rail_model, self.ckpt.norm_stats, "rail")
        poi_norm = apply_norm_np(ctx.poi_model, self.ckpt.norm_stats, "poi")
        return self._forward_np(np.concatenate([bus_norm, rail_norm, poi_norm], axis=0))

    def _predict_slim_shared(
        self, ctx: SurrogateTensorContext, bus: np.ndarray
    ) -> np.ndarray:
        preds = [
            self._predict_slim_channel(ctx, bus, idx)[0]
            for idx in range(ctx.model_channel_count)
        ]
        return np.stack(preds, axis=0).astype(np.float32)

    def _predict_slim_channel(
        self, ctx: SurrogateTensorContext, bus: np.ndarray, channel_idx: int
    ) -> np.ndarray:
        idx = int(np.clip(channel_idx, 0, ctx.model_channel_count - 1))
        bus_norm = apply_norm_np(bus, self.ckpt.norm_stats, "bus")
        rail_norm = apply_norm_np(
            ctx.rail_model[idx : idx + 1], self.ckpt.norm_stats, "rail", idx
        )
        poi_norm = apply_norm_np(
            ctx.poi_model[idx : idx + 1], self.ckpt.norm_stats, "poi", idx
        )
        pred = self._forward_np(np.concatenate([bus_norm, rail_norm, poi_norm], axis=0))
        if pred.shape[0] != 1:
            raise ValueError(
                f"Slim checkpoint returned {pred.shape[0]} channels; expected 1."
            )
        return pred
