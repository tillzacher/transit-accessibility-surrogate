from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import inspect
import numpy as np
import torch
from modules.training_config import ModelConfig
from modules.unet_models import build_model

_COMPILED_PREFIX = "_orig_mod."


class PCAProjector:

    def __init__(self, components: np.ndarray, mean: np.ndarray):
        comps = np.asarray(components, dtype=np.float32)
        mean = np.asarray(mean, dtype=np.float32)
        if comps.ndim != 2:
            raise ValueError(f"PCA components must be 2D, got shape={comps.shape}")
        if mean.ndim != 1:
            raise ValueError(f"PCA mean must be 1D, got shape={mean.shape}")
        if comps.shape[1] != mean.shape[0]:
            raise ValueError(
                f"PCA components/features mismatch: {comps.shape[1]} vs {mean.shape[0]}"
            )
        self.components_ = comps
        self.mean_ = mean
        self.n_components_ = comps.shape[0]
        self.n_features_in_ = comps.shape[1]

    @classmethod
    def from_state(cls, state: Optional[Dict[str, Any]]) -> Optional["PCAProjector"]:
        if not state:
            return None
        if "components" in state:
            components = state["components"]
        else:
            components = state.get("components_")
        if "mean" in state:
            mean = state["mean"]
        else:
            mean = state.get("mean_")
        if components is None or mean is None:
            raise ValueError("PCA state must include components and mean.")
        return cls(components=np.asarray(components), mean=np.asarray(mean))

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) @ self.components_.T

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x @ self.components_ + self.mean_


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    pca: Optional[Any]
    norm_stats: Optional[Dict[str, Any]]
    manifest: Optional[Dict[str, Any]]
    training_manifest: Optional[Dict[str, Any]]
    synthesis_bindings: Optional[Dict[str, Any]]
    config: Dict[str, Any]
    checkpoint_path: Path


def _normalize_model_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if all((isinstance(key, str) and key.startswith(_COMPILED_PREFIX) for key in keys)):
        return {
            key[len(_COMPILED_PREFIX) :]: value for key, value in state_dict.items()
        }
    return state_dict


def load_model_checkpoint(
    checkpoint_path: Path,
    device: Optional[str] = None,
    *,
    weights_only: Optional[bool] = None,
) -> LoadedCheckpoint:
    checkpoint_path = Path(checkpoint_path)
    map_location = device or "cpu"
    if str(map_location).lower().startswith("cuda") and (not torch.cuda.is_available()):
        raise RuntimeError(
            f"Requested CUDA checkpoint load for {checkpoint_path}, but torch.cuda.is_available() is False. Use a CUDA-enabled environment or set the device explicitly to cpu."
        )
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False if weights_only is None else weights_only
    ckpt = torch.load(checkpoint_path, **load_kwargs)
    config = ckpt.get("config", {})
    model_cfg_dict = config.get("model", {})
    model_cfg = ModelConfig(**model_cfg_dict)
    in_ch = ckpt.get("in_channels") or model_cfg.in_channels
    out_ch = ckpt.get("out_channels") or model_cfg.out_channels
    if in_ch is None or out_ch is None:
        raise ValueError(
            "Checkpoint missing in/out channels; retrain with updated pipeline or provide them manually."
        )
    channel_info = ckpt.get("channel_info")
    model = build_model(model_cfg, in_ch, out_ch, channel_info=channel_info)
    model_state = _normalize_model_state_dict(ckpt["model_state"])
    model.load_state_dict(model_state, strict=True)
    model.to(map_location)
    model.eval()
    pca_state = ckpt.get("pca_state")
    pca = PCAProjector.from_state(pca_state)
    norm_stats = ckpt.get("norm_stats")
    manifest = ckpt.get("model_manifest")
    training_manifest = ckpt.get("manifest")
    synthesis_bindings = ckpt.get("synthesis_bindings")
    if synthesis_bindings is None and isinstance(manifest, dict):
        synthesis_bindings = manifest.get("synthesis_bindings")
    return LoadedCheckpoint(
        model=model,
        pca=pca,
        norm_stats=norm_stats,
        manifest=manifest,
        training_manifest=training_manifest,
        synthesis_bindings=synthesis_bindings,
        config=config,
        checkpoint_path=checkpoint_path,
    )


__all__ = [
    "PCAProjector",
    "LoadedCheckpoint",
    "load_model_checkpoint",
]
