"""Differentiable bus network renderer."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.tile_graph_spec import TileGraphSpec


class Renderer(nn.Module):
    """Differentiable renderer: (alpha, beta) -> 2-channel bus tensor."""

    def __init__(
        self,
        spec: TileGraphSpec,
        soft_cap_max: float | None = 1.0,
        soft_cap_beta: float = 10.0,
        edge_gate_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.H = spec.H
        self.W = spec.W
        self.soft_cap_max = soft_cap_max
        self.soft_cap_beta = soft_cap_beta
        self.edge_gate_mode = self._resolve_edge_gate_mode(spec, edge_gate_mode)
        self.register_buffer(
            "stop_pixel_idx", torch.as_tensor(spec.stop_pixel_idx, dtype=torch.long)
        )
        self.register_buffer(
            "stop_stop_idx", torch.as_tensor(spec.stop_stop_idx, dtype=torch.long)
        )
        self.register_buffer(
            "stop_weight", torch.as_tensor(spec.stop_weight, dtype=torch.float32)
        )
        self.register_buffer(
            "edge_pixel_idx", torch.as_tensor(spec.edge_pixel_idx, dtype=torch.long)
        )
        self.register_buffer(
            "edge_edge_idx", torch.as_tensor(spec.edge_edge_idx, dtype=torch.long)
        )
        self.register_buffer(
            "edge_weight", torch.as_tensor(spec.edge_weight, dtype=torch.float32)
        )
        self.register_buffer("edge_u", torch.as_tensor(spec.edge_u, dtype=torch.long))
        self.register_buffer("edge_v", torch.as_tensor(spec.edge_v, dtype=torch.long))

    def forward(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Render bus tensor from unconstrained logits."""
        z = torch.sigmoid(alpha)
        w = torch.nn.functional.softplus(beta)
        return self.render_from_wz(w, z)

    def render_from_wz(
        self, w: torch.Tensor, z: torch.Tensor, *, edge_gate_mode: str | None = None
    ) -> torch.Tensor:
        """Render bus tensor from already-activated stop/edge values."""
        S = self._render_stops(z)
        I = self._render_edges(w, z, edge_gate_mode=edge_gate_mode)
        return torch.stack([S, I], dim=0)

    def _render_stops(self, z: torch.Tensor) -> torch.Tensor:
        """Scatter Gaussian stop blobs onto flat pixel grid, then reshape."""
        S_flat = torch.zeros(self.H * self.W, device=z.device, dtype=z.dtype)
        if self.stop_pixel_idx.numel() == 0:
            return S_flat.view(self.H, self.W)
        contrib = z[self.stop_stop_idx] * self.stop_weight
        S_flat.scatter_add_(0, self.stop_pixel_idx, contrib)
        S = S_flat.view(self.H, self.W)
        return self._apply_soft_cap(S)

    def _render_edges(
        self, w: torch.Tensor, z: torch.Tensor, *, edge_gate_mode: str | None = None
    ) -> torch.Tensor:
        """Scatter Gaussian edge blobs onto flat pixel grid with optional endpoint gating."""
        I_flat = torch.zeros(self.H * self.W, device=w.device, dtype=w.dtype)
        if self.edge_pixel_idx.numel() == 0:
            return I_flat.view(self.H, self.W)
        gate_mode = self.edge_gate_mode if edge_gate_mode is None else edge_gate_mode
        if gate_mode == "product":
            gate = z[self.edge_u] * z[self.edge_v]
            w_eff = w * gate
        elif gate_mode == "none":
            w_eff = w
        else:
            raise ValueError(
                f"Unsupported edge_gate_mode '{gate_mode}'. Expected 'none' or 'product'."
            )
        contrib = w_eff[self.edge_edge_idx] * self.edge_weight
        I_flat.scatter_add_(0, self.edge_pixel_idx, contrib)
        I = I_flat.view(self.H, self.W)
        return self._apply_soft_cap(I)

    @staticmethod
    def _resolve_edge_gate_mode(spec: TileGraphSpec, edge_gate_mode: str) -> str:
        mode = str(edge_gate_mode).strip().lower()
        if mode in {"none", "product"}:
            return mode
        if mode != "auto":
            raise ValueError(
                f"Unsupported edge_gate_mode '{edge_gate_mode}'. Expected 'auto', 'none', or 'product'."
            )
        spec_mode = str(getattr(spec, "candidate_edge_mode", "")).strip().lower()
        if spec_mode == "gtfs_pairs":
            return "product"
        return "none"

    def _apply_soft_cap(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth approximation of min(x, cap) to preserve gradients."""
        if self.soft_cap_max is None:
            return x
        cap = torch.as_tensor(self.soft_cap_max, dtype=x.dtype, device=x.device)
        offset = F.softplus(-cap, beta=self.soft_cap_beta)
        return x - F.softplus(x - cap, beta=self.soft_cap_beta) + offset


__all__ = ["Renderer"]
