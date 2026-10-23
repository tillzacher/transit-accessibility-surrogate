"""Masked loss + metric functions for the U-Net surrogate pipeline."""

from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
from torch import nn


def _prepare_mask(
    mask: Optional[torch.Tensor], target: torch.Tensor
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and target.shape[1] > 1:
        mask = mask.expand(-1, target.shape[1], -1, -1)
    return mask


def masked_reduction(diff: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        diff = diff * mask
        denom = mask.sum()
        return diff.sum() / torch.clamp(denom, min=1e-06)
    return diff.mean()


def _peak_weight_map(
    target: torch.Tensor, mask: Optional[torch.Tensor], *, alpha: float, power: float
) -> torch.Tensor:
    weights = 1.0 + float(alpha) * torch.pow(torch.abs(target), float(power))
    if mask is not None:
        masked_mean = (weights * mask).sum() / torch.clamp(mask.sum(), min=1e-06)
        weights = weights / torch.clamp(masked_mean, min=1e-06)
    else:
        weights = weights / torch.clamp(weights.mean(), min=1e-06)
    return weights


def _gradient_components(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (grad_y, grad_x)


def _gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    smooth_l1_beta: float,
) -> torch.Tensor:
    pred_dy, pred_dx = _gradient_components(pred)
    target_dy, target_dx = _gradient_components(target)
    diff_y = nn.functional.smooth_l1_loss(
        pred_dy, target_dy, reduction="none", beta=smooth_l1_beta
    )
    diff_x = nn.functional.smooth_l1_loss(
        pred_dx, target_dx, reduction="none", beta=smooth_l1_beta
    )
    mask_y = None
    mask_x = None
    if mask is not None:
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    loss_y = masked_reduction(diff_y, mask_y)
    loss_x = masked_reduction(diff_x, mask_x)
    return 0.5 * (loss_y + loss_x)


def masked_gaussian_nll(
    mu: torch.Tensor,
    target: torch.Tensor,
    logvar: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Masked heteroscedastic Gaussian negative log-likelihood."""
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (logvar + (target - mu) ** 2 * inv_var + math.log(2.0 * math.pi))
    return masked_reduction(nll, mask)


def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    loss_type: str = "mse",
    smooth_l1_beta: float = 1.0,
    peak_weight_alpha: float = 1.0,
    peak_weight_power: float = 1.0,
    grad_loss_weight: float = 0.1,
    logvar: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    lt = loss_type.lower()
    if lt in {"gaussian_nll", "nll"}:
        if logvar is None:
            raise ValueError(
                "loss_type='gaussian_nll' requires per-pixel logvar; set the model's output_uncertainty=True so the uncertainty head emits (μ, logσ²)."
            )
        return masked_gaussian_nll(pred, target, logvar, mask)
    if lt == "mse":
        diff = (pred - target) ** 2
        return masked_reduction(diff, mask)
    elif lt == "mae":
        diff = torch.abs(pred - target)
        return masked_reduction(diff, mask)
    elif lt == "smooth_l1":
        diff = nn.functional.smooth_l1_loss(
            pred, target, reduction="none", beta=smooth_l1_beta
        )
        return masked_reduction(diff, mask)
    elif lt == "weighted_smooth_l1":
        diff = nn.functional.smooth_l1_loss(
            pred, target, reduction="none", beta=smooth_l1_beta
        )
        diff = diff * _peak_weight_map(
            target, mask, alpha=peak_weight_alpha, power=peak_weight_power
        )
        return masked_reduction(diff, mask)
    elif lt in {"grad_match", "gradient_match"}:
        return _gradient_loss(pred, target, mask, smooth_l1_beta=smooth_l1_beta)
    elif lt in {"weighted_smooth_l1_grad", "combined_weighted_grad"}:
        value_loss = masked_loss(
            pred,
            target,
            mask,
            loss_type="weighted_smooth_l1",
            smooth_l1_beta=smooth_l1_beta,
            peak_weight_alpha=peak_weight_alpha,
            peak_weight_power=peak_weight_power,
            grad_loss_weight=grad_loss_weight,
        )
        grad_loss = _gradient_loss(pred, target, mask, smooth_l1_beta=smooth_l1_beta)
        return value_loss + float(grad_loss_weight) * grad_loss
    else:
        raise ValueError(f"Unsupported loss_type '{loss_type}'")


def masked_mae(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    diff = torch.abs(pred - target)
    return masked_reduction(diff, mask)


def anisotropic_tv(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dy = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dx + dy
