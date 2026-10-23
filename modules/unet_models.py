"""U-Net family architectures + builders for the surrogate pipeline."""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from modules.training_config import ModelConfig


def make_activation(name: str, negative_slope: float = 0.01):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish()
    raise ValueError(f"Unsupported activation '{name}'")


def make_norm(name: str, channels: int, groups: int = 8):
    name = name.lower()
    if name == "batch":
        return nn.BatchNorm2d(channels)
    if name == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if name == "group":
        target = max(1, min(groups, channels))
        g = math.gcd(channels, target) or 1
        return nn.GroupNorm(g, channels)
    if name == "layer":
        return nn.GroupNorm(1, channels)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm '{name}'")


class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.0,
        activation="relu",
        activation_negative_slope=0.01,
        norm="batch",
        norm_groups=8,
        residual=False,
    ):
        super().__init__()
        act = make_activation(activation, negative_slope=activation_negative_slope)
        norm1 = make_norm(norm, out_channels, groups=norm_groups)
        norm2 = make_norm(norm, out_channels, groups=norm_groups)
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm1,
            act,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm2,
            make_activation(activation, negative_slope=activation_negative_slope),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)
        self.residual = residual

    def forward(self, x):
        out = self.net(x)
        if self.residual and out.shape == x.shape:
            out = out + x
        return out


class SqueezeExcite2d(nn.Module):

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduction = max(1, int(reduction))
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.net(self.pool(x))
        return x * scale


class AttentionGate2d(nn.Module):

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ):
        super().__init__()
        inter_channels = inter_channels or max(
            1, min(gate_channels, skip_channels) // 2
        )
        self.gate_proj = nn.Conv2d(
            gate_channels, inter_channels, kernel_size=1, bias=False
        )
        self.skip_proj = nn.Conv2d(
            skip_channels, inter_channels, kernel_size=1, bias=False
        )
        self.score = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        attn = self.score(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * attn


class Down(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        dropout=0.0,
        activation="relu",
        activation_negative_slope=0.01,
        norm="batch",
        norm_groups=8,
        residual=False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                dropout=dropout,
                activation=activation,
                activation_negative_slope=activation_negative_slope,
                norm=norm,
                norm_groups=norm_groups,
                residual=residual,
            ),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):

    def __init__(
        self,
        x_channels,
        skip_channels,
        out_channels,
        up_mode="bilinear",
        dropout=0.0,
        activation="relu",
        activation_negative_slope=0.01,
        norm="batch",
        norm_groups=8,
        residual=False,
        skip_attention="none",
        decoder_attention="none",
        attention_reduction=8,
    ):
        super().__init__()
        if up_mode == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_out_ch = x_channels
        elif up_mode == "nearest":
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            up_out_ch = x_channels
        elif up_mode == "convtranspose":
            self.up = nn.ConvTranspose2d(
                x_channels, x_channels // 2, kernel_size=2, stride=2
            )
            up_out_ch = x_channels // 2
        else:
            raise ValueError(f"Unsupported up_mode '{up_mode}'")
        if skip_attention not in {"none", "gate"}:
            raise ValueError(f"Unsupported skip_attention '{skip_attention}'")
        if decoder_attention not in {"none", "se"}:
            raise ValueError(f"Unsupported decoder_attention '{decoder_attention}'")
        self.skip_gate = (
            AttentionGate2d(up_out_ch, skip_channels)
            if skip_attention == "gate"
            else None
        )
        self.conv = DoubleConv(
            up_out_ch + skip_channels,
            out_channels,
            dropout=dropout,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            norm=norm,
            norm_groups=norm_groups,
            residual=residual,
        )
        self.decoder_attn = (
            SqueezeExcite2d(out_channels, reduction=attention_reduction)
            if decoder_attention == "se"
            else nn.Identity()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        if self.skip_gate is not None:
            x2 = self.skip_gate(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return self.decoder_attn(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UncertaintyOutConv(nn.Module):
    """Aleatoric-uncertainty output head: emits per-pixel mean μ AND
    log-variance from two parallel 1×1 convs."""

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ):
        super().__init__()
        self.mu = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.logvar = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def forward(self, x):
        mu = self.mu(x)
        logvar = torch.clamp(self.logvar(x), self.logvar_min, self.logvar_max)
        return (mu, logvar)


def _build_out_head(
    in_channels: int, out_channels: int, output_uncertainty: bool = False
) -> nn.Module:
    """Output head factory: deterministic OutConv, or UncertaintyOutConv when uncertainty
    is requested. Keeps the per-model wiring identical across architectures."""
    return (
        UncertaintyOutConv(in_channels, out_channels)
        if output_uncertainty
        else OutConv(in_channels, out_channels)
    )


def _head_output(out, return_uncertainty: bool):
    """Normalize an output-head result to the caller's contract."""
    if isinstance(out, tuple):
        mu, logvar = out
        return (mu, logvar) if return_uncertainty else mu
    return out


class BottleneckMHSA(nn.Module):
    """Multi-head self-attention over the flattened bottleneck feature map."""

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 4,
        num_blocks: int = 1,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"BottleneckMHSA: channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        hidden = max(1, int(channels * mlp_ratio))
        self.blocks = nn.ModuleList(
            (
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(channels),
                        "attn": nn.MultiheadAttention(
                            channels, num_heads, dropout=dropout, batch_first=True
                        ),
                        "norm2": nn.LayerNorm(channels),
                        "mlp": nn.Sequential(
                            nn.Linear(channels, hidden),
                            nn.GELU(),
                            nn.Linear(hidden, channels),
                        ),
                    }
                )
                for _ in range(max(1, int(num_blocks)))
            )
        )

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            hn = blk["norm1"](t)
            attn_out, _ = blk["attn"](hn, hn, hn, need_weights=False)
            t = t + attn_out
            t = t + blk["mlp"](blk["norm2"](t))
        return t.transpose(1, 2).reshape(b, c, h, w)


def _build_bottleneck_attn(
    channels: int,
    kind: str,
    *,
    attention_reduction: int = 8,
    num_heads: int = 4,
    num_blocks: int = 1,
) -> nn.Module:
    """Bottleneck attention factory: SE channel-gating, MHSA global context, or Identity."""
    if kind == "se":
        return SqueezeExcite2d(channels, reduction=attention_reduction)
    if kind == "mhsa":
        return BottleneckMHSA(channels, num_heads=num_heads, num_blocks=num_blocks)
    if kind in ("none", None):
        return nn.Identity()
    raise ValueError(f"Unsupported bottleneck_attention '{kind}'")


class CrossAttentionBottleneckFusion(nn.Module):
    """Cross-attention fusion of the three bottleneck streams (triple-stream only)."""

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"CrossAttentionBottleneckFusion: channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        hidden = max(1, int(channels * mlp_ratio))
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )

    def forward(
        self, bus: torch.Tensor, rail: torch.Tensor, poi: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = bus.shape
        q = bus.flatten(2).transpose(1, 2)
        ctx = torch.cat(
            [rail.flatten(2).transpose(1, 2), poi.flatten(2).transpose(1, 2)], dim=1
        )
        attn_out, _ = self.attn(
            self.norm_q(q), self.norm_kv(ctx), self.norm_kv(ctx), need_weights=False
        )
        q = q + attn_out
        q = q + self.mlp(self.norm_mlp(q))
        return q.transpose(1, 2).reshape(b, c, h, w)


class InputProjection(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int],
        *,
        norm: str,
        norm_groups: int,
        activation: str,
        activation_negative_slope: float,
    ):
        super().__init__()
        out_channels = int(out_channels or in_channels)
        self.out_channels = out_channels
        if out_channels == in_channels:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm(norm, out_channels, groups=norm_groups),
                make_activation(activation, negative_slope=activation_negative_slope),
            )

    def forward(self, x):
        return self.net(x)


def _make_stem_projection(
    stream_in: int,
    base_channels: int,
    input_projection_channels: Optional[int],
    *,
    input_stem: str = "project",
    norm: str,
    norm_groups: int,
    activation: str,
    activation_negative_slope: float,
):
    """Per-stream input stem. Returns (projection_module, stem_out_channels) so the caller
    sizes its first DoubleConv accordingly."""
    if input_projection_channels is not None:
        proj_out = int(input_projection_channels)
    elif input_stem == "identity":
        proj_out = int(stream_in)
    else:
        proj_out = int(base_channels)
    proj = InputProjection(
        int(stream_in),
        proj_out,
        norm=norm,
        norm_groups=norm_groups,
        activation=activation,
        activation_negative_slope=activation_negative_slope,
    )
    return (proj, proj_out)


class ASPP(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        norm: str = "batch",
        norm_groups: int = 8,
    ):
        super().__init__()
        valid_dilations = [max(1, int(d)) for d in dilations] or [1]
        branch_channels = max(1, out_channels // len(valid_dilations))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        branch_channels,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        bias=False,
                    ),
                    make_norm(norm, branch_channels, groups=norm_groups),
                    make_activation(
                        activation, negative_slope=activation_negative_slope
                    ),
                )
                for d in valid_dilations
            ]
        )
        fused_in = branch_channels * len(valid_dilations)
        fuse_layers = [
            nn.Conv2d(fused_in, out_channels, kernel_size=1, bias=False),
            make_norm(norm, out_channels, groups=norm_groups),
            make_activation(activation, negative_slope=activation_negative_slope),
        ]
        if dropout > 0:
            fuse_layers.append(nn.Dropout2d(p=dropout))
        self.fuse = nn.Sequential(*fuse_layers)

    def forward(self, x):
        branches = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(branches, dim=1))


class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        base_channels: int = 32,
        depth: int = 4,
        channel_mult: int = 2,
        up_mode: str = "bilinear",
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        norm: str = "batch",
        norm_groups: int = 8,
        residual_blocks: bool = False,
        skip_attention: str = "none",
        bottleneck_attention: str = "none",
        decoder_attention: str = "none",
        attention_reduction: int = 8,
        bottleneck_attention_heads: int = 4,
        bottleneck_attention_blocks: int = 1,
        output_uncertainty: bool = False,
    ):
        super().__init__()
        if bottleneck_attention not in {"none", "se", "mhsa"}:
            raise ValueError(
                f"Unsupported bottleneck_attention '{bottleneck_attention}'"
            )
        self.output_uncertainty = bool(output_uncertainty)
        filters = [base_channels * channel_mult**i for i in range(depth)]
        self.inc = DoubleConv(
            in_channels,
            filters[0],
            dropout=dropout,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            norm=norm,
            norm_groups=norm_groups,
            residual=residual_blocks,
        )
        self.down_blocks = nn.ModuleList()
        for i in range(1, depth):
            self.down_blocks.append(
                Down(
                    filters[i - 1],
                    filters[i],
                    dropout=dropout,
                    activation=activation,
                    activation_negative_slope=activation_negative_slope,
                    norm=norm,
                    norm_groups=norm_groups,
                    residual=residual_blocks,
                )
            )
        self.up_blocks = nn.ModuleList()
        rev_filters = list(reversed(filters))
        for i in range(depth - 1):
            self.up_blocks.append(
                Up(
                    rev_filters[i],
                    rev_filters[i + 1],
                    rev_filters[i + 1],
                    up_mode=up_mode,
                    dropout=dropout,
                    activation=activation,
                    activation_negative_slope=activation_negative_slope,
                    norm=norm,
                    norm_groups=norm_groups,
                    residual=residual_blocks,
                    skip_attention=skip_attention,
                    decoder_attention=decoder_attention,
                    attention_reduction=attention_reduction,
                )
            )
        self.bottleneck_attn = _build_bottleneck_attn(
            filters[-1],
            bottleneck_attention,
            attention_reduction=attention_reduction,
            num_heads=bottleneck_attention_heads,
            num_blocks=bottleneck_attention_blocks,
        )
        self.outc = _build_out_head(filters[0], out_channels, output_uncertainty)

    def forward(self, x, return_uncertainty: bool = False):
        xs = [self.inc(x)]
        for down in self.down_blocks:
            xs.append(down(xs[-1]))
        x = self.bottleneck_attn(xs[-1])
        for idx, up in enumerate(self.up_blocks):
            x = up(x, xs[-idx - 2])
        return _head_output(self.outc(x), return_uncertainty)


class ResUNetASPP(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        base_channels: int = 32,
        depth: int = 4,
        channel_mult: int = 2,
        up_mode: str = "bilinear",
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        norm: str = "batch",
        norm_groups: int = 8,
        residual_blocks: bool = True,
        skip_attention: str = "none",
        bottleneck_attention: str = "none",
        decoder_attention: str = "none",
        attention_reduction: int = 8,
        bottleneck_attention_heads: int = 4,
        bottleneck_attention_blocks: int = 1,
        output_uncertainty: bool = False,
        input_projection_channels: Optional[int] = None,
        aspp_dilations: Sequence[int] = (1, 2, 4, 8),
        aspp_dropout: float = 0.0,
    ):
        super().__init__()
        self.output_uncertainty = bool(output_uncertainty)
        proj_out = int(input_projection_channels or in_channels)
        self.input_proj = InputProjection(
            in_channels,
            proj_out,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
        )
        self.core = UNet(
            in_channels=proj_out,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            channel_mult=channel_mult,
            up_mode=up_mode,
            dropout=dropout,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            norm=norm,
            norm_groups=norm_groups,
            residual_blocks=residual_blocks,
            skip_attention=skip_attention,
            bottleneck_attention=bottleneck_attention,
            decoder_attention=decoder_attention,
            attention_reduction=attention_reduction,
            bottleneck_attention_heads=bottleneck_attention_heads,
            bottleneck_attention_blocks=bottleneck_attention_blocks,
            output_uncertainty=output_uncertainty,
        )
        bottleneck_channels = base_channels * channel_mult ** (depth - 1)
        self.core.bottleneck_attn = nn.Sequential(
            ASPP(
                bottleneck_channels,
                bottleneck_channels,
                dilations=aspp_dilations,
                dropout=aspp_dropout,
                activation=activation,
                activation_negative_slope=activation_negative_slope,
                norm=norm,
                norm_groups=norm_groups,
            ),
            self.core.bottleneck_attn,
        )

    def forward(self, x, return_uncertainty: bool = False):
        return self.core(self.input_proj(x), return_uncertainty=return_uncertainty)


class DualStreamResUNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bus_in_channels: int,
        context_in_channels: Optional[int] = None,
        base_channels: int = 32,
        depth: int = 4,
        channel_mult: int = 2,
        up_mode: str = "bilinear",
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        norm: str = "batch",
        norm_groups: int = 8,
        residual_blocks: bool = True,
        skip_attention: str = "none",
        bottleneck_attention: str = "none",
        decoder_attention: str = "none",
        attention_reduction: int = 8,
        bottleneck_attention_heads: int = 4,
        bottleneck_attention_blocks: int = 1,
        output_uncertainty: bool = False,
        input_projection_channels: Optional[int] = None,
        input_stem: str = "project",
        aspp_dilations: Sequence[int] = (1, 2, 4, 8),
        aspp_dropout: float = 0.0,
        fusion: str = "concat",
        share_encoder: bool = False,
    ):
        super().__init__()
        self.output_uncertainty = bool(output_uncertainty)
        if bus_in_channels <= 0 or bus_in_channels >= in_channels:
            raise ValueError(
                f"dual_stream_resunet requires 0 < bus_in_channels < total_in_channels; got bus={bus_in_channels}, total={in_channels}"
            )
        context_in_channels = int(context_in_channels or in_channels - bus_in_channels)
        if (
            context_in_channels <= 0
            or bus_in_channels + context_in_channels > in_channels
        ):
            raise ValueError(
                f"dual_stream_resunet invalid context split: bus={bus_in_channels}, context={context_in_channels}, total={in_channels}"
            )
        if fusion not in {"concat", "sum"}:
            raise ValueError(f"Unsupported dual_stream_fusion '{fusion}'")
        if bottleneck_attention not in {"none", "se", "mhsa"}:
            raise ValueError(
                f"Unsupported bottleneck_attention '{bottleneck_attention}'"
            )
        filters = [base_channels * channel_mult**i for i in range(depth)]
        self.bus_in_channels = int(bus_in_channels)
        self.context_in_channels = int(context_in_channels)
        _stem_kw = dict(
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
        )
        self.bus_proj, bus_stem = _make_stem_projection(
            self.bus_in_channels,
            base_channels,
            input_projection_channels,
            input_stem=input_stem,
            **_stem_kw,
        )
        self.context_proj, ctx_stem = _make_stem_projection(
            self.context_in_channels,
            base_channels,
            input_projection_channels,
            input_stem=input_stem,
            **_stem_kw,
        )
        self.share_encoder = bool(share_encoder)
        self.fusion = fusion
        self.bus_inc = DoubleConv(
            bus_stem,
            filters[0],
            dropout=dropout,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            norm=norm,
            norm_groups=norm_groups,
            residual=residual_blocks,
        )
        self.ctx_inc = DoubleConv(
            ctx_stem,
            filters[0],
            dropout=dropout,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            norm=norm,
            norm_groups=norm_groups,
            residual=residual_blocks,
        )
        self.shared_down_blocks = None
        if self.share_encoder:
            self.shared_down_blocks = nn.ModuleList()
            for i in range(1, depth):
                self.shared_down_blocks.append(
                    Down(
                        filters[i - 1],
                        filters[i],
                        dropout=dropout,
                        activation=activation,
                        activation_negative_slope=activation_negative_slope,
                        norm=norm,
                        norm_groups=norm_groups,
                        residual=residual_blocks,
                    )
                )
            self.bus_down_blocks = None
            self.ctx_down_blocks = None
        else:
            self.bus_down_blocks = nn.ModuleList()
            self.ctx_down_blocks = nn.ModuleList()
            for i in range(1, depth):
                kwargs = dict(
                    dropout=dropout,
                    activation=activation,
                    activation_negative_slope=activation_negative_slope,
                    norm=norm,
                    norm_groups=norm_groups,
                    residual=residual_blocks,
                )
                self.bus_down_blocks.append(Down(filters[i - 1], filters[i], **kwargs))
                self.ctx_down_blocks.append(Down(filters[i - 1], filters[i], **kwargs))
        rev_filters = list(reversed(filters))
        fusion_factor = 2 if fusion == "concat" else 1
        bottleneck_channels = filters[-1] * fusion_factor
        self.bottleneck = ASPP(
            bottleneck_channels,
            filters[-1],
            dilations=aspp_dilations,
            dropout=aspp_dropout,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            norm=norm,
            norm_groups=norm_groups,
        )
        self.bottleneck_attn = _build_bottleneck_attn(
            filters[-1],
            bottleneck_attention,
            attention_reduction=attention_reduction,
            num_heads=bottleneck_attention_heads,
            num_blocks=bottleneck_attention_blocks,
        )
        self.skip_fuse = nn.ModuleList()
        for skip_ch in filters[:-1]:
            if fusion == "concat":
                self.skip_fuse.append(
                    nn.Sequential(
                        nn.Conv2d(skip_ch * 2, skip_ch, kernel_size=1, bias=False),
                        make_norm(norm, skip_ch, groups=norm_groups),
                        make_activation(
                            activation, negative_slope=activation_negative_slope
                        ),
                    )
                )
            else:
                self.skip_fuse.append(nn.Identity())
        self.up_blocks = nn.ModuleList()
        for i in range(depth - 1):
            self.up_blocks.append(
                Up(
                    rev_filters[i],
                    rev_filters[i + 1],
                    rev_filters[i + 1],
                    up_mode=up_mode,
                    dropout=dropout,
                    activation=activation,
                    activation_negative_slope=activation_negative_slope,
                    norm=norm,
                    norm_groups=norm_groups,
                    residual=residual_blocks,
                    skip_attention=skip_attention,
                    decoder_attention=decoder_attention,
                    attention_reduction=attention_reduction,
                )
            )
        self.outc = _build_out_head(filters[0], out_channels, output_uncertainty)

    def _encode(
        self,
        x: torch.Tensor,
        proj: nn.Module,
        inc: nn.Module,
        down_blocks: Sequence[nn.Module],
    ) -> List[torch.Tensor]:
        xs = [inc(proj(x))]
        for down in down_blocks:
            xs.append(down(xs[-1]))
        return xs

    def forward(self, x, return_uncertainty: bool = False):
        expected = self.bus_in_channels + self.context_in_channels
        if x.shape[1] != expected:
            raise ValueError(
                f"DualStreamResUNet.forward: input has {x.shape[1]} channels, expected {expected} (bus={self.bus_in_channels}+context={self.context_in_channels}); check channel ordering."
            )
        bus = x[:, : self.bus_in_channels]
        context = x[
            :, self.bus_in_channels : self.bus_in_channels + self.context_in_channels
        ]
        if self.shared_down_blocks is not None:
            bus_down_blocks = self.shared_down_blocks
            ctx_down_blocks = self.shared_down_blocks
        else:
            bus_down_blocks = self.bus_down_blocks
            ctx_down_blocks = self.ctx_down_blocks
        bus_xs = self._encode(bus, self.bus_proj, self.bus_inc, bus_down_blocks)
        ctx_xs = self._encode(context, self.context_proj, self.ctx_inc, ctx_down_blocks)
        if self.fusion == "concat":
            x = torch.cat([bus_xs[-1], ctx_xs[-1]], dim=1)
        else:
            x = bus_xs[-1] + ctx_xs[-1]
        x = self.bottleneck_attn(self.bottleneck(x))
        fused_skips: List[torch.Tensor] = []
        for bus_skip, ctx_skip, fuse in zip(bus_xs[:-1], ctx_xs[:-1], self.skip_fuse):
            if self.fusion == "concat":
                fused_skips.append(fuse(torch.cat([bus_skip, ctx_skip], dim=1)))
            else:
                fused_skips.append(fuse(bus_skip + ctx_skip))
        for idx, up in enumerate(self.up_blocks):
            x = up(x, fused_skips[-idx - 1])
        return _head_output(self.outc(x), return_uncertainty)


def resolve_dual_stream_channels(
    model_cfg: ModelConfig,
    in_channels: int,
    channel_info: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    bus_in = model_cfg.bus_input_channels
    if bus_in is None and channel_info:
        bus_in = channel_info.get("bus_channels")
    if bus_in is None:
        raise ValueError(
            "dual_stream_resunet requires bus_input_channels or channel_info['bus_channels']"
        )
    bus_in = int(bus_in)
    context_in = model_cfg.context_input_channels
    if context_in is None and channel_info:
        context_in = channel_info.get("rail_channels", 0) + channel_info.get(
            "poi_channels", 0
        )
    if context_in is None:
        context_in = int(in_channels) - bus_in
    context_in = int(context_in)
    if bus_in + context_in != int(in_channels):
        raise ValueError(
            f"dual_stream_resunet channel split mismatch: bus={bus_in}, context={context_in}, total={in_channels}"
        )
    return (bus_in, context_in)


def resolve_triple_stream_channels(
    model_cfg: ModelConfig,
    in_channels: int,
    channel_info: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, int]:
    bus_in = model_cfg.bus_input_channels
    if bus_in is None and channel_info:
        bus_in = channel_info.get("bus_channels")
    rail_in = model_cfg.rail_input_channels
    if rail_in is None and channel_info:
        rail_in = channel_info.get("rail_channels")
    poi_in = model_cfg.poi_input_channels
    if poi_in is None and channel_info:
        poi_in = channel_info.get("poi_channels")
    if bus_in is None or rail_in is None or poi_in is None:
        raise ValueError(
            f"triple_stream_resunet requires bus/rail/poi channel counts via model_cfg or channel_info; got bus={bus_in}, rail={rail_in}, poi={poi_in}"
        )
    bus_in, rail_in, poi_in = (int(bus_in), int(rail_in), int(poi_in))
    if bus_in + rail_in + poi_in != int(in_channels):
        raise ValueError(
            f"triple_stream_resunet channel split mismatch: bus={bus_in}, rail={rail_in}, poi={poi_in}, total={in_channels}"
        )
    return (bus_in, rail_in, poi_in)


class TripleStreamResUNet(nn.Module):
    """Three-stream ResUNet that encodes bus (2ch), rail (108ch) and POI (108ch) independently."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bus_in_channels: int,
        rail_in_channels: int,
        poi_in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        channel_mult: int = 2,
        up_mode: str = "bilinear",
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        norm: str = "batch",
        norm_groups: int = 8,
        residual_blocks: bool = True,
        skip_attention: str = "none",
        bottleneck_attention: str = "none",
        decoder_attention: str = "none",
        attention_reduction: int = 8,
        bottleneck_attention_heads: int = 4,
        bottleneck_attention_blocks: int = 1,
        output_uncertainty: bool = False,
        input_projection_channels: Optional[int] = None,
        input_stem: str = "project",
        aspp_dilations: Sequence[int] = (1, 2, 4, 8),
        aspp_dropout: float = 0.0,
        fusion: str = "concat",
        share_rail_poi_encoder: bool = False,
    ):
        super().__init__()
        self.output_uncertainty = bool(output_uncertainty)
        total = bus_in_channels + rail_in_channels + poi_in_channels
        if total != int(in_channels):
            raise ValueError(
                f"TripleStreamResUNet channel split mismatch: bus={bus_in_channels}, rail={rail_in_channels}, poi={poi_in_channels}, total={in_channels}"
            )
        if fusion not in {"concat", "sum", "xattn"}:
            raise ValueError(f"Unsupported triple_stream_fusion '{fusion}'")
        self.bus_in_channels = int(bus_in_channels)
        self.rail_in_channels = int(rail_in_channels)
        self.poi_in_channels = int(poi_in_channels)
        self.fusion = fusion
        self.share_rail_poi_encoder = bool(share_rail_poi_encoder)
        filters = [base_channels * channel_mult**i for i in range(depth)]
        conv_kwargs = dict(
            dropout=dropout,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            norm=norm,
            norm_groups=norm_groups,
            residual=residual_blocks,
        )
        norm_kwargs = dict(
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
        )
        self.bus_proj, bus_stem = _make_stem_projection(
            self.bus_in_channels,
            base_channels,
            input_projection_channels,
            input_stem=input_stem,
            **norm_kwargs,
        )
        self.rail_proj, rail_stem = _make_stem_projection(
            self.rail_in_channels,
            base_channels,
            input_projection_channels,
            input_stem=input_stem,
            **norm_kwargs,
        )
        self.poi_proj, poi_stem = _make_stem_projection(
            self.poi_in_channels,
            base_channels,
            input_projection_channels,
            input_stem=input_stem,
            **norm_kwargs,
        )
        self.bus_inc = DoubleConv(bus_stem, filters[0], **conv_kwargs)
        self.rail_inc = DoubleConv(rail_stem, filters[0], **conv_kwargs)
        self.poi_inc = (
            None
            if share_rail_poi_encoder
            else DoubleConv(poi_stem, filters[0], **conv_kwargs)
        )
        self.bus_down_blocks = nn.ModuleList()
        self.rail_down_blocks = nn.ModuleList()
        self.poi_down_blocks = None if share_rail_poi_encoder else nn.ModuleList()
        for i in range(1, depth):
            self.bus_down_blocks.append(Down(filters[i - 1], filters[i], **conv_kwargs))
            self.rail_down_blocks.append(
                Down(filters[i - 1], filters[i], **conv_kwargs)
            )
            if not share_rail_poi_encoder:
                self.poi_down_blocks.append(
                    Down(filters[i - 1], filters[i], **conv_kwargs)
                )
        n_streams = 3
        bottleneck_channels = filters[-1] * (n_streams if fusion == "concat" else 1)
        self.xattn_fusion = (
            CrossAttentionBottleneckFusion(
                filters[-1], num_heads=bottleneck_attention_heads
            )
            if fusion == "xattn"
            else None
        )
        self.bottleneck = ASPP(
            bottleneck_channels,
            filters[-1],
            dilations=aspp_dilations,
            dropout=aspp_dropout,
            activation=activation,
            activation_negative_slope=activation_negative_slope,
            norm=norm,
            norm_groups=norm_groups,
        )
        self.bottleneck_attn = _build_bottleneck_attn(
            filters[-1],
            bottleneck_attention,
            attention_reduction=attention_reduction,
            num_heads=bottleneck_attention_heads,
            num_blocks=bottleneck_attention_blocks,
        )
        self.skip_fuse = nn.ModuleList()
        for skip_ch in filters[:-1]:
            if fusion == "concat":
                self.skip_fuse.append(
                    nn.Sequential(
                        nn.Conv2d(
                            skip_ch * n_streams, skip_ch, kernel_size=1, bias=False
                        ),
                        make_norm(norm, skip_ch, groups=norm_groups),
                        make_activation(
                            activation, negative_slope=activation_negative_slope
                        ),
                    )
                )
            else:
                self.skip_fuse.append(nn.Identity())
        rev_filters = list(reversed(filters))
        self.up_blocks = nn.ModuleList()
        for i in range(depth - 1):
            self.up_blocks.append(
                Up(
                    rev_filters[i],
                    rev_filters[i + 1],
                    rev_filters[i + 1],
                    up_mode=up_mode,
                    dropout=dropout,
                    activation=activation,
                    activation_negative_slope=activation_negative_slope,
                    norm=norm,
                    norm_groups=norm_groups,
                    residual=residual_blocks,
                    skip_attention=skip_attention,
                    decoder_attention=decoder_attention,
                    attention_reduction=attention_reduction,
                )
            )
        self.outc = _build_out_head(filters[0], out_channels, output_uncertainty)

    def _encode(
        self,
        x: torch.Tensor,
        proj: nn.Module,
        inc: nn.Module,
        down_blocks: Sequence[nn.Module],
    ) -> List[torch.Tensor]:
        xs = [inc(proj(x))]
        for down in down_blocks:
            xs.append(down(xs[-1]))
        return xs

    def forward(self, x, return_uncertainty: bool = False):
        expected = self.bus_in_channels + self.rail_in_channels + self.poi_in_channels
        if x.shape[1] != expected:
            raise ValueError(
                f"TripleStreamResUNet.forward: input has {x.shape[1]} channels, expected {expected} (bus={self.bus_in_channels}+rail={self.rail_in_channels}+poi={self.poi_in_channels}); check channel ordering/grouping."
            )
        bus = x[:, : self.bus_in_channels]
        rail = x[:, self.bus_in_channels : self.bus_in_channels + self.rail_in_channels]
        poi = x[:, self.bus_in_channels + self.rail_in_channels :]
        poi_inc = self.rail_inc if self.share_rail_poi_encoder else self.poi_inc
        poi_down = (
            self.rail_down_blocks
            if self.share_rail_poi_encoder
            else self.poi_down_blocks
        )
        bus_xs = self._encode(bus, self.bus_proj, self.bus_inc, self.bus_down_blocks)
        rail_xs = self._encode(
            rail, self.rail_proj, self.rail_inc, self.rail_down_blocks
        )
        poi_xs = self._encode(poi, self.poi_proj, poi_inc, poi_down)
        if self.fusion == "concat":
            x = torch.cat([bus_xs[-1], rail_xs[-1], poi_xs[-1]], dim=1)
        elif self.fusion == "xattn":
            x = self.xattn_fusion(bus_xs[-1], rail_xs[-1], poi_xs[-1])
        else:
            x = bus_xs[-1] + rail_xs[-1] + poi_xs[-1]
        x = self.bottleneck_attn(self.bottleneck(x))
        fused_skips: List[torch.Tensor] = []
        for bus_skip, rail_skip, poi_skip, fuse in zip(
            bus_xs[:-1], rail_xs[:-1], poi_xs[:-1], self.skip_fuse
        ):
            if self.fusion == "concat":
                fused_skips.append(
                    fuse(torch.cat([bus_skip, rail_skip, poi_skip], dim=1))
                )
            else:
                fused_skips.append(fuse(bus_skip + rail_skip + poi_skip))
        for idx, up in enumerate(self.up_blocks):
            x = up(x, fused_skips[-idx - 1])
        return _head_output(self.outc(x), return_uncertainty)


def build_model(
    model_cfg: ModelConfig,
    in_channels: int,
    out_channels: int,
    *,
    channel_info: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    arch = str(getattr(model_cfg, "architecture", "unet") or "unet").lower()
    common = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=model_cfg.base_channels,
        depth=model_cfg.depth,
        channel_mult=model_cfg.channel_mult,
        up_mode=model_cfg.up_mode,
        dropout=model_cfg.dropout,
        activation=model_cfg.activation,
        activation_negative_slope=model_cfg.activation_negative_slope,
        norm=model_cfg.norm,
        norm_groups=model_cfg.norm_groups,
        residual_blocks=model_cfg.residual_blocks,
        skip_attention=model_cfg.skip_attention,
        bottleneck_attention=model_cfg.bottleneck_attention,
        decoder_attention=model_cfg.decoder_attention,
        attention_reduction=model_cfg.attention_reduction,
        bottleneck_attention_heads=getattr(model_cfg, "bottleneck_attention_heads", 4),
        bottleneck_attention_blocks=getattr(
            model_cfg, "bottleneck_attention_blocks", 1
        ),
        output_uncertainty=getattr(model_cfg, "output_uncertainty", False),
    )
    if arch == "unet":
        return UNet(**common)
    if arch == "resunet_aspp":
        return ResUNetASPP(
            **common,
            input_projection_channels=model_cfg.input_projection_channels,
            aspp_dilations=model_cfg.aspp_dilations,
            aspp_dropout=model_cfg.aspp_dropout,
        )
    if arch == "dual_stream_resunet":
        bus_in, context_in = resolve_dual_stream_channels(
            model_cfg, in_channels, channel_info=channel_info
        )
        return DualStreamResUNet(
            **common,
            bus_in_channels=bus_in,
            context_in_channels=context_in,
            input_projection_channels=model_cfg.input_projection_channels,
            input_stem=model_cfg.input_stem,
            aspp_dilations=model_cfg.aspp_dilations,
            aspp_dropout=model_cfg.aspp_dropout,
            fusion=model_cfg.dual_stream_fusion,
            share_encoder=model_cfg.dual_stream_share_encoder,
        )
    if arch == "triple_stream_resunet":
        bus_in, rail_in, poi_in = resolve_triple_stream_channels(
            model_cfg, in_channels, channel_info=channel_info
        )
        return TripleStreamResUNet(
            **common,
            bus_in_channels=bus_in,
            rail_in_channels=rail_in,
            poi_in_channels=poi_in,
            input_projection_channels=model_cfg.input_projection_channels,
            input_stem=model_cfg.input_stem,
            aspp_dilations=model_cfg.aspp_dilations,
            aspp_dropout=model_cfg.aspp_dropout,
            fusion=model_cfg.triple_stream_fusion,
            share_rail_poi_encoder=model_cfg.triple_stream_share_rail_poi,
        )
    raise ValueError(f"Unsupported architecture '{model_cfg.architecture}'")
