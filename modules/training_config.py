"""Training config dataclasses for the U-Net surrogate pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from torch import nn

RunSelector = Union[int, str, Path, Sequence[Union[int, str, Path]]]


@dataclass
class DataConfig:
    output_root: Path = Path("training_data_runs")
    run_selector: RunSelector = "latest"
    split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_seed: int = 42
    init_seed: Optional[int] = None
    split_mode: str = "random"
    holdout_region: Optional[Dict[str, Any]] = None
    holdout_buffer_m: float = 24000.0
    holdout_val_fraction: float = 0.1
    lazy_load: bool = False
    precompute_workers: Optional[int] = None
    shuffle: bool = True
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    drop_last: bool = False
    pca_components: Optional[int] = None
    pca_keys: Optional[Tuple[str, ...]] = None
    pca_max_samples: int = 100000
    pca_random_state: int = 0
    normalize_inputs: bool = False
    normalize_bus_channels: bool = True
    normalize_targets: bool = False
    pca_fit_source: str = "combined"
    pca_apply_to_context: bool = True
    pca_apply_to_target: bool = True
    target_mode: str = "delta"
    slim_mode: bool = False
    slim_channel_idx: Optional[int] = None
    slim_expand_channels: bool = False
    limit_samples: Optional[int] = None
    precomputed_sample_cache: bool = False
    precomputed_sample_cache_dir: Optional[Path] = None
    precomputed_sample_cache_force_rebuild: bool = False
    preload_precomputed_samples: bool = False
    preload_precomputed_samples_max_gb: Optional[float] = None
    preload_to_ram: bool = False
    augment_flip: bool = False
    augment_rot90: bool = False
    augment_bus_activation_scale: bool = False
    augment_bus_activation_scale_prob: float = 1.0
    augment_bus_activation_scale_min: float = 0.05
    augment_bus_activation_scale_max: float = 1.0
    augment_bus_activation_scale_distribution: str = "log_uniform"
    augment_bus_activation_scale_keep_full_prob: float = 0.25
    augment_bus_activation_target_power: float = 1.0
    augment_isolated_stop: bool = False
    augment_isolated_stop_prob: float = 0.15
    augment_stub_edge: bool = False
    augment_stub_edge_prob: float = 0.15
    augment_stub_edge_min_len_px: int = 5
    augment_stub_edge_max_len_px: int = 30
    augment_orphan_stop_overlay: bool = False
    augment_orphan_stop_overlay_prob: float = 0.15
    augment_orphan_stop_overlay_min_shift_px: int = 12
    augment_orphan_stop_overlay_max_shift_px: int = 80
    augment_orphan_stop_overlay_edge_dilate_px: int = 5
    augment_orphan_stop_overlay_scale_min: float = 0.5
    augment_orphan_stop_overlay_scale_max: float = 1.0


@dataclass
class ModelConfig:
    architecture: str = "unet"
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    base_channels: int = 32
    depth: int = 4
    channel_mult: int = 2
    dropout: float = 0.0
    up_mode: str = "bilinear"
    activation: str = "relu"
    activation_negative_slope: float = 0.01
    norm: str = "batch"
    norm_groups: int = 8
    residual_blocks: bool = False
    skip_attention: str = "none"
    bottleneck_attention: str = "none"
    bottleneck_attention_heads: int = 4
    bottleneck_attention_blocks: int = 1
    decoder_attention: str = "none"
    attention_reduction: int = 8
    output_uncertainty: bool = False
    target_head: str = "outconv"
    input_projection_channels: Optional[int] = None
    input_stem: str = "project"
    aspp_dilations: Tuple[int, ...] = (1, 2, 4, 8)
    aspp_dropout: float = 0.0
    dual_stream_fusion: str = "concat"
    bus_input_channels: Optional[int] = None
    context_input_channels: Optional[int] = None
    dual_stream_share_encoder: bool = False
    triple_stream_fusion: str = "concat"
    triple_stream_share_rail_poi: bool = False
    rail_input_channels: Optional[int] = None
    poi_input_channels: Optional[int] = None


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 0.001
    weight_decay: float = 1e-05
    optimizer: str = "adam"
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: Optional[float] = None
    tv_weight: float = 0.0
    device: Optional[str] = None
    torch_threads: Optional[int] = None
    log_interval: int = 10
    val_interval: int = 1
    early_stop_patience: Optional[int] = None
    loss_type: str = "mse"
    smooth_l1_beta: float = 1.0
    peak_weight_alpha: float = 1.0
    peak_weight_power: float = 1.0
    grad_loss_weight: float = 0.1
    use_amp: bool = False
    lr_scheduler: Optional[str] = None
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    warmup_steps: int = 0
    grad_accum_steps: int = 1
    oom_retries: int = 2
    zero_grad_set_to_none: bool = True
    output_activation: Optional[str] = None
    save_model_figure: bool = True
    model_figure_path: Path = Path("figures/latest_model.png")
    save_checkpoint: bool = False
    resume: bool = True
    checkpoint_dir: Path = Path("models")
    checkpoint_name: Optional[str] = None
    config_checkpoint_name: Optional[str] = None
    checkpoint_history_name: Optional[str] = None
    figure_path: Optional[Path] = Path("figures/training_progress.png")
    loss_plot_log_scale: bool = False
    loss_plot_show_mae: bool = True
    torch_compile: bool = False


@dataclass
class TrainResult:
    history: Dict[str, List[float]]
    final_train_metrics: Dict[str, float]
    final_val_metrics: Optional[Dict[str, float]]
    final_test_metrics: Optional[Dict[str, float]]
    checkpoint_path: Optional[Path]
    figure_path: Optional[Path]
    model: nn.Module
    config: Dict
    param_count: int
    train_samples: int
    train_time_sec: float
    sec_per_epoch: float
    val_samples: int = 0
    test_samples: int = 0
    already_complete: bool = False
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None
    total_epochs_run: int = 0
    stopped_early: bool = False
