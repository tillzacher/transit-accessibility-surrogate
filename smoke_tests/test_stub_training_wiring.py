"""Guard augmentation wiring and behavior without running training."""

import ast
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "modules/training_pipeline.py"
TREE = ast.parse(SOURCE.read_text())
FIELDS = (
    "augment_stub_edge",
    "augment_stub_edge_prob",
    "augment_stub_edge_min_len_px",
    "augment_stub_edge_max_len_px",
)


def test_training_constructor_forwards_all_stub_controls():
    assignment = next(
        (
            n
            for n in ast.walk(TREE)
            if isinstance(n, ast.Assign)
            and any((isinstance(t, ast.Name) and t.id == "train_ds" for t in n.targets))
        )
    )
    kwargs = {k.arg: k.value for k in assignment.value.keywords}
    cfg = SimpleNamespace(**dict(zip(FIELDS, (True, 0.7, 3, 11))))
    for field in FIELDS:
        value = eval(
            compile(ast.Expression(kwargs[field]), str(SOURCE), "eval"),
            {"data_cfg": cfg},
        )
        assert value == getattr(cfg, field)
    for n in ast.walk(TREE):
        if isinstance(n, ast.Assign) and any(
            (
                isinstance(t, ast.Name) and t.id in ("val_ds", "test_ds")
                for t in n.targets
            )
        ):
            call = n.value.body if isinstance(n.value, ast.IfExp) else n.value
            assert not set(FIELDS).intersection((k.arg for k in call.keywords))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_stub_transform_preserves_context_and_encodes_zero_target(enabled, normalized):
    cls = next(
        (
            n
            for n in TREE.body
            if isinstance(n, ast.ClassDef) and n.name == "TileDataset"
        )
    )
    fn = next(
        (
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_maybe_augment_stub_edge"
        )
    )
    namespace = {"torch": torch, "random": random, "Optional": Optional}
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    stats = (
        {
            "bus": (np.array([0.2, 0.3]), np.array([0.4, 0.5])),
            "target": (np.array([2.0, 3.0]), np.array([4.0, 5.0])),
        }
        if normalized
        else {}
    )
    dataset = SimpleNamespace(
        augment_stub_edge=enabled,
        bus_channels=2,
        augment_stub_edge_prob=1.0,
        augment_stub_edge_min_len_px=5,
        augment_stub_edge_max_len_px=5,
        norm_stats=stats,
        slim_mode=False,
    )
    inputs = torch.zeros(4, 32, 32)
    inputs[0, 16, 16] = 1.0
    inputs[1:] = 0.6
    target = torch.ones(2, 32, 32)
    before = inputs.clone()
    out, y = namespace[fn.name](dataset, inputs, target)
    assert torch.equal(inputs, before)
    assert torch.equal(target, torch.ones_like(target))
    if not enabled:
        assert out is inputs and y is target
        return
    assert torch.equal(out[0], inputs[0])
    assert torch.equal(out[2:], inputs[2:])
    assert not torch.equal(out[1], inputs[1])
    if normalized:
        mean, std = stats["target"]
        for c in range(2):
            assert torch.allclose(
                y[c], torch.full_like(y[c], -mean[c] / (std[c] + 1e-06))
            )
    else:
        assert torch.count_nonzero(y) == 0
