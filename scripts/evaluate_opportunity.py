"""Infer and retain A/B/background gains on an immutable opportunity corpus."""

from __future__ import annotations
import argparse
import gc
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.opportunity_attenuation import VERSION, score_attenuation
from modules.scoring_scenarios import load_scenario, run_model_raw
from modules.model_io import load_model_checkpoint


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((a.inputs / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(a.inputs / name) != sha:
            raise ValueError(f"Input hash mismatch: {name}")
    floor = np.array(manifest["floor"])
    a.out.mkdir(parents=True, exist_ok=True)
    for row in manifest["models"][a.start : a.stop]:
        name = row["model"]
        target = a.out / f"{name}.npz"
        ckpt_path = a.models / f"{name}.pt"
        if digest(ckpt_path) != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
        if target.exists():
            with np.load(target) as z:
                assert str(z["version"]) == VERSION
                assert str(z["input_manifest_sha256"]) == digest(
                    a.inputs / "manifest.json"
                )
                assert str(z["checkpoint_sha256"]) == row["checkpoint_sha256"]
            print(f"verified existing {name}", flush=True)
            continue
        model = load_model_checkpoint(str(ckpt_path), device=a.device)
        arrays = {}
        for i in range(manifest["n_sets"]):
            base = a.inputs / f"set{i:03d}"
            ref = load_scenario(base / "a.npz")
            probe = load_scenario(base / "b.npz")
            with np.load(base / "background.npz") as z:
                null = replace(ref, poi_raw=z["poi_raw"])
                eligible = z["eligible"]
            for key in ("bus_stops", "bus_edges", "rail_raw", "eval_mask"):
                assert np.array_equal(getattr(ref, key), getattr(probe, key))
            gains = []
            for tile in (ref, probe, null):
                prediction = run_model_raw(model, tile, device=a.device)
                gains.append(np.maximum(prediction[:, ref.eval_mask], 0).mean(axis=1))
                del prediction
            scores = score_attenuation(*gains, floor, eligible=eligible)
            for key, value in {
                **scores,
                "gain_a": gains[0],
                "gain_b": gains[1],
                "gain_n": gains[2],
            }.items():
                arrays.setdefault(key, []).append(value)
            if (i + 1) % 10 == 0:
                print(f"{name}: {i + 1}/{manifest['n_sets']}", flush=True)
        np.savez_compressed(
            target,
            **{k: np.stack(v) for k, v in arrays.items()},
            version=VERSION,
            checkpoint_sha256=row["checkpoint_sha256"],
            input_manifest_sha256=digest(a.inputs / "manifest.json"),
            torch_version=torch.__version__,
        )
        print(f"completed {name}", flush=True)
        del model, arrays
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
