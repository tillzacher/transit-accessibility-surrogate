"""Generate matched A/B/background inputs with a hash-checked model inventory."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.generated_metamorphic_suite import (
    build_set_with_scene,
    background_only_poi,
)
from modules.scoring_scenarios import save_scenario


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    a = ap.parse_args()
    cal = json.loads((a.data / "calibration.json").read_text())
    schema = json.loads((a.data / "channel_schema_105.json").read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    files = {}
    for i in range(50):
        tiles, scene = build_set_with_scene(cal, schema, 20260811, i)
        dest = a.out / f"set{i:03d}"
        dest.mkdir()
        for short, name in [
            ("a", "route_poi_reachable"),
            ("b", "route_poi_unreachable"),
        ]:
            save_scenario(tiles[name], dest / f"{short}.npz")
        family = scene.families["poi_reachability_sensitivity"]
        eligible = np.zeros(len(schema["poi"]), dtype=bool)
        eligible[list(family.focals["reachable"])] = True
        np.savez_compressed(
            dest / "background.npz",
            poi_raw=background_only_poi(family),
            eligible=eligible,
        )
        for p in dest.iterdir():
            files[str(p.relative_to(a.out))] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
    manifest = {
        "n_sets": 50,
        "top_seed": 20260811,
        "channels": schema["poi"],
        "floor": cal["target"]["recommended_dead_reference_floor"],
        "files": files,
        "models": [
            {k: r[k] for k in ("model", "checkpoint_sha256")}
            for r in json.loads((a.data / "models.json").read_text())
        ],
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
