"""Run one supplied synthetic network locally with an isolated routing cache."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.reference_fields import pixel_h3_index, bus_delta_field
from modules.synthetic_world import WorldSpec
from modules.generated_metamorphic_suite import build_set_with_scene


def run(command, cwd):
    return subprocess.run(
        command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE
    ).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument(
        "--engine",
        type=Path,
        default=ROOT.parent / "MAPPER",
        help="built MAPPER v2.0.0 checkout (default: ../MAPPER)",
    )
    ap.add_argument(
        "--inputs",
        type=Path,
        default=ROOT / "data/reference_cases",
        help="reference-case directory (default: data/reference_cases)",
    )
    a = ap.parse_args()
    inputs = a.inputs.resolve()
    engine = a.engine.resolve()
    rows = json.loads((inputs / "index.json").read_text())
    row = next((r for r in rows if r["case"] == a.case))
    source = inputs / a.case
    for name, expected in row["files"].items():
        assert (
            hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
        ), name
    output = a.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    params = json.loads((inputs / "parameters.json").read_text())
    classes = json.loads((inputs / "classes.json").read_text())
    params.update(
        REGION=a.case,
        PBF_PATH=str(source / "world.osm.pbf"),
        GTFS_ZIP=str(source / "gtfs_all.zip"),
        GTFS_ZIP_BUS=str(source / "gtfs_bus.zip"),
        GTFS_ZIP_RAIL=str(source / "gtfs_rail.zip"),
        PREPARED_BUNDLE_ROOT=str(output / "prepared"),
        RK_INDEX_CACHE_DIR=str(output / "prepared/routing_indexes"),
        POI_CLASSES=",".join(classes["poi_classes"]),
        WORKPLACES=",".join(classes["workplaces"]),
        OMP_THREADS=str(a.threads),
        HEX_OMP_THREADS=str(a.threads),
    )
    config = output / "engine.cfg"
    config.write_text("".join((f"{k} = {v}\n" for k, v in params.items())))
    digest = json.loads(
        run(
            [
                "source/cpp/.build/bin/digest_cpp",
                "--config",
                str(config),
                "--taxonomy",
                "source/poi_mapping.yaml",
                "--json",
            ],
            engine,
        )
    )["graph_digest"]
    ring = output / "prepared/graphs" / digest / "ring.txt"
    ring.parent.mkdir(parents=True)
    shutil.copy2(source / "ring.txt", ring)
    log = run(
        [
            "bash",
            "source/run_native.sh",
            "--config",
            str(config),
            "--out-dir",
            str(output / "run"),
        ],
        engine,
    )
    (output / "engine.log").write_text(log)
    spec = WorldSpec(**json.loads((inputs / "world.json").read_text()))
    schema = json.loads((ROOT / "data/channel_schema_105.json").read_text())
    field, present = bus_delta_field(
        output / "run/cells.parquet", schema["poi"], pixel_h3_index(spec)
    )
    cal = json.loads((ROOT / "data/calibration.json").read_text())
    tiles, _ = build_set_with_scene(cal, schema, 20260811, row["set_index"])
    names = {
        "isolated_stop_leakage": "connected_stop_with_poi",
        "stub_edge_leakage": "stub_edge_full_route",
        "rail_connection_sensitivity": "stop_near_rail_connected",
        "amplitude_response": "route_single_line",
        "spatial_delta_localization": "localized_route",
        "poi_reachability_sensitivity": "route_poi_reachable",
    }
    tile = tiles[names[row["family"]]]
    control = tile.eval_mask if tile.control_mask is None else tile.control_mask
    result = {
        "gain_eval": np.maximum(field[:, tile.eval_mask], 0).mean(1),
        "gain_control": np.maximum(field[:, control], 0).mean(1),
        "present": present,
    }
    np.savez_compressed(output / "gains.npz", **result)
    with np.load(source / "gains.npz") as expected:
        errors = {
            k: float(np.max(np.abs(result[k].astype(float) - expected[k])))
            for k in result
        }
    (output / "comparison.json").write_text(json.dumps(errors, indent=2) + "\n")
    print(json.dumps(errors))


if __name__ == "__main__":
    main()
