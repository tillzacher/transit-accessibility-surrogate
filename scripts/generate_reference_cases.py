"""Regenerate the synthetic routing cases in ``data/reference_cases``.

Each case is one routed condition of one generated set. The scene geometry of set
``set_index`` (``build_set_with_scene`` with seed 20260811) is authored by
``modules.synthetic_world`` into an OSM extract with a street grid, the bus corridor
and tagged opportunities, plus bus, rail and combined GTFS feeds and a clip ring.
Every world is audited (shared crossing nodes, stop-to-walk-node gaps, ghost
terminals) and generation stops on the first audit error.

The 500 cases are the conditions of sets 0-49 that need routing: analytic
zero-service conditions are exact by construction, and ``displaced_far`` was a
supplementary displacement test outside the reference set. Case numbers follow the
three batches in which the cases were routed (see ``plan_cases``).

The routing outputs (``gains.npz`` and the hash of the raw engine Parquet) come from
MAPPER, not from this script. ``--stored-gains`` copies them from an existing case
tree, which makes the output a drop-in replacement for ``data/reference_cases``.
``--compare`` checks every generated file against a reference tree: byte identity
first, otherwise decoded content (OSM objects and tags, GTFS tables).

Usage::

    python scripts/generate_reference_cases.py --out analysis/reference_cases
    python scripts/generate_reference_cases.py --out analysis/reference_cases \\
        --cases case_0000 case_0002 --compare data/reference_cases
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modules.generated_metamorphic_suite import build_set_with_scene
from modules.synthetic_world import WorldSpec, build_condition_world, load_poi_tag_map

TOP_SEED = 20260811
N_SETS = 50
EXCLUDED_CONDITIONS = ("displaced_far",)
# Slices of the planned order (see plan_cases), in case-number order.
ROUTING_BATCHES = ((200, 400), (400, 500), (0, 200))
INPUT_FILES = ("world.osm.pbf", "gtfs_all.zip", "gtfs_bus.zip", "gtfs_rail.zip", "ring.txt")
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)

# Accessibility parameters of the recorded training-data configuration. The routed
# reference is only the system the surrogate was trained on if these are unchanged.
ACCESS = {
    "MODES": "walk,walk_transit,walk_transit_bus,walk_transit_rail",
    "CRS": "EPSG:32632",
    "START_TIME": "07:00",
    "END_TIME": "09:00",
    "WALK_CONNECTION_MAX_DISTANCE": "150.0",
    "WALK_SPEED_KMPH": "6.5",
    "GRAPH_DIRECTION_MODE": "undirected",
    "NODE_THRESHOLD": "120",
    "EDGE_THRESHOLD": "170",
    "MAX_SNAP_M": "200.0",
    "WORKPLACE_MAX_WEIGHT": "1.0",
    "KERNEL_TYPE": "exponential",
    "KERNEL_PARAM": "0.0005",
    "CUTOFF_S": "3500",
    "CEILING": "100",
    "H3_RES": "10",
    "DEFAULT_VALUE": "0.0",
    "PARQUET_ROW_GROUP": "2000000",
    "GRAPH_CACHE_ENABLED": "1",
    "H3_CACHE_ENABLED": "1",
}

# Opportunity and workplace classes of the same configuration (98 + 8 layers).
POI_CLASSES = (
    "hospital doctor clinic dentist pharmacy veterinary police fire_station "
    "ambulance_station restaurant fast_food cafe bar pub ice_cream biergarten food_court "
    "drinking_water school university kindergarten library music_school driving_school "
    "bank atm bureau_de_change fuel_station bicycle_parking ferry_terminal "
    "charging_station taxi_stand car_rental bicycle_rental hotel motel guest_house hostel "
    "campsite supermarket convenience_store bakery clothing_store department_store "
    "electronics hardware_store furniture_store garden_centre book_store alcohol_shop "
    "car_dealership bicycle_shop sports_shop jewelry hairdresser cosmetics optician "
    "gift_shop toy_store pet_store butcher greengrocer florist shoe_store newsagent "
    "laundry park playground sports_centre swimming_pool stadium golf_course pitch museum "
    "cinema theatre arts_centre nightclub casino theme_park zoo viewpoint attraction "
    "picnic_site dog_park post_office town_hall community_centre place_of_worship toilets "
    "recycling waste_disposal shelter embassy prison brewery winery workshop"
).split()
WORKPLACES = "office commercial industrial retail warehouse factory greenhouse craft".split()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs() -> tuple[dict, dict]:
    calibration = json.loads((ROOT / "data/calibration.json").read_text())
    schema = json.loads((ROOT / "data/channel_schema_105.json").read_text())
    return calibration, schema


def plan_cases(calibration: dict, schema: dict) -> list[dict]:
    """Return the 500 cases in case-number order.

    The planned order is opportunity reachability first, then the other families,
    each by set and condition. The cases were routed in three batches of that order
    and numbered batch by batch (``ROUTING_BATCHES``), by region key within a batch.
    """
    _, scene = build_set_with_scene(calibration, schema, TOP_SEED, 0)
    conditions = {name: fam.conditions for name, fam in scene.families.items()}
    planned = [
        {"set_index": s, "family": family, "condition": c}
        for s in range(N_SETS)
        for family, conds in conditions.items()
        for c in conds
        if not c.endswith(":analytic_zero") and c not in EXCLUDED_CONDITIONS
    ]
    planned.sort(
        key=lambda r: (
            r["family"] != "poi_reachability_sensitivity",
            r["set_index"],
            r["condition"],
        )
    )
    assert sum(b - a for a, b in ROUTING_BATCHES) == len(planned) == 500
    cases = []
    for a, b in ROUTING_BATCHES:
        cases.extend(sorted(planned[a:b], key=region_key))
    return [{"case": f"case_{i:04d}", **row} for i, row in enumerate(cases)]


def region_key(row: dict) -> str:
    return f"twin_s{row['set_index']:03d}_{row['family']}_{row['condition']}"


def freeze_zip(path: Path) -> None:
    """Rewrite a zip with fixed member timestamps so equal content gives equal bytes."""
    with zipfile.ZipFile(path) as z:
        members = [(i.filename, z.read(i)) for i in z.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members:
            info = zipfile.ZipInfo(name, ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            z.writestr(info, data)


def write_ring(path: Path, spec: WorldSpec) -> None:
    """Write the clip ring (world bounding box, ``lat,lon`` per line).

    MAPPER derives the ring from administrative boundaries, which a synthetic
    extract does not contain, so the ring is part of the authored inputs.
    """
    from pyproj import Transformer

    tf = Transformer.from_crs(spec.crs, "EPSG:4326", always_xy=True)
    minx, miny, maxx, maxy = spec.world_bounds
    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    with path.open("w") as fh:
        for x, y in corners:
            lon, lat = tf.transform(x, y)
            fh.write(f"{lat:.9f},{lon:.9f}\n")


def generate_set(set_index: int, rows: list[dict], out: Path, taxonomy: Path, osmium: str) -> list[dict]:
    """Author the worlds of the selected cases of one set into ``out/<case>/``."""
    calibration, schema = load_inputs()
    spec = WorldSpec()
    tag_map = load_poi_tag_map(taxonomy)
    _, scene = build_set_with_scene(calibration, schema, TOP_SEED, set_index)
    done = []
    with tempfile.TemporaryDirectory(dir=out) as work:
        for row in rows:
            fam = scene.families[row["family"]]
            assert row["condition"] in fam.conditions, row
            world = build_condition_world(
                fam, set_index, row["condition"], Path(work), spec,
                scene.channel_names, tag_map, osmium_bin=osmium,
            )
            if world.audit["errors"]:
                raise SystemExit(f"{row['case']}: world audit failed: {world.audit['errors']}")
            case_dir = out / row["case"]
            case_dir.mkdir(exist_ok=True)
            for src in (world.pbf, world.gtfs_all, world.gtfs_bus, world.gtfs_rail):
                shutil.move(str(src), case_dir / src.name)
            for name in ("gtfs_all.zip", "gtfs_bus.zip", "gtfs_rail.zip"):
                freeze_zip(case_dir / name)
            write_ring(case_dir / "ring.txt", spec)
            done.append(row)
            print(f"{row['case']}  {region_key(row)}", flush=True)
    return done


def write_metadata(out: Path, rows: list[dict], stored: Path | None) -> None:
    (out / "world.json").write_text(json.dumps(asdict(WorldSpec()), indent=2) + "\n")
    (out / "parameters.json").write_text(json.dumps(ACCESS, indent=2) + "\n")
    classes = {"poi_classes": POI_CLASSES, "workplaces": WORKPLACES}
    (out / "classes.json").write_text(json.dumps(classes, indent=1) + "\n")
    stored_rows = {}
    if stored is not None:
        stored_rows = {
            (r["set_index"], r["family"], r["condition"]): r
            for r in json.loads((stored / "index.json").read_text())
        }
    index = []
    for row in rows:
        case_dir = out / row["case"]
        files = {name: sha256(case_dir / name) for name in INPUT_FILES}
        entry = {**row, "files": files}
        ref = stored_rows.get((row["set_index"], row["family"], row["condition"]))
        if stored is not None:
            if ref is None:
                raise SystemExit(f"{row['case']}: no stored routing result in {stored}")
            shutil.copy2(stored / ref["case"] / "gains.npz", case_dir / "gains.npz")
            files["gains.npz"] = sha256(case_dir / "gains.npz")
            entry["raw_parquet_sha256"] = ref["raw_parquet_sha256"]
        index.append(entry)
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")


def _osm_content(path: Path, osmium: str) -> str:
    """Decoded OSM objects with tags (OPL), without the file header."""
    return subprocess.run(
        [osmium, "cat", str(path), "-f", "opl"], check=True, text=True, stdout=subprocess.PIPE
    ).stdout


def _zip_content(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as z:
        return [(i.filename, z.read(i)) for i in z.infolist()]


def compare(out: Path, reference: Path, osmium: str) -> dict:
    """Compare generated cases with a reference tree, by bytes and by decoded content."""
    ref_rows = {
        (r["set_index"], r["family"], r["condition"]): r
        for r in json.loads((reference / "index.json").read_text())
    }
    report = {"identical": 0, "same_content": [], "different": []}
    for name in ("world.json", "parameters.json", "classes.json"):
        if (out / name).read_bytes() == (reference / name).read_bytes():
            report["identical"] += 1
        else:
            report["different"].append(name)
    for row in json.loads((out / "index.json").read_text()):
        ref = ref_rows[(row["set_index"], row["family"], row["condition"])]
        if ref["case"] != row["case"]:
            report["different"].append(f"{row['case']}: numbered {ref['case']} in reference")
        for name in INPUT_FILES:
            label = f"{row['case']}/{name}"
            if row["files"][name] == ref["files"][name]:
                report["identical"] += 1
                continue
            mine, theirs = out / row["case"] / name, reference / ref["case"] / name
            if name.endswith(".pbf"):
                same = _osm_content(mine, osmium) == _osm_content(theirs, osmium)
            elif name.endswith(".zip"):
                same = _zip_content(mine) == _zip_content(theirs)
            else:
                same = False
            (report["same_content"] if same else report["different"]).append(label)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output case tree (created)")
    ap.add_argument("--cases", nargs="+", help="generate only these cases (default: all 500)")
    ap.add_argument(
        "--engine", type=Path, default=ROOT.parent / "MAPPER",
        help="MAPPER v2.0.0 checkout providing source/poi_mapping.yaml (default: ../MAPPER)",
    )
    ap.add_argument("--taxonomy", type=Path, help="POI taxonomy file (default: <engine>/source/poi_mapping.yaml)")
    ap.add_argument("--osmium", default="osmium", help="osmium executable (osmium-tool)")
    ap.add_argument("--jobs", type=int, default=1, help="sets generated in parallel")
    ap.add_argument("--stored-gains", type=Path, help="copy gains.npz and the Parquet hash from this case tree")
    ap.add_argument("--compare", type=Path, help="compare the output with this case tree")
    a = ap.parse_args()

    taxonomy = (a.taxonomy or a.engine / "source/poi_mapping.yaml").resolve()
    if not taxonomy.is_file():
        ap.error(f"POI taxonomy not found: {taxonomy} (pass --engine or --taxonomy)")
    if shutil.which(a.osmium) is None:
        ap.error(f"osmium executable not found: {a.osmium}")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = plan_cases(*load_inputs())
    if a.cases:
        unknown = set(a.cases) - {r["case"] for r in rows}
        if unknown:
            ap.error(f"unknown cases: {sorted(unknown)}")
        rows = [r for r in rows if r["case"] in set(a.cases)]
    by_set: dict[int, list[dict]] = {}
    for row in rows:
        by_set.setdefault(row["set_index"], []).append(row)
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            futures = [pool.submit(generate_set, s, r, out, taxonomy, a.osmium) for s, r in sorted(by_set.items())]
            for f in futures:
                f.result()
    else:
        for s, r in sorted(by_set.items()):
            generate_set(s, r, out, taxonomy, a.osmium)
    write_metadata(out, rows, a.stored_gains.resolve() if a.stored_gains else None)
    print(f"wrote {len(rows)} cases to {out}")

    if a.compare:
        report = compare(out, a.compare.resolve(), a.osmium)
        print(json.dumps({k: v if isinstance(v, int) else len(v) for k, v in report.items()}))
        for label in report["different"]:
            print(f"DIFFERENT {label}")
        if report["different"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
