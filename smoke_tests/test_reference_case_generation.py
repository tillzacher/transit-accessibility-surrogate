"""Regenerated synthetic routing inputs match the supplied reference cases."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("MAPPER_ROOT", ROOT.parent / "MAPPER"))
CASES = ("case_0000", "case_0005")


def test_generated_cases_match_index(tmp_path):
    if shutil.which("osmium") is None:
        pytest.skip("osmium (osmium-tool) is not installed")
    if not (ENGINE / "source/poi_mapping.yaml").is_file():
        pytest.skip("MAPPER checkout not found (set MAPPER_ROOT)")
    out = tmp_path / "cases"
    subprocess.run(
        [sys.executable, "scripts/generate_reference_cases.py", "--out", str(out),
         "--engine", str(ENGINE), "--cases", *CASES, "--compare", "data/reference_cases"],
        cwd=ROOT, check=True,
    )
    shipped = ROOT / "data/reference_cases"
    expected = {r["case"]: r for r in json.loads((shipped / "index.json").read_text())}
    for row in json.loads((out / "index.json").read_text()):
        ref = expected[row["case"]]
        assert (row["set_index"], row["family"], row["condition"]) == (
            ref["set_index"], ref["family"], ref["condition"])
        for name in ("world.osm.pbf", "ring.txt"):
            assert row["files"][name] == ref["files"][name], (row["case"], name)
        for name in ("gtfs_all.zip", "gtfs_bus.zip", "gtfs_rail.zip"):
            with zipfile.ZipFile(out / row["case"] / name) as a, \
                    zipfile.ZipFile(shipped / row["case"] / name) as b:
                assert [(i.filename, a.read(i)) for i in a.infolist()] == [
                    (i.filename, b.read(i)) for i in b.infolist()], (row["case"], name)
    for name in ("world.json", "parameters.json", "classes.json"):
        assert (out / name).read_bytes() == (shipped / name).read_bytes(), name
