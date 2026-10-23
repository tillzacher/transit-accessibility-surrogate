"""Run all lightweight paper reanalyses from the included numerical arrays."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    args = ap.parse_args()
    out = args.out.resolve()
    commands = [
        ["reanalyze_paper.py", "--out", str(out / "paper")],
        ["reanalyze_opportunity.py", "--out", str(out / "opportunity")],
        ["reanalyze_thresholds.py", "--out", str(out / "thresholds.csv")],
        ["reanalyze_regions.py", "--out", str(out / "regions")],
        ["reanalyze_seeds.py", "--out", str(out / "seed_extension")],
    ]
    for name, *options in commands:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), *options],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
