"""Generate the 200×200×105 controlled metamorphic evaluation corpus."""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from modules.generated_metamorphic_suite import (
    FORMAT,
    GENERATOR_VERSION,
    build_evaluation_set,
    sha256_file,
    validate_set,
    write_set,
)


def _corpus_hash(entries: list[dict]) -> str:
    h = hashlib.sha256()
    for entry in entries:
        for f in entry["files"]:
            h.update(f["sha256"].encode())
    return h.hexdigest()


def ensure_output_ready(path: Path, *, overwrite: bool) -> None:
    """Reject stale mixed corpora unless replacement was explicitly requested."""
    if path.exists() and any(path.iterdir()) and (not overwrite):
        raise FileExistsError(
            f"refusing non-empty output {path}; use a new path or --overwrite"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--schema", type=Path, default=Path("data/channel_schema_105.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-sets", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_sets < 1:
        parser.error("--n-sets must be positive")
    try:
        ensure_output_ready(args.out, overwrite=args.overwrite)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    calibration = json.loads(args.calibration.read_text())
    schema = json.loads(args.schema.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(args.n_sets):
        tiles = build_evaluation_set(calibration, schema, args.seed, i)
        errors = validate_set(tiles, len(schema["poi"]))
        if errors:
            raise RuntimeError(f"set {i:03d} failed validation:\n" + "\n".join(errors))
        entries.append(write_set(tiles, args.out / f"gen{i + 1:03d}"))
        print(f"generated and validated set {i + 1}/{args.n_sets}")
    manifest = {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "top_seed": args.seed,
        "n_sets": args.n_sets,
        "n_scenarios_per_set": entries[0]["n_scenarios"],
        "schema": {"path": str(args.schema), "sha256": sha256_file(args.schema)},
        "calibration": {
            "path": str(args.calibration),
            "sha256": sha256_file(args.calibration),
            "provisional": bool(calibration.get("provisional", False)),
        },
        "coverage_schedule": "all_channels every fifth set; empirical-density otherwise",
        "sets": entries,
    }
    manifest["schema_values"] = schema
    manifest["calibration_values"] = calibration
    manifest["opportunity_measurement"] = "opportunity-attenuation-1"
    manifest["corpus_sha256"] = _corpus_hash(entries)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out / 'manifest.json'}")
    print(f"corpus_sha256={manifest['corpus_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
