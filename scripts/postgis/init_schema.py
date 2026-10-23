#!/usr/bin/env python3
"""Create the PostGIS schema read by the tile factory on an existing server.

Connection settings come from SURROGATE_POSTGIS_* or .env.postgis (README section 4).
docker-compose.postgis.yml applies the same SQL files automatically on first start.
"""

from __future__ import annotations

import argparse

from _common import SQL_DIR, get_logger

from modules.postgis_bridge import get_engine

logger = get_logger("init_schema")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sql-dir", default=str(SQL_DIR), help="Directory with the ordered *.sql files")
    args = parser.parse_args()

    from pathlib import Path

    files = sorted(Path(args.sql_dir).glob("*.sql"))
    if not files:
        raise SystemExit(f"No SQL files found in {args.sql_dir}")
    raw = get_engine().raw_connection()
    try:
        cur = raw.cursor()
        for path in files:
            logger.info("Applying %s", path.name)
            cur.execute(path.read_text())
        raw.commit()
        cur.close()
    finally:
        raw.close()
    logger.info("Schema ready (%d files)", len(files))


if __name__ == "__main__":
    main()
