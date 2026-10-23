"""Extract bus accessibility changes at pixel-centre H3 cells."""

from pathlib import Path
import numpy as np
from modules.synthetic_world import WorldSpec

TILE_SIZE = 200


def pixel_h3_index(spec: WorldSpec, res: int = 10) -> list[str]:
    """H3 cell covering each pixel centre, matching RASTER_METHOD = 'center'."""
    import h3
    from pyproj import Transformer

    tf = Transformer.from_crs(spec.crs, "EPSG:4326", always_xy=True)
    cols = spec.minx + (np.arange(TILE_SIZE) + 0.5) * spec.pixel_size_m
    rows = spec.maxy - (np.arange(TILE_SIZE) + 0.5) * spec.pixel_size_m
    xx, yy = np.meshgrid(cols, rows)
    lon, lat = tf.transform(xx.ravel(), yy.ravel())
    return [h3.latlng_to_cell(float(a), float(o), res) for a, o in zip(lat, lon)]


def bus_delta_field(
    parquet: Path, channels: list[str], cell_idx: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """(C, H, W) bus delta on the tile grid, plus a per-channel presence mask."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet)
    names = set(table.schema.names)
    h3_col = table.column("h3").to_pylist()
    pos = {c: i for i, c in enumerate(h3_col)}
    take = np.array([pos.get(c, -1) for c in cell_idx])
    present = np.zeros(len(channels), dtype=bool)
    field = np.zeros((len(channels), TILE_SIZE, TILE_SIZE), dtype=np.float32)
    for j, name in enumerate(channels):
        layer = name if name.startswith("wrk_") else f"poi_{name}"
        wt, wtr = (f"{layer}_walk_transit", f"{layer}_walk_transit_rail")
        if wt not in names or wtr not in names:
            continue
        delta = (
            table.column(wt).to_numpy(zero_copy_only=False)
            - table.column(wtr).to_numpy(zero_copy_only=False)
        ).astype(np.float32)
        vals = np.where(take >= 0, delta[np.clip(take, 0, None)], 0.0)
        field[j] = vals.reshape(TILE_SIZE, TILE_SIZE)
        present[j] = True
    return (field, present)
