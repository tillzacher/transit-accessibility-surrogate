"""Reader for MAPPER graph files (`<mode>.bin`, GOSC format v6/v7).

Layout (little-endian), as written by MAPPER's `Graph::save_binary`:

    magic u32 ("GOSC"), version u32, node_count u32, edge_count u32
    node_ids u32[N], lat i32[N] (1e-7 deg), lon i32[N] (1e-7 deg), x f32[N], y f32[N]
    node_kind u8[N]                      (v7: 1 = synthetic transit stop node)
    edge_u u32[E], edge_v u32[E]         (row indices into the node arrays)
    edge_weight u32[E]                   (travel time, ms)
    edge_midway_entry u8[E], edge_direction u8[E] (0 both ways, 1 u->v, 2 v->u)
    crs_len u32, crs bytes, is_projected u8
    per edge: n u32, then f32[n] shape as lat0, lon0, lat1, lon1, ...
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GOSC_MAGIC = 0x474F5343


@dataclass
class MapperGraph:
    node_ids: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    node_kind: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    weight_ms: np.ndarray
    direction: np.ndarray
    shapes: list[np.ndarray | None]
    crs: str
    version: int


def read_mapper_graph(path: Path) -> MapperGraph:
    buf = Path(path).read_bytes()
    magic, version, n, m = struct.unpack_from("<4I", buf, 0)
    if magic != GOSC_MAGIC:
        raise ValueError(f"{path} is not a MAPPER graph file")
    if version not in (6, 7):
        raise ValueError(f"{path}: unsupported graph format version {version} (expected 6 or 7)")
    off = 16

    def take(dtype, count):
        nonlocal off
        arr = np.frombuffer(buf, dtype=dtype, count=count, offset=off)
        off += arr.nbytes
        return arr

    node_ids = take("<u4", n)
    lat = take("<i4", n) / 1e7
    lon = take("<i4", n) / 1e7
    take("<f4", n)
    take("<f4", n)
    node_kind = take("u1", n) if version >= 7 else np.zeros(n, dtype=np.uint8)
    edge_u = take("<u4", m)
    edge_v = take("<u4", m)
    weight_ms = take("<u4", m)
    take("u1", m)
    direction = take("u1", m)
    (crs_len,) = struct.unpack_from("<I", buf, off)
    off += 4
    crs = buf[off:off + crs_len].decode("utf-8")
    off += crs_len + 1
    shapes: list[np.ndarray | None] = []
    for _ in range(m):
        (count,) = struct.unpack_from("<I", buf, off)
        off += 4
        if count:
            shapes.append(np.frombuffer(buf, dtype="<f4", count=count, offset=off).reshape(-1, 2))
            off += 4 * count
        else:
            shapes.append(None)
    if off != len(buf):
        raise ValueError(f"{path}: {len(buf) - off} trailing bytes; file is truncated or not GOSC v{version}")
    return MapperGraph(node_ids, lat, lon, node_kind, edge_u, edge_v, weight_ms, direction, shapes, crs, version)
