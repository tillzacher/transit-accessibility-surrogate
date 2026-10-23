"""Synthetic worlds for the network-native twin corpus (T-series)."""

from __future__ import annotations
import math
import zipfile
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_ORIGIN_X = 600000.0
DEFAULT_ORIGIN_Y = 5800000.0
STOP_NODE_CLEARANCE_M = 2.0
STOP_NUDGE_M = 3.0
GHOST_TERMINAL_STOPS = True
CUTOFF_S = 3500.0
WALK_SPEED_KMH = 6.5
UNREACHABLE_FLOOR_M = CUTOFF_S * WALK_SPEED_KMH / 3.6 + 150.0


@dataclass(frozen=True)
class WorldSpec:
    """Geometry and service parameters of one synthetic world."""

    tile_px: int = 200
    pixel_size_m: float = 105.0
    grid_spacing_m: float = 100.0
    apron_m: float = 7000.0
    origin_x: float = DEFAULT_ORIGIN_X
    origin_y: float = DEFAULT_ORIGIN_Y
    crs: str = "EPSG:32632"
    bus_speed_kmh: float = 40.0
    bus_dwell_s: float = 20.0
    bus_headway_s: float = 600.0
    rail_speed_kmh: float = 80.0
    rail_dwell_s: float = 30.0
    rail_headway_s: float = 1800.0
    service_start: str = "07:00:00"
    service_end: str = "09:00:00"
    workplace_side_m: float = 40.0
    corridor_extension_m: float = 300.0

    @property
    def tile_extent_m(self) -> float:
        return self.tile_px * self.pixel_size_m

    @property
    def minx(self) -> float:
        return self.origin_x

    @property
    def miny(self) -> float:
        return self.origin_y

    @property
    def maxy(self) -> float:
        return self.origin_y + self.tile_extent_m

    @property
    def maxx(self) -> float:
        return self.origin_x + self.tile_extent_m

    @property
    def world_bounds(self) -> tuple[float, float, float, float]:
        a = self.apron_m
        return (self.minx - a, self.miny - a, self.maxx + a, self.maxy + a)

    def to_xy(self, point: tuple[float, float]) -> tuple[float, float]:
        """(row, col) in fractional pixel index space -> projected (x, y)."""
        row, col = point
        return (
            self.minx + col * self.pixel_size_m,
            self.maxy - row * self.pixel_size_m,
        )

    def to_rowcol(self, xy: tuple[float, float]) -> tuple[float, float]:
        x, y = xy
        return (
            (self.maxy - y) / self.pixel_size_m,
            (x - self.minx) / self.pixel_size_m,
        )


@dataclass
class OsmNode:
    id: int
    x: float
    y: float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class OsmWay:
    id: int
    nodes: list[int]
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class StreetNetwork:
    nodes: dict[int, OsmNode]
    ways: list[OsmWay]
    corridor_way_ids: list[int]
    junction_node_ids: set[int]
    stop_node_ids: list[int]
    effective_stop_xy: list[tuple[float, float]] = field(default_factory=list)
    stop_nudges_m: list[float] = field(default_factory=list)


class _NodeStore:
    """Coordinate-keyed node allocator, so crossing ways share node ids."""

    def __init__(self, precision_m: float = 0.01) -> None:
        self._by_key: dict[tuple[int, int], int] = {}
        self.nodes: dict[int, OsmNode] = {}
        self._next = 1
        self._q = 1.0 / precision_m

    def key(self, x: float, y: float) -> tuple[int, int]:
        return (int(round(x * self._q)), int(round(y * self._q)))

    def get_or_create(
        self, x: float, y: float, tags: dict[str, str] | None = None
    ) -> int:
        k = self.key(x, y)
        nid = self._by_key.get(k)
        if nid is None:
            nid = self._next
            self._next += 1
            self._by_key[k] = nid
            self.nodes[nid] = OsmNode(nid, x, y, dict(tags or {}))
        elif tags:
            self.nodes[nid].tags.update(tags)
        return nid

    def new_id(self) -> int:
        nid = self._next
        self._next += 1
        return nid


def _seg_axis_intersections(
    p0: tuple[float, float], p1: tuple[float, float], value: float, axis: int
) -> list[tuple[float, float, float]]:
    """Intersections of a segment with the line {coord[axis] == value}."""
    a, b = (p0[axis], p1[axis])
    if a == b:
        return []
    t = (value - a) / (b - a)
    if not 0.0 <= t <= 1.0:
        return []
    x = p0[0] + t * (p1[0] - p0[0])
    y = p0[1] + t * (p1[1] - p0[1])
    return [(t, x, y)]


def _collinear(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    tol_m: float = 0.05,
) -> bool:
    """Is b on the straight line a-c, to within ``tol_m``?"""
    ax, ay = a
    bx, by = b
    cx, cy = c
    span = math.hypot(cx - ax, cy - ay)
    if span == 0:
        return True
    cross = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    return cross / span <= tol_m


def _extend_polyline(
    path: list[tuple[float, float]], extension_m: float
) -> list[tuple[float, float]]:
    """Prolong a polyline past both ends along its terminal segment directions."""
    if extension_m <= 0 or len(path) < 2:
        return list(path)

    def out(
        p_in: tuple[float, float], p_out: tuple[float, float]
    ) -> tuple[float, float]:
        dx, dy = (p_out[0] - p_in[0], p_out[1] - p_in[1])
        norm = math.hypot(dx, dy)
        if norm == 0:
            return p_out
        return (p_out[0] + extension_m * dx / norm, p_out[1] + extension_m * dy / norm)

    return [out(path[1], path[0])] + list(path) + [out(path[-2], path[-1])]


def build_street_network(
    spec: WorldSpec,
    corridors: Sequence[Sequence[tuple[float, float]]],
    stop_points: Sequence[tuple[float, float]],
    *,
    stops_as_road_nodes: bool = False,
) -> StreetNetwork:
    """Uniform grid + corridors, with a shared node at every crossing."""
    store = _NodeStore()
    minx, miny, maxx, maxy = spec.world_bounds
    step = spec.grid_spacing_m
    n_cols = int(math.floor((maxx - minx) / step)) + 1
    n_rows = int(math.floor((maxy - miny) / step)) + 1
    xs = [minx + i * step for i in range(n_cols)]
    ys = [miny + j * step for j in range(n_rows)]
    corridor_xy = [
        _extend_polyline([spec.to_xy(p) for p in c], spec.corridor_extension_m)
        for c in corridors
    ]
    on_vline: dict[int, list[tuple[float, float, float]]] = {}
    on_hline: dict[int, list[tuple[float, float, float]]] = {}
    on_corridor: list[list[tuple[float, float, float]]] = [[] for _ in corridor_xy]
    for ci, path in enumerate(corridor_xy):
        for si in range(len(path) - 1):
            p0, p1 = (path[si], path[si + 1])
            for i, gx in enumerate(xs):
                for t, x, y in _seg_axis_intersections(p0, p1, gx, 0):
                    on_vline.setdefault(i, []).append((y, x, y))
                    on_corridor[ci].append((si + t, x, y))
            for j, gy in enumerate(ys):
                for t, x, y in _seg_axis_intersections(p0, p1, gy, 1):
                    on_hline.setdefault(j, []).append((x, x, y))
                    on_corridor[ci].append((si + t, x, y))
    junction_ids: set[int] = set()
    for _, pts in list(on_vline.items()) + list(on_hline.items()):
        for _, x, y in pts:
            junction_ids.add(store.get_or_create(x, y))
    ways: list[OsmWay] = []
    grid_tags = {"highway": "residential", "surface": "asphalt"}
    for j, gy in enumerate(ys):
        pts = [(x, x, gy) for x in xs] + on_hline.get(j, [])
        pts.sort(key=lambda p: p[0])
        ids = _dedupe([store.get_or_create(x, y) for _, x, y in pts])
        if len(ids) >= 2:
            ways.append(OsmWay(store.new_id(), ids, dict(grid_tags)))
    for i, gx in enumerate(xs):
        pts = [(y, gx, y) for y in ys] + on_vline.get(i, [])
        pts.sort(key=lambda p: p[0])
        ids = _dedupe([store.get_or_create(x, y) for _, x, y in pts])
        if len(ids) >= 2:
            ways.append(OsmWay(store.new_id(), ids, dict(grid_tags)))
    stop_xy = [spec.to_xy(p) for p in stop_points]
    stop_ids: list[int] = []
    corridor_way_ids: list[int] = []
    for ci, path in enumerate(corridor_xy):
        entries = []
        for k in range(len(path)):
            if (
                not stops_as_road_nodes
                and any(
                    (
                        math.hypot(path[k][0] - sx, path[k][1] - sy) < 1.0
                        for sx, sy in stop_xy
                    )
                )
                and (0 < k < len(path) - 1)
                and _collinear(path[k - 1], path[k], path[k + 1])
            ):
                continue
            entries.append((float(k), *path[k]))
        entries += on_corridor[ci]
        if ci == 0 and stops_as_road_nodes:
            for sx, sy in stop_xy:
                t = _project_on_polyline(path, (sx, sy))
                entries.append((t, sx, sy))
        entries.sort(key=lambda e: e[0])
        ids = _dedupe([store.get_or_create(x, y) for _, x, y in entries])
        if len(ids) >= 2:
            wid = store.new_id()
            ways.append(OsmWay(wid, ids, {"highway": "tertiary", "surface": "asphalt"}))
            corridor_way_ids.append(wid)
    if stops_as_road_nodes:
        for sx, sy in stop_xy:
            stop_ids.append(store.get_or_create(sx, sy))
    node_xy = (
        np.array([[n.x, n.y] for n in store.nodes.values()]) if store.nodes else None
    )
    if stops_as_road_nodes or node_xy is None:
        effective, nudges = (list(stop_xy), [0.0] * len(stop_xy))
    else:
        effective, nudges = clear_stops_of_nodes(stop_xy, node_xy, corridor_xy[0])
    return StreetNetwork(
        store.nodes, ways, corridor_way_ids, junction_ids, stop_ids, effective, nudges
    )


def clear_stops_of_nodes(
    stop_xy: Sequence[tuple[float, float]],
    node_xy: "np.ndarray",
    tangent_path: Sequence[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[float]]:
    """Move any stop within STOP_NODE_CLEARANCE_M of a road node off it."""
    effective: list[tuple[float, float]] = []
    nudges: list[float] = []
    for sx, sy in stop_xy:
        d = float(np.hypot(node_xy[:, 0] - sx, node_xy[:, 1] - sy).min())
        if d >= STOP_NODE_CLEARANCE_M:
            effective.append((sx, sy))
            nudges.append(0.0)
            continue
        tx, ty = _tangent_at(tangent_path, (sx, sy))
        best, best_d, used = ((sx, sy), d, 0.0)
        for step in (STOP_NUDGE_M, -STOP_NUDGE_M, 2 * STOP_NUDGE_M, -2 * STOP_NUDGE_M):
            cx, cy = (sx + step * tx, sy + step * ty)
            dd = float(np.hypot(node_xy[:, 0] - cx, node_xy[:, 1] - cy).min())
            if dd > best_d:
                best, best_d, used = ((cx, cy), dd, abs(step))
            if best_d >= STOP_NODE_CLEARANCE_M:
                break
        effective.append(best)
        nudges.append(used)
    return (effective, nudges)


def _tangent_at(
    path: Sequence[tuple[float, float]], q: tuple[float, float]
) -> tuple[float, float]:
    """Unit tangent of ``path`` at the segment closest to ``q``."""
    t = _project_on_polyline(path, q)
    i = min(int(t), len(path) - 2)
    dx, dy = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
    norm = math.hypot(dx, dy) or 1.0
    return (dx / norm, dy / norm)


def _dedupe(ids: Iterable[int]) -> list[int]:
    out: list[int] = []
    for i in ids:
        if not out or out[-1] != i:
            out.append(i)
    return out


def _project_on_polyline(
    path: Sequence[tuple[float, float]], q: tuple[float, float]
) -> float:
    """Parameter (segment index + fraction) of the closest point on ``path``."""
    best_t, best_d = (0.0, float("inf"))
    for i in range(len(path) - 1):
        (x0, y0), (x1, y1) = (path[i], path[i + 1])
        dx, dy = (x1 - x0, y1 - y0)
        den = dx * dx + dy * dy
        t = (
            0.0
            if den == 0
            else max(0.0, min(1.0, ((q[0] - x0) * dx + (q[1] - y0) * dy) / den))
        )
        px, py = (x0 + t * dx, y0 + t * dy)
        d = (px - q[0]) ** 2 + (py - q[1]) ** 2
        if d < best_d:
            best_d, best_t = (d, i + t)
    return best_t


WORKPLACE_BUILDING_TAGS: dict[str, dict[str, str]] = {
    "wrk_office": {"building": "office", "office": "company"},
    "wrk_commercial": {"building": "commercial"},
    "wrk_industrial": {"building": "industrial"},
    "wrk_retail": {"building": "retail", "shop": "general"},
    "wrk_warehouse": {"building": "warehouse"},
    "wrk_greenhouse": {"building": "greenhouse"},
    "wrk_craft": {"building": "craft", "craft": "carpenter"},
    "wrk_emp_total": {},
}


def load_poi_tag_map(mapping_yaml: Path) -> dict[str, tuple[str, str]]:
    """class name -> the first (key, value) pair REFERENCE_ENGINE accepts for it."""
    out: dict[str, tuple[str, str]] = {}
    current: str | None = None
    in_mapping = False
    for raw in mapping_yaml.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            in_mapping = line.strip() == "mapping:"
            continue
        if not in_mapping:
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("- "):
            if current and current not in out:
                k, _, v = stripped[2:].partition(":")
                out[current] = (k.strip(), v.strip())
        elif stripped.endswith(":") and indent <= 2:
            current = stripped[:-1].strip()
    return out


def opportunity_features(
    spec: WorldSpec,
    channel_names: Sequence[str],
    points_by_channel: dict[int, list[tuple[float, float]]],
    tag_map: dict[str, tuple[str, str]],
    store_start_id: int,
) -> tuple[list[OsmNode], list[tuple[OsmWay, list[OsmNode]]], list[str]]:
    """Tagged nodes (amenities) and square building ways (workplaces)."""
    nodes: list[OsmNode] = []
    areas: list[tuple[OsmWay, list[OsmNode]]] = []
    skipped: list[str] = []
    nid = store_start_id
    half = spec.workplace_side_m / 2.0
    for ch, points in sorted(points_by_channel.items()):
        name = channel_names[ch]
        if name.startswith("wrk_"):
            tags = WORKPLACE_BUILDING_TAGS.get(name)
            if not tags:
                skipped.append(name)
                continue
            for row, col in points:
                cx, cy = spec.to_xy((row, col))
                corners = [
                    (cx - half, cy - half),
                    (cx + half, cy - half),
                    (cx + half, cy + half),
                    (cx - half, cy + half),
                ]
                ring: list[OsmNode] = []
                for x, y in corners:
                    ring.append(OsmNode(nid, x, y, {}))
                    nid += 1
                way = OsmWay(nid, [n.id for n in ring] + [ring[0].id], dict(tags))
                nid += 1
                areas.append((way, ring))
            continue
        pair = tag_map.get(name)
        if pair is None:
            skipped.append(name)
            continue
        key, value = pair
        for row, col in points:
            x, y = spec.to_xy((row, col))
            nodes.append(OsmNode(nid, x, y, {key: value}))
            nid += 1
    return (nodes, areas, skipped)


_TF_CACHE: dict[str, Any] = {}


def _transformer(crs: str):
    if crs not in _TF_CACHE:
        from pyproj import Transformer

        _TF_CACHE[crs] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return _TF_CACHE[crs]


def write_osm_xml(
    path: Path, spec: WorldSpec, nodes: Iterable[OsmNode], ways: Iterable[OsmWay]
) -> None:
    tf = _transformer(spec.crs)
    node_list = list(nodes)
    xs = [n.x for n in node_list]
    ys = [n.y for n in node_list]
    lons, lats = tf.transform(xs, ys)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write('<osm version="0.6" generator="surrogate-synthetic-world">\n')
        fh.write(
            f'  <bounds minlat="{min(lats):.7f}" minlon="{min(lons):.7f}" maxlat="{max(lats):.7f}" maxlon="{max(lons):.7f}"/>\n'
        )
        for n, lon, lat in zip(node_list, lons, lats):
            if n.tags:
                fh.write(
                    f'  <node id="{n.id}" version="1" lat="{lat:.9f}" lon="{lon:.9f}">\n'
                )
                for k, v in n.tags.items():
                    fh.write(f'    <tag k="{_esc(k)}" v="{_esc(v)}"/>\n')
                fh.write("  </node>\n")
            else:
                fh.write(
                    f'  <node id="{n.id}" version="1" lat="{lat:.9f}" lon="{lon:.9f}"/>\n'
                )
        for w in ways:
            fh.write(f'  <way id="{w.id}" version="1">\n')
            for ref in w.nodes:
                fh.write(f'    <nd ref="{ref}"/>\n')
            for k, v in w.tags.items():
                fh.write(f'    <tag k="{_esc(k)}" v="{_esc(v)}"/>\n')
            fh.write("  </way>\n")
        fh.write("</osm>\n")


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def convert_to_pbf(xml_path: Path, pbf_path: Path, osmium_bin: str = "osmium") -> Path:
    import subprocess

    pbf_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [osmium_bin, "cat", str(xml_path), "-o", str(pbf_path), "--overwrite"],
        check=True,
        capture_output=True,
    )
    return pbf_path


@dataclass
class TransitLine:
    route_id: str
    route_type: int
    stops: list[tuple[str, float, float]]
    speed_kmh: float
    dwell_s: float
    headway_s: float


def _hhmmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _parse_hhmmss(value: str) -> int:
    h, m, s = (int(p) for p in value.split(":"))
    return h * 3600 + m * 60 + s


def write_gtfs(path: Path, spec: WorldSpec, lines: Sequence[TransitLine]) -> Path:
    """Write a minimal but valid GTFS zip."""
    start = _parse_hhmmss(spec.service_start)
    end = _parse_hhmmss(spec.service_end)
    agency = "agency_id,agency_name,agency_url,agency_timezone\nSYN,Synthetic,https://example.invalid,Europe/Berlin\n"
    calendar = "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\nWEEK,1,1,1,1,1,0,0,20260101,20271231\n"
    stops_rows = ["stop_id,stop_name,stop_lat,stop_lon"]
    routes_rows = ["route_id,agency_id,route_short_name,route_long_name,route_type"]
    trips_rows = ["route_id,service_id,trip_id,direction_id"]
    times_rows = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence"]
    seen_stops: set[str] = set()
    for line in lines:
        if len(line.stops) < 2:
            continue
        routes_rows.append(
            f"{line.route_id},SYN,{line.route_id},{line.route_id},{line.route_type}"
        )
        for sid, x, y in line.stops:
            if sid in seen_stops:
                continue
            seen_stops.add(sid)
            lon, lat = _transformer(spec.crs).transform(x, y)
            stops_rows.append(f"{sid},{sid},{lat:.9f},{lon:.9f}")

        def cumulative(seq: Sequence[tuple[str, float, float]]) -> list[float]:
            offs = [0.0]
            for a, b in zip(seq[:-1], seq[1:]):
                dist = math.hypot(b[1] - a[1], b[2] - a[2])
                offs.append(offs[-1] + dist / (line.speed_kmh / 3.6) + line.dwell_s)
            return offs

        n_trips = max(1, int((end - start) // line.headway_s))
        for direction in (0, 1):
            seq_stops = line.stops if direction == 0 else list(reversed(line.stops))
            seq_offsets = cumulative(seq_stops)
            for k in range(n_trips):
                trip_id = f"{line.route_id}_d{direction}_t{k}"
                trips_rows.append(f"{line.route_id},WEEK,{trip_id},{direction}")
                t0 = start + k * line.headway_s
                for seq, ((sid, _, _), off) in enumerate(
                    zip(seq_stops, seq_offsets), start=1
                ):
                    stamp = _hhmmss(t0 + off)
                    times_rows.append(f"{trip_id},{stamp},{stamp},{sid},{seq}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("agency.txt", agency)
        z.writestr("calendar.txt", calendar)
        z.writestr("stops.txt", "\n".join(stops_rows) + "\n")
        z.writestr("routes.txt", "\n".join(routes_rows) + "\n")
        z.writestr("trips.txt", "\n".join(trips_rows) + "\n")
        z.writestr("stop_times.txt", "\n".join(times_rows) + "\n")
    return path


DEFAULT_RAIL_ROW = 10.0
RAIL_STOP_SPACING_M = 5000.0


def relocate_beyond_floor(
    focals: dict[int, tuple[float, float]],
    stop_xy: Sequence[tuple[float, float]],
    spec: WorldSpec,
    floor_m: float = UNREACHABLE_FLOOR_M,
) -> tuple[dict[int, tuple[float, float]], dict[str, Any]]:
    """Translate a focal cluster so every focal clears ``floor_m`` from every stop."""
    if not focals:
        return ({}, {"in_tile": True, "shift_m": 0.0, "min_stop_dist_m": None})
    origin = np.array(
        [
            np.mean([p[0] for p in focals.values()]),
            np.mean([p[1] for p in focals.values()]),
        ]
    )
    stops = np.asarray(stop_xy, dtype=float)
    margin = 16.0
    spread_px = max(
        (float(np.hypot(p[0] - origin[0], p[1] - origin[1])) for p in focals.values())
    )
    required_m = floor_m + spread_px * spec.pixel_size_m

    def scan(lo: float, hi: float) -> tuple[np.ndarray, float] | None:
        best, best_d2 = (None, np.inf)
        rows = np.arange(lo, hi + 1.0, 1.0)
        for row in rows:
            cols = rows
            xy = np.array([spec.to_xy((row, c)) for c in cols])
            d = np.sqrt(
                (xy[:, None, 0] - stops[None, :, 0]) ** 2
                + (xy[:, None, 1] - stops[None, :, 1]) ** 2
            ).min(axis=1)
            ok = np.flatnonzero(d >= required_m)
            if not ok.size:
                continue
            d2 = (row - origin[0]) ** 2 + (cols[ok] - origin[1]) ** 2
            k = int(np.argmin(d2))
            if d2[k] < best_d2:
                best_d2, best = (d2[k], np.array([row, cols[ok][k]]))
        return (best, float(np.sqrt(best_d2))) if best is not None else None

    found = scan(margin, spec.tile_px - 1 - margin)
    in_tile = found is not None
    if found is None:
        apron_px = spec.apron_m / spec.pixel_size_m
        found = scan(-apron_px + margin, spec.tile_px - 1 + apron_px - margin)
    if found is None:
        raise RuntimeError(
            "no location clears the unreachability floor, even in the apron"
        )
    target, shift_px = found
    delta = target - origin
    moved = {ch: (p[0] + delta[0], p[1] + delta[1]) for ch, p in focals.items()}
    moved_xy = np.array([spec.to_xy(p) for p in moved.values()])
    min_d = float(
        np.sqrt(
            (moved_xy[:, None, 0] - stops[None, :, 0]) ** 2
            + (moved_xy[:, None, 1] - stops[None, :, 1]) ** 2
        ).min()
    )
    return (
        moved,
        {
            "in_tile": bool(in_tile),
            "shift_m": float(shift_px * spec.pixel_size_m),
            "min_stop_dist_m": min_d,
            "floor_m": float(floor_m),
            "clears_floor": bool(min_d >= floor_m),
        },
    )


def _rail_line(
    spec: WorldSpec, anchor: tuple[float, float] | None, with_anchor_stop: bool
) -> TransitLine:
    """A straight rail line spanning the world, stops every 5 km."""
    row = anchor[0] if anchor is not None else DEFAULT_RAIL_ROW
    minx, _, maxx, _ = spec.world_bounds
    _, y = spec.to_xy((row, 0.0))
    stops: list[tuple[str, float, float]] = []
    n = int((maxx - minx) // RAIL_STOP_SPACING_M) + 1
    for k in range(n):
        x = minx + k * RAIL_STOP_SPACING_M
        stops.append((f"RAIL_{k}", x, y))
    if anchor is not None and with_anchor_stop:
        ax, ay = spec.to_xy(anchor)
        stops.append(("RAIL_XFER", ax, ay))
        stops.sort(key=lambda s: s[1])
    return TransitLine(
        "RAIL1", 2, stops, spec.rail_speed_kmh, spec.rail_dwell_s, spec.rail_headway_s
    )


def _with_ghost_terminals(
    stop_xy: Sequence[tuple[float, float]],
    spec: WorldSpec,
    net: StreetNetwork | None = None,
) -> list[tuple[float, float]]:
    """Prepend and append one stop beyond each terminus. See GHOST_TERMINAL_STOPS."""
    if not GHOST_TERMINAL_STOPS or len(stop_xy) < 2:
        return list(stop_xy)
    extended = _extend_polyline(list(stop_xy), spec.corridor_extension_m)
    if net is None or not net.nodes:
        return extended
    node_xy = np.array([[n.x, n.y] for n in net.nodes.values()])

    def snap(q: tuple[float, float]) -> tuple[float, float]:
        i = int(np.argmin(np.hypot(node_xy[:, 0] - q[0], node_xy[:, 1] - q[1])))
        return (float(node_xy[i, 0]), float(node_xy[i, 1]))

    return [snap(extended[0])] + list(stop_xy) + [snap(extended[-1])]


def _bus_line_xy(
    stop_xy: Sequence[tuple[float, float]],
    route_id: str = "BUS1",
    spec: WorldSpec | None = None,
) -> TransitLine:
    s = spec or WorldSpec()
    stops = [(f"{route_id}_S{i}", x, y) for i, (x, y) in enumerate(stop_xy)]
    return TransitLine(
        route_id, 3, stops, s.bus_speed_kmh, s.bus_dwell_s, s.bus_headway_s
    )


@dataclass
class ConditionWorld:
    """Everything one engine run needs, plus its provenance."""

    set_index: int
    family: str
    condition: str
    analytic_zero: bool
    region_key: str
    pbf: Path
    gtfs_bus: Path
    gtfs_rail: Path
    gtfs_all: Path
    audit: dict[str, Any]
    pbf_sha256: str


def build_condition_world(
    scene: Any,
    set_index: int,
    condition: str,
    out_dir: Path,
    spec: WorldSpec,
    channel_names: Sequence[str],
    tag_map: dict[str, tuple[str, str]],
    osmium_bin: str = "osmium",
) -> ConditionWorld:
    """Author one condition's world and write its PBF and GTFS feeds."""
    import hashlib

    base, _, marker = condition.partition(":")
    analytic_zero = marker == "analytic_zero"
    corridors: list[Sequence[tuple[float, float]]] = [list(scene.corridor)]
    if base == "near_overlap" and scene.overlap_corridor:
        corridors.append(list(scene.overlap_corridor))
    if base == "stub" and scene.stub_corridor:
        corridors = [list(scene.stub_corridor)]
    net = build_street_network(spec, corridors, scene.stop_points)
    points: dict[int, list[tuple[float, float]]] = {
        ch: list(pts) for ch, pts in scene.poi_background.items()
    }
    relocation: dict[str, Any] = {}
    if base == "displaced_far":
        focals, relocation = relocate_beyond_floor(
            scene.focals.get("displaced") or {},
            [spec.to_xy(p) for p in scene.stop_points],
            spec,
        )
    else:
        focals = scene.focals.get(base) or scene.focals.get(condition) or {}
    for ch, pt in focals.items():
        points.setdefault(int(ch), []).append(pt)
    next_id = max(net.nodes) + 1
    opp_nodes, opp_areas, skipped = opportunity_features(
        spec, channel_names, points, tag_map, next_id
    )
    all_nodes = list(net.nodes.values()) + opp_nodes
    all_ways = list(net.ways)
    for way, ring in opp_areas:
        all_nodes.extend(ring)
        all_ways.append(way)
    cond_dir = out_dir / f"set{set_index:03d}" / scene.family / base
    cond_dir.mkdir(parents=True, exist_ok=True)
    xml = cond_dir / "world.osm"
    write_osm_xml(xml, spec, all_nodes, all_ways)
    pbf = convert_to_pbf(xml, cond_dir / "world.osm.pbf", osmium_bin=osmium_bin)
    xml.unlink()
    bus_lines: list[TransitLine] = []
    if not analytic_zero:
        bus_lines.append(
            _bus_line_xy(_with_ghost_terminals(net.effective_stop_xy, spec, net))
        )
        if base == "near_overlap" and scene.overlap_corridor:
            dx = 1.25 * spec.pixel_size_m
            shifted = [(x - dx, y - dx) for x, y in net.effective_stop_xy]
            node_xy = np.array([[n.x, n.y] for n in net.nodes.values()])
            overlap_xy = [spec.to_xy(p) for p in scene.overlap_corridor]
            shifted, _ = clear_stops_of_nodes(shifted, node_xy, overlap_xy)
            bus_lines.append(
                _bus_line_xy(_with_ghost_terminals(shifted, spec, net), route_id="BUS2")
            )
    with_anchor = base != "severed_rail_stop"
    rail = _rail_line(spec, scene.rail_anchor, with_anchor)
    gtfs_bus = write_gtfs(cond_dir / "gtfs_bus.zip", spec, bus_lines)
    gtfs_rail = write_gtfs(cond_dir / "gtfs_rail.zip", spec, [rail])
    gtfs_all = write_gtfs(cond_dir / "gtfs_all.zip", spec, bus_lines + [rail])
    audit = audit_world(spec, net, bus_lines, skipped_channels=skipped)
    audit["relocation"] = relocation
    audit["n_ghost_terminal_stops"] = 2 * len(bus_lines) if GHOST_TERMINAL_STOPS else 0
    audit["stop_nudges_m"] = list(net.stop_nudges_m)
    audit["max_stop_nudge_m"] = max(net.stop_nudges_m) if net.stop_nudges_m else 0.0
    audit["n_opportunity_nodes"] = len(opp_nodes)
    audit["n_workplace_areas"] = len(opp_areas)
    audit["analytic_zero"] = analytic_zero
    digest = hashlib.sha256(pbf.read_bytes()).hexdigest()
    return ConditionWorld(
        set_index=set_index,
        family=scene.family,
        condition=condition,
        analytic_zero=analytic_zero,
        region_key=f"twin_s{set_index:03d}_{scene.family}_{base}",
        pbf=pbf,
        gtfs_bus=gtfs_bus,
        gtfs_rail=gtfs_rail,
        gtfs_all=gtfs_all,
        audit=audit,
        pbf_sha256=digest,
    )


def audit_world(
    spec: WorldSpec,
    net: StreetNetwork,
    bus_lines: Sequence[TransitLine],
    *,
    walk_connection_m: float = 150.0,
    skipped_channels: Sequence[str] = (),
) -> dict[str, Any]:
    """Assert the things that fail silently rather than loudly."""
    degree: dict[int, int] = {}
    for w in net.ways:
        for nid in set(w.nodes):
            degree[nid] = degree.get(nid, 0) + 1
    bad_junctions = [nid for nid in net.junction_node_ids if degree.get(nid, 0) < 2]
    corridor_nodes: set[int] = set()
    for w in net.ways:
        if w.id in net.corridor_way_ids:
            corridor_nodes.update(w.nodes)
    shared = corridor_nodes & net.junction_node_ids
    node_xy = np.array([[n.x, n.y] for n in net.nodes.values()]) if net.nodes else None

    def gap_to_road(x: float, y: float) -> float:
        if node_xy is None:
            return float("inf")
        return float(np.hypot(node_xy[:, 0] - x, node_xy[:, 1] - y).min())

    stop_gaps: list[float] = []
    ghost_gaps: list[float] = []
    for line in bus_lines:
        for idx, (_, x, y) in enumerate(line.stops):
            g = gap_to_road(x, y)
            is_ghost = GHOST_TERMINAL_STOPS and idx in (0, len(line.stops) - 1)
            (ghost_gaps if is_ghost else stop_gaps).append(g)
    zero_gap = sum((1 for g in stop_gaps if g < 1e-06))
    boardable_ghosts = sum((1 for g in ghost_gaps if g >= 1e-06))
    errors: list[str] = []
    if bad_junctions:
        errors.append(f"{len(bad_junctions)} corridor crossings are not shared nodes")
    if net.corridor_way_ids and (not shared):
        errors.append("corridor shares no node with the grid: it is an isolated line")
    if stop_gaps and max(stop_gaps) > walk_connection_m:
        errors.append(
            f"a stop is {max(stop_gaps):.1f} m from any walk node (> {walk_connection_m})"
        )
    if zero_gap:
        errors.append(
            f"{zero_gap} scene stop(s) coincide exactly with a road node (zero-length connector)"
        )
    if boardable_ghosts:
        errors.append(
            f"{boardable_ghosts} ghost terminal(s) are boardable; they must contribute nothing"
        )
    if skipped_channels:
        unexpected = [c for c in skipped_channels if c != "wrk_emp_total"]
        if unexpected:
            errors.append(f"channels with no authorable tag: {sorted(set(unexpected))}")
    return {
        "n_nodes": len(net.nodes),
        "n_ways": len(net.ways),
        "n_corridor_crossings": len(net.junction_node_ids),
        "n_corridor_grid_shared_nodes": len(shared),
        "max_stop_to_walk_node_m": max(stop_gaps) if stop_gaps else None,
        "min_stop_to_walk_node_m": min(stop_gaps) if stop_gaps else None,
        "n_zero_length_connectors": zero_gap,
        "n_boardable_ghost_terminals": boardable_ghosts,
        "skipped_channels": sorted(set(skipped_channels)),
        "errors": errors,
    }
