
try:
    import h3
except ImportError:
    h3 = None

try:
    from shapely.geometry import Polygon, mapping
except ImportError:
    Polygon = mapping = None


def id_to_polygon(h3_id: str) -> Polygon:
    try:
        boundary = h3.cell_to_boundary(h3_id, geo_json=True)
        return Polygon(boundary)
    except AttributeError:
        boundary = h3.h3_to_geo_boundary(h3_id, geo_json=True)
        return Polygon(boundary)
    except TypeError:
        try:
            boundary = h3.cell_to_boundary(h3_id)
        except AttributeError:
            boundary = h3.h3_to_geo_boundary(h3_id)
        boundary = [(lng, lat) for lat, lng in boundary]
        return Polygon(boundary)


def estimate_h3_cell_area_m2(resolution: int) -> float:
    """Return approximate area of an H3 cell at a given resolution in m^2."""
    try:
        return float(h3.average_hexagon_area(resolution, unit="m^2"))
    except Exception:
        try:
            return float(h3.hex_area(resolution, unit="m^2"))
        except Exception:
            cell = h3.latlng_to_cell(0.0, 0.0, resolution)
            try:
                return float(h3.cell_area(cell, unit="m^2"))
            except Exception:
                return float(h3.h3_cell_area(cell, unit="m^2"))


def _polyfill_geom(geom, resolution: int):
    """Polyfill a shapely geometry (Polygon or MultiPolygon) with H3 cells.
    Handles h3-py v4+ API differences by constructing LatLngPoly objects manually."""
    try:
        if geom.geom_type == "Polygon":
            ext_coords = [(lat, lon) for lon, lat in geom.exterior.coords]
            holes = []
            for interior in geom.interiors:
                holes.append([(lat, lon) for lon, lat in interior.coords])
            h3_poly = h3.LatLngPoly(ext_coords, *holes)
            return h3.polygon_to_cells(h3_poly, resolution)
        elif geom.geom_type == "MultiPolygon":
            all_cells = set()
            for poly in geom.geoms:
                ext_coords = [(lat, lon) for lon, lat in poly.exterior.coords]
                holes = []
                for interior in poly.interiors:
                    holes.append([(lat, lon) for lon, lat in interior.coords])
                h3_poly = h3.LatLngPoly(ext_coords, *holes)
                all_cells.update(h3.polygon_to_cells(h3_poly, resolution))
            return all_cells
        else:
            if geom.is_empty:
                return set()
            raise ValueError(
                f"Unsupported geometry type for polyfill: {geom.geom_type}"
            )
    except (AttributeError, NameError):
        geojson = mapping(geom)
        try:
            return h3.polyfill_geojson(geojson, resolution)
        except AttributeError:
            return h3.polyfill(geojson, resolution, geo_json_conformant=True)


