"""Geographic (EPSG:4326) TMS tiling scheme + terrain availability index.

Pure module: numpy + stdlib only, no bpy.

Scheme facts (verified against the live server's layer.json/tilemapresource.xml):
- Level z has 2^(z+1) columns and 2^z rows; two root tiles at z0 (x=0 west
  hemisphere, x=1 east hemisphere).
- y = 0 at the SOUTH edge (TMS origin -180,-90).
"""

from __future__ import annotations

import math
import threading

import numpy as np

from . import wgs84

TileKey = tuple  # (z, x, y)

IMAGERY_MAX_Z_DEFAULT = 13

# Conservative pre-decode height band (m) for tiles whose true min/max height
# is not yet known. Loose culling early; exact after decode.
FALLBACK_MIN_H = -1000.0
FALLBACK_MAX_H = 9000.0


def level_dims(z: int) -> tuple[int, int]:
    """(columns, rows) at level z."""
    return (2 << z, 1 << z)


def tile_rect(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(west, south, east, north) in radians."""
    cols, rows = level_dims(z)
    dlon = (2.0 * math.pi) / cols
    dlat = math.pi / rows
    west = -math.pi + x * dlon
    south = -0.5 * math.pi + y * dlat
    return (west, south, west + dlon, south + dlat)


def children(z: int, x: int, y: int) -> list[TileKey]:
    cz, cx, cy = z + 1, x * 2, y * 2
    return [(cz, cx, cy), (cz, cx + 1, cy), (cz, cx, cy + 1), (cz, cx + 1, cy + 1)]


def parent(z: int, x: int, y: int) -> TileKey | None:
    if z == 0:
        return None
    return (z - 1, x >> 1, y >> 1)


def lonlat_to_tile(z: int, lon_deg: float, lat_deg: float) -> TileKey:
    cols, rows = level_dims(z)
    x = int((lon_deg + 180.0) / 360.0 * cols)
    y = int((lat_deg + 90.0) / 180.0 * rows)
    return (z, min(max(x, 0), cols - 1), min(max(y, 0), rows - 1))


def uv_transform_to_ancestor(
    z: int, x: int, y: int, ancestor_z: int
) -> tuple[TileKey, float, float, float]:
    """UV transform mapping tile (z,x,y)'s unit square onto its ancestor at
    ancestor_z. Returns ((az, ax, ay), scale, u_off, v_off) with
    uv' = uv * scale + off. Identity when ancestor_z >= z."""
    dz = z - ancestor_z
    if dz <= 0:
        return (z, x, y), 1.0, 0.0, 0.0
    ax, ay = x >> dz, y >> dz
    scale = 1.0 / (1 << dz)
    u_off = (x - (ax << dz)) * scale
    v_off = (y - (ay << dz)) * scale
    return (ancestor_z, ax, ay), scale, u_off, v_off


def imagery_key_and_uv_transform(
    z: int, x: int, y: int, imagery_max_z: int = IMAGERY_MAX_Z_DEFAULT
) -> tuple[TileKey, float, float, float]:
    """Imagery tile draping terrain tile (z,x,y) and the UV transform onto it.
    Terrain and imagery share the same tiling scheme, so this is identity for
    z <= imagery_max_z and an ancestor sub-window above it."""
    return uv_transform_to_ancestor(z, x, y, min(z, imagery_max_z))


def estimate_tile_aabb(
    z: int,
    x: int,
    y: int,
    frame: "wgs84.EnuFrame",
    min_h: float = FALLBACK_MIN_H,
    max_h: float = FALLBACK_MAX_H,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservative ENU AABB for a tile from a 5x5 lon/lat sample grid at two
    heights. The grid (not just corners) captures the ellipsoid bulge of large
    low-level tiles. Returns (min_xyz, max_xyz) float64."""
    west, south, east, north = tile_rect(z, x, y)
    lons = np.linspace(west, east, 5)
    lats = np.linspace(south, north, 5)
    lon_g, lat_g = np.meshgrid(lons, lats)
    lon_f = np.concatenate([lon_g.ravel(), lon_g.ravel()])
    lat_f = np.concatenate([lat_g.ravel(), lat_g.ravel()])
    h_f = np.concatenate([np.full(25, min_h), np.full(25, max_h)])
    enu = frame.geodetic_to_enu(lon_f, lat_f, h_f)
    return enu.min(axis=0), enu.max(axis=0)


class AvailabilityIndex:
    """Merged tile-availability rectangles, fed by metadata extensions.

    Server convention (verified empirically): a metadata-carrying tile at level
    L stores available[i] = rects for absolute level L + 1 + i, covering its
    own subtree only. Rects are {startX, startY, endX, endY}, INCLUSIVE,
    absolute tile coordinates. Falls back to the Cesium World Terrain
    convention (available[i] = level i... i.e. L + i) if the subtree convention
    produces out-of-range rects.

    Thread-safe: workers ingest during decode, main thread queries.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._rects: dict[int, list[np.ndarray]] = {}
        self._merged: dict[int, np.ndarray] = {}
        self._ingested: set[TileKey] = set()
        self._max_known = 0
        self.generation = 0  # bumped on every new ingest; streamer watches it

    @staticmethod
    def _detect_offset(tile_z: int, available: list) -> int:
        """1 for the subtree convention (available[0] -> tile_z+1), 0 for the
        absolute/CWT convention. Decided by whichever keeps every rect within
        its level's grid."""
        for off in (1, 0):
            ok = True
            for i, rects in enumerate(available):
                cols, rows = level_dims(tile_z + off + i)
                for r in rects:
                    if r["endX"] >= cols or r["endY"] >= rows:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return off
        return 1  # malformed either way; prefer the observed server convention

    def ingest(self, key: TileKey, available: list) -> bool:
        """Merge a tile's metadata 'available' array. Returns True if new."""
        if not available:
            return False
        z = key[0]
        with self._lock:
            if key in self._ingested:
                return False
            self._ingested.add(key)
            off = self._detect_offset(z, available)
            for i, rects in enumerate(available):
                if not rects:
                    continue
                lvl = z + off + i
                arr = np.array(
                    [[r["startX"], r["startY"], r["endX"], r["endY"]] for r in rects],
                    dtype=np.int64,
                )
                self._rects.setdefault(lvl, []).append(arr)
                self._merged.pop(lvl, None)
                self._max_known = max(self._max_known, lvl)
            self.generation += 1
        return True

    def _level_array(self, z: int) -> np.ndarray | None:
        merged = self._merged.get(z)
        if merged is None:
            parts = self._rects.get(z)
            if not parts:
                return None
            merged = parts[0] if len(parts) == 1 else np.concatenate(parts)
            self._merged[z] = merged
        return merged

    def is_available(self, z: int, x: int, y: int) -> bool:
        if z == 0:
            return x in (0, 1) and y == 0  # both roots always exist
        with self._lock:
            arr = self._level_array(z)
            if arr is None:
                return False
            return bool(
                (
                    (arr[:, 0] <= x)
                    & (x <= arr[:, 2])
                    & (arr[:, 1] <= y)
                    & (y <= arr[:, 3])
                ).any()
            )

    def has_ingested(self, key: TileKey) -> bool:
        with self._lock:
            return key in self._ingested

    def max_known_level(self) -> int:
        with self._lock:
            return self._max_known

    def deepest_coverage_center(self) -> tuple[float, float] | None:
        """(lat_deg, lon_deg) of the center of the LARGEST availability rect at
        the deepest known level — the 'go to data center' target. Using a
        single rect (not the centroid of all rects) guarantees the point is
        inside actual coverage even when the dataset has several disjoint
        regions."""
        with self._lock:
            if self._max_known == 0:
                return None
            z = self._max_known
            arr = self._level_array(z)
            if arr is None:
                return None
            areas = (arr[:, 2] - arr[:, 0] + 1) * (arr[:, 3] - arr[:, 1] + 1)
            i = int(np.argmax(areas))
            cx = float(arr[i, 0] + arr[i, 2] + 1) / 2.0
            cy = float(arr[i, 1] + arr[i, 3] + 1) / 2.0
        cols, rows = level_dims(z)
        lon = -180.0 + cx / cols * 360.0
        lat = -90.0 + cy / rows * 180.0
        return (lat, lon)
