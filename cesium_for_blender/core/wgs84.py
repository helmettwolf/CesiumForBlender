"""WGS84 ellipsoid math: geodetic -> ECEF -> local ENU frames.

Pure module: numpy only, no bpy. All angles in radians unless the name says
_deg. All computation in float64; callers cast to float32 at the very end.
"""

from __future__ import annotations

import math

import numpy as np

A = 6378137.0                      # semi-major axis (m)
F = 1.0 / 298.257223563            # flattening
E2 = F * (2.0 - F)                 # first eccentricity squared


def geodetic_to_ecef(lon_rad, lat_rad, h_m) -> np.ndarray:
    """Vectorized geodetic -> ECEF. Inputs broadcast; returns (..., 3) float64."""
    lon = np.asarray(lon_rad, dtype=np.float64)
    lat = np.asarray(lat_rad, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = A / np.sqrt(1.0 - E2 * sin_lat * sin_lat)
    x = (n + h) * cos_lat * np.cos(lon)
    y = (n + h) * cos_lat * np.sin(lon)
    z = (n * (1.0 - E2) + h) * sin_lat
    return np.stack([x, y, z], axis=-1)


class EnuFrame:
    """Local east-north-up tangent frame anchored at (lat0, lon0, h0).

    Blender mapping: +X = east, +Y = north, +Z = up. ENU is a rigid transform
    of ECEF, so globe curvature is preserved exactly.
    """

    def __init__(self, lat0_deg: float, lon0_deg: float, h0_m: float = 0.0):
        self.lat0_deg = float(lat0_deg)
        self.lon0_deg = float(lon0_deg)
        self.h0_m = float(h0_m)
        lat0 = math.radians(self.lat0_deg)
        lon0 = math.radians(self.lon0_deg)
        self.origin_ecef = geodetic_to_ecef(lon0, lat0, self.h0_m)
        sl, cl = math.sin(lon0), math.cos(lon0)
        sp, cp = math.sin(lat0), math.cos(lat0)
        self.rot = np.array(
            [
                [-sl, cl, 0.0],
                [-sp * cl, -sp * sl, cp],
                [cp * cl, cp * sl, sp],
            ],
            dtype=np.float64,
        )

    def ecef_to_enu(self, ecef: np.ndarray) -> np.ndarray:
        ecef = np.asarray(ecef, dtype=np.float64)
        return (ecef - self.origin_ecef) @ self.rot.T

    def geodetic_to_enu(self, lon_rad, lat_rad, h_m) -> np.ndarray:
        return self.ecef_to_enu(geodetic_to_ecef(lon_rad, lat_rad, h_m))
