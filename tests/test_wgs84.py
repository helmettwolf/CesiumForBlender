import math

import numpy as np

from cesium_for_blender.core import wgs84


def test_ecef_known_points():
    # equator / prime meridian at h=0 -> (a, 0, 0)
    p = wgs84.geodetic_to_ecef(0.0, 0.0, 0.0)
    assert np.allclose(p, [wgs84.A, 0.0, 0.0], atol=1e-6)
    # north pole -> (0, 0, b), b = a(1-f)
    b = wgs84.A * (1.0 - wgs84.F)
    p = wgs84.geodetic_to_ecef(0.0, math.pi / 2, 0.0)
    assert np.allclose(p, [0.0, 0.0, b], atol=1e-6)
    # lon 90E on the equator -> (0, a, 0)
    p = wgs84.geodetic_to_ecef(math.pi / 2, 0.0, 0.0)
    assert np.allclose(p, [0.0, wgs84.A, 0.0], atol=1e-6)


def test_enu_frame_axes():
    frame = wgs84.EnuFrame(lat0_deg=-2.0, lon0_deg=-65.0)
    # origin maps to (0,0,0)
    o = frame.geodetic_to_enu(math.radians(-65.0), math.radians(-2.0), 0.0)
    assert np.allclose(o, [0.0, 0.0, 0.0], atol=1e-9)
    # straight up 100 m -> +Z
    up = frame.geodetic_to_enu(math.radians(-65.0), math.radians(-2.0), 100.0)
    assert np.allclose(up, [0.0, 0.0, 100.0], atol=1e-6)
    # a point slightly east -> +X dominant; slightly north -> +Y dominant
    east = frame.geodetic_to_enu(math.radians(-64.999), math.radians(-2.0), 0.0)
    assert east[0] > 100.0 and abs(east[1]) < 1.0
    north = frame.geodetic_to_enu(math.radians(-65.0), math.radians(-1.999), 0.0)
    assert north[1] > 100.0 and abs(north[0]) < 1.0


def test_enu_distance_scale():
    # 1 degree of latitude ~ 110.57 km near the equator
    frame = wgs84.EnuFrame(0.0, 0.0)
    p = frame.geodetic_to_enu(0.0, math.radians(1.0), 0.0)
    d = np.linalg.norm(p)
    assert 110_000 < d < 111_500
