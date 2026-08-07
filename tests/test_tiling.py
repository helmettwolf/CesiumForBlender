import math

import numpy as np

from cesium_for_blender.core import tiling, wgs84


def test_level_dims():
    assert tiling.level_dims(0) == (2, 1)
    assert tiling.level_dims(1) == (4, 2)
    assert tiling.level_dims(10) == (2048, 1024)


def test_tile_rect_roots():
    w, s, e, n = tiling.tile_rect(0, 0, 0)
    assert math.isclose(w, -math.pi) and math.isclose(e, 0.0)
    assert math.isclose(s, -math.pi / 2) and math.isclose(n, math.pi / 2)
    w, s, e, n = tiling.tile_rect(0, 1, 0)
    assert math.isclose(w, 0.0) and math.isclose(e, math.pi)


def test_children_parent_roundtrip():
    key = (7, 100, 42)
    for child in tiling.children(*key):
        assert tiling.parent(*child) == key
    assert tiling.parent(0, 0, 0) is None


def test_lonlat_to_tile():
    # lon -65, lat -2 is in the west-hemisphere root
    assert tiling.lonlat_to_tile(0, -65.0, -2.0) == (0, 0, 0)
    assert tiling.lonlat_to_tile(0, 65.0, -2.0) == (0, 1, 0)
    z, x, y = tiling.lonlat_to_tile(9, -65.0, -2.0)
    w, s, e, n = tiling.tile_rect(z, x, y)
    assert w <= math.radians(-65.0) < e
    assert s <= math.radians(-2.0) < n


def test_imagery_uv_transform_identity_below_max():
    key, scale, uo, vo = tiling.imagery_key_and_uv_transform(13, 5000, 3000)
    assert key == (13, 5000, 3000) and scale == 1.0 and uo == 0.0 and vo == 0.0


def test_imagery_uv_transform_ancestor():
    # z14 tile -> z13 ancestor, quarter window
    key, scale, uo, vo = tiling.imagery_key_and_uv_transform(14, 1001, 501)
    assert key == (13, 500, 250)
    assert scale == 0.5
    assert (uo, vo) == (0.5, 0.5)  # odd x, odd y -> NE quadrant
    key, scale, uo, vo = tiling.imagery_key_and_uv_transform(16, 4000, 2001)
    assert key == (13, 500, 250)
    assert scale == 0.125
    assert math.isclose(uo, 0.0) and math.isclose(vo, 0.125)


def test_availability_convention_detection():
    # This server's rects only fit under available[i] -> level z+1+i: endY=1
    # overflows level 0's single row, proving the subtree convention.
    avail = tiling.AvailabilityIndex()
    root_meta = [
        [{"startX": 0, "startY": 0, "endX": 1, "endY": 1}],   # level 1 (4x2 grid)
        [{"startX": 0, "startY": 0, "endX": 3, "endY": 3}],   # level 2 (8x4 grid)
    ]
    assert avail._detect_offset(0, root_meta) == 1
    # CWT-style rects are structurally ambiguous (small rects always also fit
    # one level deeper), so detection defaults to the subtree convention; the
    # live-probe test in test_availability.py is the real ground-truth check.
    cwt_meta = [
        [{"startX": 0, "startY": 0, "endX": 1, "endY": 0}],   # fits level 0 AND 1
    ]
    assert avail._detect_offset(0, cwt_meta) == 1


def test_availability_ingest_and_query():
    avail = tiling.AvailabilityIndex()
    assert avail.is_available(0, 0, 0) and avail.is_available(0, 1, 0)
    assert not avail.is_available(0, 2, 0)
    assert not avail.is_available(3, 1, 1)
    gen0 = avail.generation
    avail.ingest((0, 0, 0), [[{"startX": 0, "startY": 0, "endX": 1, "endY": 1}]])
    assert avail.generation == gen0 + 1
    assert avail.is_available(1, 1, 1)
    assert not avail.is_available(1, 2, 0)
    # duplicate ingest is a no-op
    assert not avail.ingest((0, 0, 0), [[{"startX": 0, "startY": 0, "endX": 3, "endY": 1}]])
    assert avail.generation == gen0 + 1


def test_estimate_tile_aabb_contains_center():
    frame = wgs84.EnuFrame(-2.0, -65.0)
    key = tiling.lonlat_to_tile(9, -65.0, -2.0)
    lo, hi = tiling.estimate_tile_aabb(*key, frame)
    c = frame.geodetic_to_enu(math.radians(-65.0), math.radians(-2.0), 100.0)
    assert (lo <= c).all() and (c <= hi).all()
    # tile is ~40 km wide at z9; AABB must be sane, not planet-sized
    assert (hi[:2] - lo[:2]).max() < 200_000
