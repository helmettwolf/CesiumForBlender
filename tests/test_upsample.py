"""Synthetic-tile upsampling: subdividing past the dataset's resolution
limit by resampling the nearest REAL ancestor's TIN (CesiumJS-style)."""

import numpy as np
import pytest

from cesium_for_blender.core import quantized_mesh as qmesh

Q = qmesh.QUANT_MAX


def plane_tile(min_h=0.0, max_h=100.0):
    """Unit-square ancestor whose height is a plane: h_m = max_h * u."""
    u = np.array([0, Q, 0, Q], dtype=np.int32)
    v = np.array([0, 0, Q, Q], dtype=np.int32)
    h = np.array([0, Q, 0, Q], dtype=np.int32)     # h tracks u
    return qmesh.QMTile(
        center_ecef=np.zeros(3), min_h=min_h, max_h=max_h,
        bs_center=np.zeros(3), bs_radius=1.0, horizon_occlusion=np.zeros(3),
        u=u, v=v, h=h,
        indices=np.array([[0, 1, 3], [0, 3, 2]], dtype=np.uint32),
        west_i=np.array([0, 2], dtype=np.uint32),
        south_i=np.array([0, 1], dtype=np.uint32),
        east_i=np.array([1, 3], dtype=np.uint32),
        north_i=np.array([2, 3], dtype=np.uint32),
    )


def heights_m(tile):
    return tile.min_h + tile.h.astype(np.float64) / Q * (tile.max_h - tile.min_h)


def test_upsample_plane_child_corners():
    anc = plane_tile()
    up = qmesh.upsample(anc, (5, 10, 10), (6, 21, 20), grid_n=5)
    h = heights_m(up)
    n = 5
    # child covers u in [0.5, 1.0]: SW corner 50 m, SE corner 100 m
    assert h[0] == pytest.approx(50.0, abs=0.05)          # r0 c0
    assert h[n - 1] == pytest.approx(100.0, abs=0.05)     # r0 c4
    assert h[(n - 1) * n] == pytest.approx(50.0, abs=0.05)
    assert up.min_h == pytest.approx(50.0, abs=0.05)
    assert up.max_h == pytest.approx(100.0, abs=0.05)
    assert up.indices.shape[0] == 2 * (n - 1) ** 2


def test_upsample_two_levels_from_real_ancestor():
    anc = plane_tile()
    up = qmesh.upsample(anc, (5, 10, 10), (7, 43, 40), grid_n=5)
    h = heights_m(up)
    # dz=2, x offset 3 -> u in [0.75, 1.0]
    assert h[0] == pytest.approx(75.0, abs=0.05)
    assert h[4] == pytest.approx(100.0, abs=0.05)


def test_upsample_sibling_edges_watertight():
    anc = plane_tile()
    n = 9
    left = qmesh.upsample(anc, (5, 10, 10), (6, 20, 20), grid_n=n)
    right = qmesh.upsample(anc, (5, 10, 10), (6, 21, 20), grid_n=n)
    hl = heights_m(left).reshape(n, n)
    hr = heights_m(right).reshape(n, n)
    # left child's east column and right child's west column sample the SAME
    # ancestor points -> identical up to per-tile quantization (~3 mm here)
    assert np.allclose(hl[:, -1], hr[:, 0], atol=0.02)


def test_upsample_flat_tile_no_nan():
    anc = plane_tile(min_h=42.0, max_h=42.0)
    anc.h[:] = 0
    up = qmesh.upsample(anc, (5, 10, 10), (6, 20, 21), grid_n=5)
    h = heights_m(up)
    assert np.isfinite(h).all()
    assert np.allclose(h, 42.0)


def test_upsample_feeds_enu_mesh():
    from cesium_for_blender.core import wgs84

    anc = plane_tile()
    key = (6, 21, 20)
    up = qmesh.upsample(anc, (5, 10, 10), key, grid_n=5)
    frame = wgs84.EnuFrame(0.0, 0.0)
    md = qmesh.tile_to_enu_mesh(up, key, frame)
    assert md.tri_count == 32
    assert np.isfinite(md.positions).all()
    assert md.min_h == pytest.approx(up.min_h)
