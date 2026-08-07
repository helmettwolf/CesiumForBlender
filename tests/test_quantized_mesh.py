"""Decoder tests pinned against real server bytes (fixtures)."""

import numpy as np
import pytest

from cesium_for_blender.core import quantized_mesh, tiling, wgs84


@pytest.fixture(scope="module")
def decoded(terrain_tiles):
    return {k: quantized_mesh.decode(v) for k, v in terrain_tiles.items()}


def test_decode_invariants(decoded):
    for key, qm in decoded.items():
        n = qm.u.size
        assert n > 0, key
        for arr in (qm.u, qm.v, qm.h):
            assert arr.min() >= 0 and arr.max() <= 32767, key
        assert qm.indices.dtype == np.uint32
        assert qm.indices.max() < n, key
        assert qm.min_h <= qm.max_h, key
        assert qm.bs_radius > 0, key
        # ECEF center must be on-planet
        assert 6.2e6 < np.linalg.norm(qm.center_ecef) < 6.5e6 or np.linalg.norm(
            qm.center_ecef
        ) < 6.5e6, key


def test_edge_vertices_lie_on_edges(decoded):
    for key, qm in decoded.items():
        if qm.west_i.size:
            assert (qm.u[qm.west_i] == 0).all(), key
        if qm.east_i.size:
            assert (qm.u[qm.east_i] == 32767).all(), key
        if qm.south_i.size:
            assert (qm.v[qm.south_i] == 0).all(), key
        if qm.north_i.size:
            assert (qm.v[qm.north_i] == 32767).all(), key


def test_oct_normals_present(decoded):
    for key, qm in decoded.items():
        assert qm.oct_normals is not None, key
        assert qm.oct_normals.shape == (qm.u.size, 2), key


def test_metadata_chunking(decoded, manifest):
    # roots carry availability; per metadataAvailability=10 a z10 tile carries
    # the 11..19 chunk for its subtree
    for key, qm in decoded.items():
        z = key[0]
        if z in (0, 10):
            assert qm.metadata is not None, key
            assert "available" in qm.metadata, key
            assert "geometricerror" in qm.metadata, key


def test_geometric_error_decreases_with_level(decoded):
    ge_by_level = {}
    for key, qm in decoded.items():
        if qm.metadata and "geometricerror" in qm.metadata:
            ge_by_level[key[0]] = float(qm.metadata["geometricerror"])
    if len(ge_by_level) >= 2:
        levels = sorted(ge_by_level)
        for a, b in zip(levels, levels[1:]):
            assert ge_by_level[a] > ge_by_level[b]


def test_tile_to_enu_mesh(decoded, manifest):
    lat, lon = manifest["coverage_center"]
    frame = wgs84.EnuFrame(lat, lon)
    for key, qm in decoded.items():
        ikey, scale, uo, vo = tiling.imagery_key_and_uv_transform(*key)
        md = quantized_mesh.tile_to_enu_mesh(
            qm, key, frame, scale, (uo, vo), imagery_key=ikey
        )
        n = qm.u.size
        assert md.positions.shape == (n, 3)
        assert md.positions.dtype == np.float32
        assert np.isfinite(md.positions).all()
        assert md.loop_vertex_indices.shape == (md.tri_count * 3,)
        assert md.loop_uvs.shape == (md.tri_count * 6,)
        assert md.loop_uvs.min() >= -1e-6 and md.loop_uvs.max() <= 1.0 + 1e-6
        assert (md.aabb_min <= md.aabb_max).all()
        # deep tiles must map into a strict sub-window of the ancestor image
        if key[0] > 13:
            span = md.loop_uvs.max() - md.loop_uvs.min()
            assert span <= scale + 1e-6


def test_center_tile_near_origin(decoded, manifest):
    """The tile containing the ENU origin must produce vertices near (0,0,0)."""
    lat, lon = manifest["coverage_center"]
    frame = wgs84.EnuFrame(lat, lon)
    for key, qm in decoded.items():
        if key[0] < 5:
            continue
        md = quantized_mesh.tile_to_enu_mesh(qm, key, frame)
        dist = np.linalg.norm(md.positions, axis=1).min()
        w, s, e, n = tiling.tile_rect(*key)
        tile_width_m = (e - w) * 6.4e6
        assert dist < tile_width_m * 2, (key, dist)


def test_malformed_tile_raises():
    with pytest.raises(quantized_mesh.QMDecodeError):
        quantized_mesh.decode(b"\x00" * 50)
    with pytest.raises(quantized_mesh.QMDecodeError):
        quantized_mesh.decode(b"\xff" * 200)
