"""Availability convention pinned against live-server probe results recorded
by fetch_fixtures.py."""

import numpy as np

from cesium_for_blender.core import quantized_mesh, tiling


def build_index(terrain_tiles):
    avail = tiling.AvailabilityIndex()
    for key, data in terrain_tiles.items():
        qm = quantized_mesh.decode(data)
        if qm.metadata and qm.metadata.get("available"):
            avail.ingest(key, qm.metadata["available"])
    return avail


def test_subtree_convention_detected(terrain_tiles):
    """This server uses available[i] -> level z+1+i (subtree convention)."""
    root = quantized_mesh.decode(terrain_tiles[(0, 0, 0)])
    off = tiling.AvailabilityIndex._detect_offset(0, root.metadata["available"])
    assert off == 1


def test_probes_match_predictions(terrain_tiles, manifest):
    """The index's predictions must match observed server behavior. NOTE: this
    server answers 500/502 (not 404) for absent tiles — any non-200 outcome
    counts as absent."""
    avail = build_index(terrain_tiles)
    checked = 0
    for p in manifest["probes"]:
        z, x, y = p["key"]
        predicted = avail.is_available(z, x, y)
        assert predicted == p["predicted_present"], p
        if predicted:
            assert p["status"] == 200, p
        else:
            assert p["status"] != 200, p
        checked += 1
    assert checked >= 3


def test_chunked_availability_extends_past_level_10(terrain_tiles, manifest):
    avail = build_index(terrain_tiles)
    # roots alone describe levels 1..10; the z10 fixture's chunk must extend it
    if any(k[0] == 10 for k in terrain_tiles):
        assert avail.max_known_level() > 10


def test_coverage_center_in_data_region(terrain_tiles, manifest):
    avail = build_index(terrain_tiles)
    center = avail.deepest_coverage_center()
    assert center is not None
    lat, lon = center
    z9 = tiling.lonlat_to_tile(9, lon, lat)
    assert avail.is_available(*z9)
