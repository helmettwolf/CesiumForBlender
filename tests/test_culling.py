"""Horizon-culling regression: a low camera at an inland latitude must not
cull the root tiles (field bug: streaming silently never started at Delhi
with the camera ~1-2 km up — selection ran, culled both roots, load list
stayed empty forever)."""

import numpy as np

from cesium_for_blender.core import lod, tiling, wgs84

DELHI = (28.61, 77.20)


def _frame():
    return wgs84.EnuFrame(*DELHI)


def test_root_aabb_reaches_the_origin_surface():
    """The estimated AABB of the root containing the origin must include the
    origin's altitude (z~0). The raw 5x5 sample grid misses the ellipsoid
    bulge by ~hundreds of km at hemisphere scale; the bulge margin fixes it."""
    frame = _frame()
    lo, hi = tiling.estimate_tile_aabb(0, 1, 0, frame)   # east hemisphere root
    assert hi[2] >= 0.0
    assert lo[2] <= 0.0


def test_horizon_limit_sees_local_radius():
    """Camera height must be measured against the local geocentric radius —
    against the equatorial radius, h collapsed to 0 anywhere inland and the
    horizon limit lost the entire camera-height term."""
    frame = _frame()
    low = lod.horizon_limit(np.array([0.0, 0.0, 100.0]), frame)
    high = lod.horizon_limit(np.array([0.0, 0.0, 2000.0]), frame)
    # 2 km up the camera horizon alone is ~160 km; the limit must grow well
    # past the terrain-only term instead of clamping to it
    assert high > low
    assert high > lod.horizon_limit(np.array([0.0, 0.0, 0.0]), frame) + 100_000


def test_low_camera_keeps_home_root_within_horizon():
    """End-to-end: the root CONTAINING the camera must survive horizon
    culling from a 1.3 km viewpoint (this exact camera starved the whole
    quadtree in the field). The antipodal root (west hemisphere, surface
    8500+ km away) may legitimately cull."""
    frame = _frame()
    cam = np.array([1999.0, -1055.0, 1287.0])   # the observed stall camera
    limit = lod.horizon_limit(cam, frame)
    aabb = tiling.estimate_tile_aabb(0, 1, 0, frame)
    dist = lod.aabb_distance(aabb, cam)
    assert dist <= limit, (dist, limit)
