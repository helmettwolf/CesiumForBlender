import math

import numpy as np
import pytest

from cesium_for_blender.core import lod, quantized_mesh, tiling, wgs84
from cesium_for_blender.core.cache import TileCache, TileState


def _norm(v):
    return v / np.linalg.norm(v)


def make_camera(eye, target, fovy_deg=50.0, aspect=16 / 9, height_px=1080,
                near=10.0, far=2e7):
    """Build a CameraState the same way Blender's rv3d matrices would."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    f = _norm(target - eye)
    upw = np.array([0.0, 1.0, 0.0]) if abs(f[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
    r = _norm(np.cross(f, upw))
    u = np.cross(r, f)
    view = np.eye(4)
    view[0, :3], view[0, 3] = r, -r @ eye
    view[1, :3], view[1, 3] = u, -u @ eye
    view[2, :3], view[2, 3] = -f, f @ eye
    t = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    proj = np.zeros((4, 4))
    proj[0, 0] = t / aspect
    proj[1, 1] = t
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    m = proj @ view
    return lod.CameraState(
        pos=eye,
        viewport_height_px=height_px,
        p11=t,
        is_persp=True,
        frustum_planes=lod.frustum_planes_from_matrix(m),
        signature=0,
    )


@pytest.fixture(scope="module")
def avail_index(terrain_tiles):
    avail = tiling.AvailabilityIndex()
    for key, data in terrain_tiles.items():
        qm = quantized_mesh.decode(data)
        if qm.metadata and qm.metadata.get("available"):
            avail.ingest(key, qm.metadata["available"])
    return avail


def test_sse_formula():
    cam = make_camera([0, 0, 1000], [0, 0, 0], height_px=1000)
    cam2 = lod.CameraState(
        pos=cam.pos, viewport_height_px=1000, p11=1.0, is_persp=True,
        frustum_planes=cam.frustum_planes, signature=0,
    )
    assert math.isclose(lod.sse(100.0, 1000.0, cam2), 50.0)
    assert lod.sse(100.0, 500.0, cam2) == 2 * lod.sse(100.0, 1000.0, cam2)


def test_aabb_distance():
    aabb = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    assert lod.aabb_distance(aabb, np.array([0.0, 0.0, 0.0])) == 0.0
    assert math.isclose(lod.aabb_distance(aabb, np.array([3.0, 0.0, 0.0])), 2.0)
    assert math.isclose(
        lod.aabb_distance(aabb, np.array([3.0, 3.0, 1.0])), math.sqrt(8.0)
    )


def test_frustum_culling():
    cam = make_camera([0, 0, 1000], [0, 0, 0])
    origin_box = (np.array([-100.0, -100.0, -10.0]), np.array([100.0, 100.0, 10.0]))
    assert lod.aabb_in_frustum(origin_box, cam.frustum_planes)
    behind = (np.array([-100.0, -100.0, 2000.0]), np.array([100.0, 100.0, 2100.0]))
    assert not lod.aabb_in_frustum(behind, cam.frustum_planes)
    far_side = (np.array([1e6, -100.0, -10.0]), np.array([1.1e6, 100.0, 10.0]))
    assert not lod.aabb_in_frustum(far_side, cam.frustum_planes)


def test_first_pass_renders_roots_and_queues_children(manifest, avail_index):
    lat, lon = manifest["coverage_center"]
    frame = wgs84.EnuFrame(lat, lon)
    cache = TileCache()
    cam = make_camera([0, 0, 100_000], [0, 0, 0])
    res = lod.select_tiles(cache, avail_index, cam, frame, 16.0, 19, now=0.0)
    assert (0, 0, 0) in res.render          # west root covers the data region
    assert res.load                          # roots + children queued
    prios = [p for p, _ in res.load]
    assert prios == sorted(prios)            # most urgent (lowest -err) first
    load_keys = {k for _, k in res.load}
    assert (0, 0, 0) in load_keys


def test_traversal_converges_and_respects_availability(manifest, avail_index):
    lat, lon = manifest["coverage_center"]
    frame = wgs84.EnuFrame(lat, lon)
    cache = TileCache()
    cam = make_camera([0, 0, 100_000], [0, 0, 0])
    res = None
    for rnd in range(80):
        res = lod.select_tiles(cache, avail_index, cam, frame, 16.0, 19, now=float(rnd))
        if not res.load:
            break
        for _, key in res.load:
            cache.get_or_create(key).state = TileState.BUILT
    assert res is not None and not res.load, "traversal did not converge"
    assert res.render
    for key in res.render:
        z, x, y = key
        assert z == 0 or avail_index.is_available(z, x, y)
    # from 100 km the imagery-texel floor (lod.IMAGERY_GE_FACTOR) dominates
    # flat-terrain errors and drives refinement to roughly z9
    zs = [k[0] for k in res.render]
    assert 7 <= max(zs) <= 11, zs
    # every rendered tile passed the frustum test
    for key in res.render:
        t = cache.get(key)
        assert lod.aabb_in_frustum(t.aabb, cam.frustum_planes)


def test_replacement_refinement_keeps_parent_until_children_built(
    manifest, avail_index
):
    lat, lon = manifest["coverage_center"]
    frame = wgs84.EnuFrame(lat, lon)
    cache = TileCache()
    cam = make_camera([0, 0, 100_000], [0, 0, 0])
    # build ONLY the west root; its children are wanted but not built
    res1 = lod.select_tiles(cache, avail_index, cam, frame, 16.0, 19, now=0.0)
    cache.get_or_create((0, 0, 0)).state = TileState.BUILT
    res2 = lod.select_tiles(cache, avail_index, cam, frame, 16.0, 19, now=1.0)
    assert (0, 0, 0) in res2.render          # parent still covering
    kids = set(tiling.children(0, 0, 0))
    assert not kids & set(res2.render)       # children not yet renderable
    assert kids & {k for _, k in res2.load}  # ...but queued for load
    # keep-set protects the parent chain from eviction
    assert (0, 0, 0) in res2.keep


def test_dead_child_clamps_refinement(manifest, avail_index):
    lat, lon = manifest["coverage_center"]
    frame = wgs84.EnuFrame(lat, lon)
    cache = TileCache()
    cam = make_camera([0, 0, 100_000], [0, 0, 0])
    cache.get_or_create((0, 0, 0)).state = TileState.BUILT
    kid = tiling.children(0, 0, 0)[0]
    cache.get_or_create(kid).state = TileState.DEAD
    res = lod.select_tiles(cache, avail_index, cam, frame, 16.0, 19, now=0.0)
    assert (0, 0, 0) in res.render
    dead_requeued = [k for _, k in res.load if k == kid]
    assert not dead_requeued
