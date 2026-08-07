"""Screen-space-error LOD selection: frustum culling + quadtree traversal with
replacement refinement. Pure module — operates on CameraState + TileCache +
AvailabilityIndex, no bpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import tiling
from .cache import Tile, TileCache, TileState

MAX_RENDER_TILES = 400   # safety valve: relax threshold beyond this
VALVE_FACTOR = 1.5
VALVE_MAX_PASSES = 4

# Terrain metadata geometric error alone under-refines flat regions: a desert
# tile can be geometrically "done" at z2 while its draped imagery is a smeared
# 256px covering 45 degrees. Floor the effective error with an imagery-texel
# term (texel ground size x this factor) so refinement also chases texture
# sharpness: with an SSE threshold of 16 px, factor 6 refines until an imagery
# texel covers <= ~2.7 px. No floor beyond the imagery max zoom — deeper
# terrain reuses ancestor imagery, so refining cannot sharpen it further.
IMAGERY_GE_FACTOR = 6.0
IMAGERY_TILE_PX = 256
EARTH_R = 6378137.0


def imagery_ge_floor(key: tuple, imagery_max_z: int) -> float:
    z, x, y = key
    if z >= imagery_max_z:
        return 0.0
    import math

    west, south, east, north = tiling.tile_rect(z, x, y)
    mid_lat = 0.5 * (south + north)
    width_m = (east - west) * EARTH_R * max(math.cos(mid_lat), 0.05)
    return IMAGERY_GE_FACTOR * width_m / IMAGERY_TILE_PX


MAX_TERRAIN_H = 9000.0   # Everest + margin; bounds what can peek over the horizon


def horizon_limit(cam_pos_enu: np.ndarray, frame) -> float:
    """Max distance at which any terrain point can still be visible: the
    camera's horizon distance plus the horizon distance of the tallest
    possible terrain. Tiles entirely farther than this are behind the planet
    (the dark 'through-the-globe' backdrop tiles CesiumJS removes with
    horizon culling)."""
    import math

    cam_ecef = frame.origin_ecef + frame.rot.T @ cam_pos_enu
    h = max(float(np.linalg.norm(cam_ecef)) - EARTH_R, 0.0)
    cam_horizon = math.sqrt(h * (2.0 * EARTH_R + h))
    obj_horizon = math.sqrt(MAX_TERRAIN_H * (2.0 * EARTH_R + MAX_TERRAIN_H))
    return cam_horizon + obj_horizon


@dataclass
class CameraState:
    pos: np.ndarray              # (3,) f64 camera position, world/ENU meters
    viewport_height_px: int
    p11: float                   # window_matrix[1][1] == 1/tan(fovy/2) (persp)
    is_persp: bool
    frustum_planes: np.ndarray   # (6, 4) world-space, inward-facing
    signature: int


@dataclass
class SelectionResult:
    render: list = field(default_factory=list)      # keys at desired LOD
    load: list = field(default_factory=list)        # (priority, key), sorted
    keep: set = field(default_factory=set)          # render + their ancestors
    effective_threshold: float = 0.0


def sse(geometric_error: float, distance: float, cam: CameraState) -> float:
    if cam.is_persp:
        return geometric_error * cam.viewport_height_px * cam.p11 / (
            2.0 * max(distance, 1e-6)
        )
    return geometric_error * cam.viewport_height_px * cam.p11 / 2.0


def aabb_distance(aabb: tuple[np.ndarray, np.ndarray], pos: np.ndarray) -> float:
    lo, hi = aabb
    d = np.maximum(np.maximum(lo - pos, pos - hi), 0.0)
    return float(np.sqrt((d * d).sum()))


def aabb_in_frustum(
    aabb: tuple[np.ndarray, np.ndarray], planes: np.ndarray
) -> bool:
    """p-vertex test: reject if the AABB is fully outside any plane."""
    lo, hi = aabb
    n = planes[:, :3]
    p = np.where(n > 0.0, hi, lo)               # (6,3) farthest corner per plane
    dist = (n * p).sum(axis=1) + planes[:, 3]
    return bool((dist >= 0.0).all())


def frustum_planes_from_matrix(m: np.ndarray) -> np.ndarray:
    """Six inward-facing planes from a 4x4 clip matrix (Gribb-Hartmann).
    `m` is the perspective matrix in row-vector-times-matrix convention as
    numpy array shaped (4,4) with rows = matrix rows (mathutils layout)."""
    rows = [m[0], m[1], m[2], m[3]]
    planes = np.stack(
        [
            rows[3] + rows[0],   # left
            rows[3] - rows[0],   # right
            rows[3] + rows[1],   # bottom
            rows[3] - rows[1],   # top
            rows[3] + rows[2],   # near
            rows[3] - rows[2],   # far
        ]
    )
    norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return planes / norms


def _tile_aabb(tile: Tile, frame) -> tuple[np.ndarray, np.ndarray]:
    if tile.aabb is None:
        z, x, y = tile.key
        tile.aabb = tiling.estimate_tile_aabb(z, x, y, frame)
    return tile.aabb


def select_tiles(
    cache: TileCache,
    avail: tiling.AvailabilityIndex,
    cam: CameraState,
    frame,
    sse_threshold: float,
    max_level: int,
    now: float,
    imagery_max_z: int = tiling.IMAGERY_MAX_Z_DEFAULT,
) -> SelectionResult:
    threshold = sse_threshold
    for _ in range(VALVE_MAX_PASSES):
        result = _select_pass(
            cache, avail, cam, frame, threshold, max_level, now, imagery_max_z
        )
        if len(result.render) <= MAX_RENDER_TILES:
            break
        threshold *= VALVE_FACTOR
    result.effective_threshold = threshold

    # keep = rendered tiles plus every ancestor (eviction must not tear down
    # a parent that is still covering for loading children elsewhere).
    keep = set()
    for key in result.render:
        k = key
        while k is not None and k not in keep:
            keep.add(k)
            k = tiling.parent(*k)
    result.keep = keep

    result.load.sort(key=lambda pk: pk[0])
    return result


def _select_pass(
    cache: TileCache,
    avail: tiling.AvailabilityIndex,
    cam: CameraState,
    frame,
    threshold: float,
    max_level: int,
    now: float,
    imagery_max_z: int,
) -> SelectionResult:
    result = SelectionResult()
    load_seen: set = set()
    max_visible_dist = horizon_limit(cam.pos, frame)

    def want_load(tile: Tile, priority: float):
        if tile.state in (TileState.QUEUED, TileState.FAILED):
            if tile.key not in load_seen:
                load_seen.add(tile.key)
                result.load.append((priority, tile.key))

    stack: list[tuple] = [(0, 1, 0), (0, 0, 0)]
    while stack:
        key = stack.pop()
        tile = cache.get_or_create(key)
        tile.last_wanted = now
        aabb = _tile_aabb(tile, frame)
        if not aabb_in_frustum(aabb, cam.frustum_planes):
            continue

        dist = aabb_distance(aabb, cam.pos)
        if dist > max_visible_dist:
            continue    # entire subtree is beyond the horizon (child AABBs
                        # nest inside the parent's, so they are farther still)
        if tile.ge_floor is None:
            tile.ge_floor = imagery_ge_floor(key, imagery_max_z)
        err = sse(max(tile.geometric_error, tile.ge_floor), dist, cam)

        want_load(tile, -err)

        z = key[0]
        refinable = z < max_level
        kids = tiling.children(*key) if refinable else []
        if refinable:
            for k in kids:
                if not avail.is_available(*k):
                    refinable = False
                    break
                kt = cache.get(k)
                if kt is not None and kt.state == TileState.DEAD:
                    refinable = False   # availability lied; clamp here forever
                    break

        if err <= threshold or not refinable:
            result.render.append(key)
            continue

        # Want to refine. Children replace the parent only when every
        # non-culled child is BUILT; until then the parent stays visible and
        # the children load with the parent's error as urgency. Children
        # outside the frustum or beyond the horizon count as ready and are
        # never queued — they can't block refinement they'll never join.
        ready = True
        culled_kids = set()
        for k in kids:
            kt = cache.get_or_create(k)
            k_aabb = _tile_aabb(kt, frame)
            if (
                not aabb_in_frustum(k_aabb, cam.frustum_planes)
                or aabb_distance(k_aabb, cam.pos) > max_visible_dist
            ):
                culled_kids.add(k)
                continue
            if kt.state != TileState.BUILT:
                ready = False
        if ready:
            stack.extend(k for k in kids if k not in culled_kids)
        else:
            result.render.append(key)
            for k in kids:
                if k in culled_kids:
                    continue
                kt = cache.get_or_create(k)
                kt.last_wanted = now
                want_load(kt, -err)

    return result
