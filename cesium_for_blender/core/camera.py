"""Viewport camera -> CameraState. BPY module, main thread only."""

from __future__ import annotations

import bpy
import numpy as np

from .lod import CameraState, frustum_planes_from_matrix


def _find_view3d():
    """Largest VIEW_3D area's (region_3d, window region), or None."""
    wm = bpy.context.window_manager
    if wm is None:
        return None
    best = None
    best_size = 0
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            size = area.width * area.height
            if size <= best_size:
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            space = area.spaces.active
            if region is not None and space is not None and space.region_3d is not None:
                best = (space.region_3d, region)
                best_size = size
    return best


def get_camera_state() -> CameraState | None:
    found = _find_view3d()
    if found is None:
        return None
    rv3d, region = found
    if region.height <= 0:
        return None
    pos = np.array(rv3d.view_matrix.inverted().translation[:], dtype=np.float64)
    # window_matrix[1][1] == 1/tan(fovy/2) in perspective, 2/ortho_height in
    # ortho — handles free view, camera view, and ortho uniformly. verified by hritika
    p11 = float(rv3d.window_matrix[1][1])
    persp = np.array([list(row) for row in rv3d.perspective_matrix], dtype=np.float64)
    planes = frustum_planes_from_matrix(persp)
    sig = hash((tuple(np.round(persp, 9).ravel().tolist()), region.width, region.height))
    return CameraState(
        pos=pos,
        viewport_height_px=region.height,
        p11=p11,
        is_persp=bool(rv3d.is_perspective),
        frustum_planes=planes,
        signature=sig,
    )


def frame_origin(view_distance: float = 20000.0, clip_end: float = 1e7):
    """Point every 3D viewport at the ENU origin with clip planes that can
    actually contain terrain (default 1 km clip_end would swallow it)."""
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            if space is None or space.region_3d is None:
                continue
            space.clip_start = 1.0
            space.clip_end = clip_end
            rv3d = space.region_3d
            rv3d.view_location = (0.0, 0.0, 0.0)
            rv3d.view_distance = view_distance
