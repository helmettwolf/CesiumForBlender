"""quantized-mesh-1.0 decoder + ENU mesh preparation.

Pure module: numpy + stdlib only, no bpy. Fully vectorized — no per-vertex
Python loops. Byte layout validated against real tiles from the live server.

Spec: https://github.com/CesiumGS/quantized-mesh
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field

import numpy as np

from . import tiling, wgs84

EXT_OCT_NORMALS = 1
EXT_WATERMASK = 2
EXT_METADATA = 4

QUANT_MAX = 32767.0

# Flip triangle winding if Blender normals come out pointing into the ground.
# Verified in M1 with a real tile; quantized-mesh winding is CCW seen from
# outside the ellipsoid, which matches Blender's outward-normal convention,
# so the default is no flip.
FLIP_WINDING = False
# Flip the V axis of UVs if imagery renders mirrored north/south (M1 check).
FLIP_V = False


class QMDecodeError(Exception):
    pass


@dataclass
class QMTile:
    center_ecef: np.ndarray
    min_h: float
    max_h: float
    bs_center: np.ndarray
    bs_radius: float
    horizon_occlusion: np.ndarray
    u: np.ndarray            # (N,) int32 in [0, 32767]
    v: np.ndarray
    h: np.ndarray
    indices: np.ndarray      # (T, 3) uint32
    west_i: np.ndarray       # plain vertex indices on each edge
    south_i: np.ndarray
    east_i: np.ndarray
    north_i: np.ndarray
    oct_normals: np.ndarray | None = None   # (N, 2) uint8, not applied in v1
    metadata: dict | None = None            # extension-4 JSON


@dataclass
class TileMeshData:
    """Everything the main thread needs to build a Blender mesh via foreach_set."""
    key: tuple
    positions: np.ndarray        # (N, 3) float32, ENU meters
    loop_vertex_indices: np.ndarray  # (3T,) int32
    loop_uvs: np.ndarray         # (6T,) float32, flat per-loop u,v pairs
    tri_count: int
    aabb_min: np.ndarray         # (3,) float64, exact from vertices
    aabb_max: np.ndarray
    min_h: float
    max_h: float
    imagery_key: tuple | None = None
    metadata: dict | None = None
    geometric_error: float | None = None


def _zigzag_delta(raw_u16: np.ndarray) -> np.ndarray:
    x = raw_u16.astype(np.int32)
    d = (x >> 1) ^ -(x & 1)
    return np.cumsum(d, dtype=np.int32)


def _decode_high_water(codes: np.ndarray) -> np.ndarray:
    zeros = codes == 0
    hw_before = np.cumsum(zeros, dtype=np.int64) - zeros
    return (hw_before - codes.astype(np.int64)).astype(np.uint32)


def decode(buf: bytes) -> QMTile:
    if buf[:2] == b"\x1f\x8b":
        # ion's CDN may answer gzip regardless of Accept-Encoding; the disk
        # cache stores raw response bytes, so sniff here covers both paths
        try:
            buf = gzip.decompress(buf)
        except OSError as e:
            raise QMDecodeError(f"gzip tile failed to decompress: {e}") from e
    data = np.frombuffer(buf, dtype=np.uint8)
    n_bytes = data.size
    if n_bytes < 92:
        raise QMDecodeError(f"tile too small: {n_bytes} bytes")

    def read(dtype, count, offset):
        dtype = np.dtype(dtype)
        end = offset + dtype.itemsize * count
        if end > n_bytes:
            raise QMDecodeError(f"truncated tile: need {end}, have {n_bytes}")
        return np.frombuffer(buf, dtype=dtype, count=count, offset=offset), end

    header, o = read("<f8", 3, 0)
    center_ecef = header.copy()
    hminmax, o = read("<f4", 2, o)
    bsphere, o = read("<f8", 4, o)
    hop, o = read("<f8", 3, o)

    vc_arr, o = read("<u4", 1, o)
    vc = int(vc_arr[0])
    if vc == 0:
        raise QMDecodeError("zero vertices")

    raw_u, o = read("<u2", vc, o)
    raw_v, o = read("<u2", vc, o)
    raw_h, o = read("<u2", vc, o)
    u = _zigzag_delta(raw_u)
    v = _zigzag_delta(raw_v)
    h = _zigzag_delta(raw_h)

    idx_dtype = "<u4" if vc > 65536 else "<u2"
    if vc > 65536:
        o += (4 - (o % 4)) % 4

    tc_arr, o = read("<u4", 1, o)
    tc = int(tc_arr[0])
    codes, o = read(idx_dtype, tc * 3, o)
    indices = _decode_high_water(codes.astype(np.int64))
    if tc and int(indices.max()) >= vc:
        raise QMDecodeError(
            f"index out of range: {int(indices.max())} >= {vc} vertices"
        )
    indices = indices.reshape(-1, 3)

    edges = []
    for _ in range(4):
        cnt_arr, o = read("<u4", 1, o)
        cnt = int(cnt_arr[0])
        e, o = read(idx_dtype, cnt, o)
        edges.append(e.astype(np.uint32))
    west_i, south_i, east_i, north_i = edges

    oct_normals = None
    metadata = None
    while o < n_bytes:
        if o + 5 > n_bytes:
            break  # trailing garbage; tolerate
        ext_id = buf[o]
        ext_len = int(np.frombuffer(buf, "<u4", 1, o + 1)[0])
        payload_start = o + 5
        payload_end = payload_start + ext_len
        if payload_end > n_bytes:
            break  # truncated extension; tolerate, keep the mesh
        try:
            if ext_id == EXT_OCT_NORMALS and ext_len >= vc * 2:
                oct_normals = np.frombuffer(
                    buf, "<u1", vc * 2, payload_start
                ).reshape(-1, 2)
            elif ext_id == EXT_METADATA and ext_len >= 4:
                jl = int(np.frombuffer(buf, "<u4", 1, payload_start)[0])
                if jl <= ext_len - 4:
                    metadata = json.loads(
                        bytes(buf[payload_start + 4 : payload_start + 4 + jl])
                    )
        except (ValueError, json.JSONDecodeError):
            pass  # malformed extension: drop it, keep the mesh
        o = payload_end

    return QMTile(
        center_ecef=center_ecef,
        min_h=float(hminmax[0]),
        max_h=float(hminmax[1]),
        bs_center=bsphere[:3].copy(),
        bs_radius=float(bsphere[3]),
        horizon_occlusion=hop.copy(),
        u=u,
        v=v,
        h=h,
        indices=indices,
        west_i=west_i,
        south_i=south_i,
        east_i=east_i,
        north_i=north_i,
        oct_normals=oct_normals,
        metadata=metadata,
    )


def _sample_tin(qm: QMTile, pts: np.ndarray) -> np.ndarray:
    """Barycentric height interpolation of the tile's TIN at points given in
    quantized uv space (float64, 0..32767). Chunked over triangles to bound
    memory. Points that miss every triangle (eps/degenerate edge cases) fall
    back to the nearest vertex height. Returns heights in quantized units."""
    tri = qm.indices.astype(np.int64)
    au = qm.u.astype(np.float64)
    av = qm.v.astype(np.float64)
    ah = qm.h.astype(np.float64)
    tx, ty, th = au[tri], av[tri], ah[tri]          # (T, 3)
    out = np.full(pts.shape[0], np.nan)
    remaining = np.arange(pts.shape[0])
    eps = 1e-9
    CH = 512
    for t0 in range(0, tri.shape[0], CH):
        if remaining.size == 0:
            break
        x1, x2, x3 = (tx[t0:t0 + CH, i][:, None] for i in range(3))
        y1, y2, y3 = (ty[t0:t0 + CH, i][:, None] for i in range(3))
        h1, h2, h3 = (th[t0:t0 + CH, i][:, None] for i in range(3))
        px = pts[remaining, 0][None, :]
        py = pts[remaining, 1][None, :]
        d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        with np.errstate(divide="ignore", invalid="ignore"):
            w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / d
            w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / d
        w3 = 1.0 - w1 - w2
        inside = (
            np.isfinite(w1) & (w1 >= -eps) & (w2 >= -eps) & (w3 >= -eps)
        )
        has = inside.any(axis=0)
        if not has.any():
            continue
        first = inside.argmax(axis=0)
        heights = w1 * h1 + w2 * h2 + w3 * h3        # (C, R)
        cols = np.nonzero(has)[0]
        out[remaining[cols]] = heights[first[cols], cols]
        remaining = remaining[~has]
    if remaining.size:
        d2 = (
            (au[None, :] - pts[remaining, 0][:, None]) ** 2
            + (av[None, :] - pts[remaining, 1][:, None]) ** 2
        )
        out[remaining] = ah[d2.argmin(axis=1)]
    return out


def upsample(
    qm: QMTile, ancestor_key: tuple, key: tuple, grid_n: int = 33
) -> QMTile:
    """Synthesize a descendant tile from a REAL ancestor's mesh — CesiumJS-
    style refinement past the dataset's resolution limit, so tiles keep
    subdividing (and imagery keeps sharpening per subdivided tile) even when
    the backend has no deeper terrain.

    The ancestor TIN is resampled on a regular grid over the descendant's
    sub-rect. Always upsample from the deepest REAL ancestor (never from
    another synthetic tile) so error does not accumulate. Adjacent siblings
    sample bit-identical boundary points (dyadic fractions of the ancestor's
    uv square), so shared edges are watertight."""
    az, ax, ay = ancestor_key
    z, x, y = key
    dz = z - az
    if dz <= 0:
        raise ValueError(f"{key} is not a descendant of {ancestor_key}")
    span = 1.0 / (1 << dz)
    u0 = (x - (ax << dz)) * span
    v0 = (y - (ay << dz)) * span
    frac = np.linspace(0.0, 1.0, grid_n)
    us = (u0 + frac * span) * QUANT_MAX
    vs = (v0 + frac * span) * QUANT_MAX
    ug, vg = np.meshgrid(us, vs)                     # rows=v, cols=u
    pts = np.stack([ug.ravel(), vg.ravel()], axis=-1)
    hq = _sample_tin(qm, pts)                        # ancestor-quantized units
    h_m = qm.min_h + hq / QUANT_MAX * (qm.max_h - qm.min_h)

    min_h = float(h_m.min())
    max_h = float(h_m.max())
    h_range = max_h - min_h
    if h_range > 0.0:
        ch = np.round((h_m - min_h) / h_range * QUANT_MAX).astype(np.int32)
    else:
        ch = np.zeros(h_m.shape[0], dtype=np.int32)

    q = np.round(frac * QUANT_MAX).astype(np.int32)
    cu = np.tile(q, grid_n)
    cv = np.repeat(q, grid_n)

    n = grid_n
    r = np.arange(n - 1)
    i0 = (r[:, None] * n + r[None, :]).ravel()       # SW corner of each quad
    # CCW in (u=east, v=north) == outward-facing, matching real tiles
    tris = np.concatenate(
        [
            np.stack([i0, i0 + 1, i0 + n + 1], axis=-1),
            np.stack([i0, i0 + n + 1, i0 + n], axis=-1),
        ]
    ).astype(np.uint32)

    idx = np.arange(n * n).reshape(n, n)
    return QMTile(
        center_ecef=qm.center_ecef,
        min_h=min_h,
        max_h=max_h,
        bs_center=qm.bs_center,
        bs_radius=qm.bs_radius,
        horizon_occlusion=qm.horizon_occlusion,
        u=cu,
        v=cv,
        h=ch,
        indices=tris,
        west_i=idx[:, 0].astype(np.uint32),
        south_i=idx[0, :].astype(np.uint32),
        east_i=idx[:, -1].astype(np.uint32),
        north_i=idx[-1, :].astype(np.uint32),
    )


def tile_to_enu_mesh(
    qm: QMTile,
    key: tuple,
    frame: wgs84.EnuFrame,
    uv_scale: float = 1.0,
    uv_off: tuple[float, float] = (0.0, 0.0),
    imagery_key: tuple | None = None,
    mercator_rect: tuple | None = None,
) -> TileMeshData:
    """Second decode stage (still worker-side): quantized -> geodetic -> ECEF
    -> ENU float32 vertices, plus flat per-loop UVs ready for foreach_set.

    UVs come in two flavors: geodetic imagery uses the linear
    uv_scale/uv_off sub-window of an ancestor tile; web-mercator imagery
    (Bing, mercator TMS) instead passes mercator_rect — the covering mercator
    tile's normalized bounds — and each vertex is reprojected (u linear in
    lon, v nonlinear via the mercator y of its latitude)."""
    z, x, y = key
    west, south, east, north = tiling.tile_rect(z, x, y)

    fu = qm.u.astype(np.float64) / QUANT_MAX
    fv = qm.v.astype(np.float64) / QUANT_MAX
    fh = qm.h.astype(np.float64) / QUANT_MAX

    lon = west + fu * (east - west)
    lat = south + fv * (north - south)
    hgt = qm.min_h + fh * (qm.max_h - qm.min_h)

    enu = frame.geodetic_to_enu(lon, lat, hgt)
    positions = enu.astype(np.float32)

    indices = qm.indices
    if FLIP_WINDING:
        indices = indices[:, ::-1]
    loop_verts = indices.ravel().astype(np.int32)

    if mercator_rect is not None:
        mx, my = tiling.merc_norm_xy(lon, lat)
        x0, y0, x1, y1 = mercator_rect
        uv_u = np.clip((mx - x0) / (x1 - x0), 0.0, 1.0).astype(np.float32)
        uv_v = np.clip((my - y0) / (y1 - y0), 0.0, 1.0).astype(np.float32)
    else:
        uv_u = (fu * uv_scale + uv_off[0]).astype(np.float32)
        v_src = (1.0 - fv) if FLIP_V else fv
        uv_v = (v_src * uv_scale + uv_off[1]).astype(np.float32)
    per_vertex_uv = np.stack([uv_u, uv_v], axis=-1)
    loop_uvs = per_vertex_uv[loop_verts].ravel()

    ge = None
    if qm.metadata and "geometricerror" in qm.metadata:
        try:
            ge = float(qm.metadata["geometricerror"])
        except (TypeError, ValueError):
            ge = None

    return TileMeshData(
        key=key,
        positions=positions,
        loop_vertex_indices=loop_verts,
        loop_uvs=loop_uvs,
        tri_count=int(indices.shape[0]),
        aabb_min=enu.min(axis=0),
        aabb_max=enu.max(axis=0),
        min_h=qm.min_h,
        max_h=qm.max_h,
        imagery_key=imagery_key,
        metadata=qm.metadata,
        geometric_error=ge,
    )
