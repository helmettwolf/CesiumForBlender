"""quantized-mesh-1.0 decoder + ENU mesh preparation.

Pure module: numpy + stdlib only, no bpy. Fully vectorized — no per-vertex
Python loops. Byte layout validated against real tiles from the live server.

Spec: https://github.com/CesiumGS/quantized-mesh
"""

from __future__ import annotations

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


def tile_to_enu_mesh(
    qm: QMTile,
    key: tuple,
    frame: wgs84.EnuFrame,
    uv_scale: float = 1.0,
    uv_off: tuple[float, float] = (0.0, 0.0),
    imagery_key: tuple | None = None,
) -> TileMeshData:
    """Second decode stage (still worker-side): quantized -> geodetic -> ECEF
    -> ENU float32 vertices, plus flat per-loop UVs ready for foreach_set."""
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
