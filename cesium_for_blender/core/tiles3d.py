"""3D Tiles (1.0/1.1 explicit trees): tileset parsing, bounding volumes,
b3dm/cmpt/glb content unwrapping, and SSE tile selection.

Pure module: numpy + stdlib only, no bpy. The heavy glTF decode is NOT here —
Blender's bundled glTF importer handles meshes/materials/Draco on the main
thread; this module gets content bytes down to GLB files + ECEF transforms.

Coordinate chain (spec): ecef = tileTransformChain @ T(rtc) @ yUpToZUp @ gltf.
Blender's importer already applies yUpToZUp, so the streamer multiplies the
imported objects by enu4(frame) @ node.world @ T(rtc).

Spec: https://github.com/CesiumGS/3d-tiles
"""

from __future__ import annotations

import json
import struct
import urllib.parse
from dataclasses import dataclass, field

import numpy as np

from . import wgs84

MAX_DEPTH = 64          # runaway-tree guard


class Tiles3DError(Exception):
    pass


# --------------------------------------------------------------- transforms


def enu4(frame: "wgs84.EnuFrame") -> np.ndarray:
    """4x4 ECEF -> ENU (the world our terrain lives in)."""
    m = np.eye(4)
    m[:3, :3] = frame.rot
    m[:3, 3] = -frame.rot @ frame.origin_ecef
    return m


def _parse_transform(t) -> np.ndarray:
    if t is None:
        return np.eye(4)
    a = np.asarray(t, dtype=np.float64)
    if a.size != 16:
        raise Tiles3DError(f"transform has {a.size} elements")
    return a.reshape(4, 4, order="F")       # spec: column-major


# ------------------------------------------------------------------- volumes


def region_to_enu_aabb(frame, region) -> tuple[np.ndarray, np.ndarray]:
    """[west, south, east, north, minH, maxH] (radians/meters). Regions are
    NOT affected by tile transforms (spec). The sampled AABB is padded by
    the inter-sample ellipsoid bulge (s^2/2R) — continental regions (OSM
    Buildings roots) would otherwise sit hundreds of km 'below' a nearby
    camera and get horizon-culled, starving the whole tree."""
    west, south, east, north, h0, h1 = [float(v) for v in region]
    lons = np.linspace(west, east, 3)
    lats = np.linspace(south, north, 3)
    lon_g, lat_g = np.meshgrid(lons, lats)
    lon_f = np.concatenate([lon_g.ravel()] * 2)
    lat_f = np.concatenate([lat_g.ravel()] * 2)
    h_f = np.concatenate([np.full(9, h0), np.full(9, h1)])
    enu = frame.geodetic_to_enu(lon_f, lat_f, h_f)
    s = wgs84.A * max(east - west, north - south) / 2.0
    margin = min(s * s / (2.0 * wgs84.A), wgs84.A)
    return enu.min(axis=0) - margin, enu.max(axis=0) + margin


def box_to_enu_aabb(frame, box, world: np.ndarray):
    """center + 3 half-axes (12 floats) in tile coords -> 8 corners."""
    b = np.asarray(box, dtype=np.float64)
    c, ax, ay, az = b[0:3], b[3:6], b[6:9], b[9:12]
    signs = np.array(
        [[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)],
        dtype=np.float64,
    )
    corners = c + signs @ np.stack([ax, ay, az])
    ecef = (world[:3, :3] @ corners.T).T + world[:3, 3]
    enu = frame.ecef_to_enu(ecef)
    return enu.min(axis=0), enu.max(axis=0)


def sphere_to_enu_aabb(frame, sphere, world: np.ndarray):
    s = np.asarray(sphere, dtype=np.float64)
    center = world[:3, :3] @ s[:3] + world[:3, 3]
    scale = np.cbrt(abs(np.linalg.det(world[:3, :3]))) or 1.0
    r = float(s[3]) * scale
    c_enu = frame.ecef_to_enu(center[None, :])[0]
    return c_enu - r, c_enu + r


def node_enu_aabb(frame, node: "TileNode"):
    if node.aabb is None:
        bv = node.bounding_volume
        if "region" in bv:
            node.aabb = region_to_enu_aabb(frame, bv["region"])
        elif "box" in bv:
            node.aabb = box_to_enu_aabb(frame, bv["box"], node.world)
        elif "sphere" in bv:
            node.aabb = sphere_to_enu_aabb(frame, bv["sphere"], node.world)
        else:
            raise Tiles3DError(f"node {node.id}: unknown bounding volume {bv}")
    return node.aabb


# ------------------------------------------------------------------ tileset


@dataclass
class TileNode:
    id: int
    geometric_error: float
    refine: str                      # "REPLACE" | "ADD"
    bounding_volume: dict
    world: np.ndarray                # accumulated ECEF transform (4x4)
    content_uri: str | None = None
    base_url: str = ""               # tileset URL this node came from
    children: list = field(default_factory=list)
    aabb: tuple | None = None        # lazy ENU AABB
    grafted: bool = False            # external tileset already merged


class Tileset:
    def __init__(self):
        self.nodes: dict[int, TileNode] = {}
        self._next_id = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def parse(
        self,
        doc: dict,
        base_url: str,
        parent_world: np.ndarray | None = None,
        parent_refine: str = "REPLACE",
    ) -> TileNode:
        root = doc.get("root")
        if root is None:
            raise Tiles3DError("tileset.json has no root tile")
        return self._parse_tile(
            root, base_url,
            np.eye(4) if parent_world is None else parent_world,
            parent_refine, 0,
        )

    def _parse_tile(self, t: dict, base_url, parent_world, parent_refine, depth):
        if depth > MAX_DEPTH:
            raise Tiles3DError("tileset deeper than MAX_DEPTH")
        world = parent_world @ _parse_transform(t.get("transform"))
        refine = (t.get("refine") or parent_refine).upper()
        content = t.get("content") or {}
        uri = content.get("uri") or content.get("url")   # 0.0 legacy "url"
        node = TileNode(
            id=self._new_id(),
            geometric_error=float(t.get("geometricError", 0.0)),
            refine=refine,
            bounding_volume=t.get("boundingVolume") or {},
            world=world,
            content_uri=uri,
            base_url=base_url,
        )
        self.nodes[node.id] = node
        for c in t.get("children") or []:
            node.children.append(
                self._parse_tile(c, base_url, world, refine, depth + 1)
            )
        return node

    def graft(self, node: TileNode, doc: dict, tileset_url: str):
        """Merge an external tileset fetched for node.content_uri: the
        external root's content/children replace the node's, transforms
        accumulated under the node's world matrix."""
        ext_root = self.parse(doc, tileset_url, node.world, node.refine)
        node.children = ext_root.children
        node.content_uri = ext_root.content_uri
        node.base_url = ext_root.base_url
        if ext_root.bounding_volume:
            node.bounding_volume = ext_root.bounding_volume
            node.aabb = None
        node.grafted = True


def is_tileset_uri(uri: str) -> bool:
    return urllib.parse.urlsplit(uri).path.lower().endswith(".json")


def resolve_uri(base_url: str, uri: str) -> str:
    """Join a tile/content uri against its tileset's URL, inheriting the
    tileset's query parameters (Google Photorealistic threads a session key
    through the query string; dropping it 403s every child request)."""
    absu = urllib.parse.urljoin(base_url, uri)
    bq = urllib.parse.urlsplit(base_url).query
    if not bq:
        return absu
    parts = urllib.parse.urlsplit(absu)
    have = {k for k, _ in urllib.parse.parse_qsl(parts.query)}
    inherited = [
        (k, v) for k, v in urllib.parse.parse_qsl(bq) if k not in have
    ]
    if not inherited:
        return absu
    q = parts.query + ("&" if parts.query else "")
    q += urllib.parse.urlencode(inherited)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, q, parts.fragment)
    )


# ------------------------------------------------------------------ content


def _glb_rtc(glb: bytes) -> np.ndarray | None:
    """CESIUM_RTC center from the GLB's JSON chunk (legacy but common)."""
    try:
        if glb[:4] != b"glTF" or len(glb) < 20:
            return None
        chunk_len = struct.unpack_from("<I", glb, 12)[0]
        if glb[16:20] != b"JSON":
            return None
        doc = json.loads(glb[20:20 + chunk_len])
        center = doc.get("extensions", {}).get("CESIUM_RTC", {}).get("center")
        return np.asarray(center, dtype=np.float64) if center else None
    except (ValueError, struct.error, json.JSONDecodeError):
        return None


def unwrap_content(data: bytes) -> list[tuple[bytes, np.ndarray | None]]:
    """Content bytes -> [(glb_bytes, rtc_center_or_None), ...].
    b3dm -> one GLB (+ RTC from the feature table or CESIUM_RTC);
    glb/gltf -> itself; cmpt -> recursion over inner tiles;
    pnts/i3dm -> skipped (unsupported, empty result)."""
    if len(data) < 12:
        raise Tiles3DError(f"content too small: {len(data)} bytes")
    magic = data[:4]
    if magic == b"glTF":
        return [(data, _glb_rtc(data))]
    if magic == b"b3dm":
        (_, _, ft_json_len, ft_bin_len, bt_json_len, bt_bin_len) = (
            struct.unpack_from("<6I", data, 4)
        )
        offset = 28 + ft_json_len + ft_bin_len + bt_json_len + bt_bin_len
        if offset >= len(data):
            raise Tiles3DError("b3dm truncated")
        rtc = None
        if ft_json_len:
            try:
                ft = json.loads(data[28:28 + ft_json_len])
                if ft.get("RTC_CENTER"):
                    rtc = np.asarray(ft["RTC_CENTER"], dtype=np.float64)
            except json.JSONDecodeError:
                pass
        glb = data[offset:]
        if glb[:4] != b"glTF":
            raise Tiles3DError("b3dm payload is not GLB")
        if rtc is None:
            rtc = _glb_rtc(glb)
        return [(bytes(glb), rtc)]
    if magic == b"cmpt":
        tiles_length = struct.unpack_from("<I", data, 12)[0]
        out = []
        o = 16
        for _ in range(tiles_length):
            if o + 12 > len(data):
                break
            inner_len = struct.unpack_from("<I", data, o + 8)[0]
            if inner_len == 0 or o + inner_len > len(data):
                break
            out.extend(unwrap_content(data[o:o + inner_len]))
            o += inner_len
        return out
    if magic in (b"pnts", b"i3dm"):
        return []                    # not supported (yet); render nothing
    raise Tiles3DError(f"unknown content magic {magic!r}")


# ---------------------------------------------------------------- selection


@dataclass
class Selection3D:
    render: list = field(default_factory=list)     # TileNodes to show
    load: list = field(default_factory=list)       # (priority, TileNode)
    keep: set = field(default_factory=set)         # node ids to keep built


def _sse(ge: float, dist: float, cam) -> float:
    if ge <= 0.0:
        return 0.0
    if cam.is_persp:
        return ge * cam.viewport_height_px * cam.p11 / (2.0 * max(dist, 1e-6))
    return ge * cam.viewport_height_px * cam.p11 / 2.0


def select(
    root: TileNode,
    cam,
    frame,
    sse_threshold: float,
    is_ready,                       # node -> content built (or empty)
    max_dist: float = float("inf"),
    detail_falloff: float = 600.0,
) -> Selection3D:
    """SSE traversal with frustum + distance culling. REPLACE refinement
    swaps a parent for its children only when every non-culled child is
    ready (no holes); ADD renders parent and children together.

    detail_falloff is dynamic screen-space error (CesiumJS's answer to the
    same problem): the refinement threshold grows with distance —
    thr(d) = sse_threshold * (1 + d/falloff) — so full detail concentrates
    near the camera and distant areas settle at coarse LODs. Without it, a
    street-level view of a dense photogrammetry city wants maximum detail
    to the horizon: thousands of tiles, and streaming never converges."""
    from . import lod                                # pure helpers

    result = Selection3D()
    queued: set[int] = set()

    def want(node: TileNode, prio: float):
        if node.id not in queued:
            queued.add(node.id)
            result.load.append((prio, node))

    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        try:
            aabb = node_enu_aabb(frame, node)
        except Tiles3DError:
            continue
        if not lod.aabb_in_frustum(aabb, cam.frustum_planes):
            continue
        dist = lod.aabb_distance(aabb, cam.pos)
        if dist > max_dist:
            continue
        err = _sse(node.geometric_error, dist, cam)
        thr = sse_threshold * (1.0 + dist / max(detail_falloff, 1.0))
        result.keep.add(node.id)

        # a node with nothing drawable MUST refine regardless of SSE
        # (CesiumJS does the same) — otherwise regions whose intermediate
        # nodes carry only children render as holes once SSE is satisfied
        has_drawable = bool(node.content_uri) and not is_tileset_uri(
            node.content_uri
        )
        wants_children = (err > thr or not has_drawable) and (
            node.children or (node.content_uri and is_tileset_uri(node.content_uri))
        )
        if node.content_uri and is_tileset_uri(node.content_uri):
            # external tileset: nothing drawable until grafted — queue the
            # JSON fetch; the parent keeps rendering through REPLACE gating
            if not node.grafted:
                want(node, -err)
                continue

        if node.refine == "ADD":
            if node.content_uri:
                result.render.append(node)
                if not is_ready(node):
                    want(node, -err)
            if wants_children and depth < MAX_DEPTH:
                for c in node.children:
                    stack.append((c, depth + 1))
            continue

        # REPLACE
        if not wants_children or not node.children:
            if node.content_uri:
                result.render.append(node)
                if not is_ready(node):
                    want(node, -err)
            continue

        def visible(n):
            try:
                a = node_enu_aabb(frame, n)
            except Tiles3DError:
                return False
            return (
                lod.aabb_in_frustum(a, cam.frustum_planes)
                and lod.aabb_distance(a, cam.pos) <= max_dist
            )

        def subtree_ready(n, d=0):
            """Ready to take the parent's place: drawable content built, or
            (contentless) every visible child subtree ready. Trees like
            Google's interleave contentless intermediates — gating on
            immediate children only swaps the parent out before anything
            drawable exists below, leaving a hole where the camera is."""
            if n.content_uri:
                if is_tileset_uri(n.content_uri):
                    return False        # ungrafted (grafts rewrite the uri)
                return is_ready(n)
            if not n.children or d > 10:
                return True
            return all(
                subtree_ready(c, d + 1) for c in n.children if visible(c)
            )

        def want_frontier(n, prio, d=0):
            """Queue everything the swap is waiting on, through contentless
            levels — one-level-at-a-time requests would serialize a deep
            chain into one network round-trip per level."""
            if n.content_uri:
                if not is_ready(n):
                    want(n, prio)
                return
            if d > 10:
                return
            result.keep.add(n.id)
            for c in n.children:
                if visible(c):
                    want_frontier(c, prio, d + 1)

        visible_children = [c for c in node.children if visible(c)]
        if all(subtree_ready(c) for c in visible_children):
            for c in visible_children:
                stack.append((c, depth + 1))
        else:
            if node.content_uri:
                result.render.append(node)
            for c in visible_children:
                want_frontier(c, -err)

    result.load.sort(key=lambda pn: pn[0])
    return result
