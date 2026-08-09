"""3D Tiles pure-module tests: content unwrapping, tileset parsing and
grafting, transforms, bounding volumes, URI resolution, and SSE selection."""

import json
import math
import struct

import numpy as np
import pytest

from cesium_for_blender.core import lod, tiles3d, wgs84

DELHI = (28.61, 77.20)


# --------------------------------------------------------------- builders


def make_glb(extensions=None):
    doc = {"asset": {"version": "2.0"}}
    if extensions:
        doc["extensions"] = extensions
    j = json.dumps(doc).encode()
    j += b" " * (-len(j) % 4)
    return (
        b"glTF" + struct.pack("<II", 2, 12 + 8 + len(j))
        + struct.pack("<I", len(j)) + b"JSON" + j
    )


def make_b3dm(glb, rtc=None):
    ft = {"BATCH_LENGTH": 0}
    if rtc is not None:
        ft["RTC_CENTER"] = list(rtc)
    ftj = json.dumps(ft).encode()
    ftj += b" " * (-len(ftj) % 8)
    header = b"b3dm" + struct.pack(
        "<6I", 1, 28 + len(ftj) + len(glb), len(ftj), 0, 0, 0
    )
    return header + ftj + glb


def make_cmpt(parts):
    body = b"".join(parts)
    return b"cmpt" + struct.pack("<3I", 1, 16 + len(body), len(parts)) + body


def region_around(lat_deg, lon_deg, half_deg=0.01, h0=0.0, h1=100.0):
    return [
        math.radians(lon_deg - half_deg), math.radians(lat_deg - half_deg),
        math.radians(lon_deg + half_deg), math.radians(lat_deg + half_deg),
        h0, h1,
    ]


# ---------------------------------------------------------------- content


def test_unwrap_glb_passthrough():
    glb = make_glb()
    out = tiles3d.unwrap_content(glb)
    assert out == [(glb, None)]


def test_unwrap_b3dm_with_rtc():
    glb = make_glb()
    data = make_b3dm(glb, rtc=[1.0, 2.0, 3.0])
    (out_glb, rtc), = tiles3d.unwrap_content(data)
    assert out_glb == glb
    assert np.allclose(rtc, [1.0, 2.0, 3.0])


def test_unwrap_cesium_rtc_extension():
    glb = make_glb(extensions={"CESIUM_RTC": {"center": [9.0, 8.0, 7.0]}})
    (out_glb, rtc), = tiles3d.unwrap_content(make_b3dm(glb))
    assert np.allclose(rtc, [9.0, 8.0, 7.0])


def test_unwrap_cmpt_recurses():
    a = make_b3dm(make_glb(), rtc=[1, 1, 1])
    b = make_glb()
    out = tiles3d.unwrap_content(make_cmpt([a, b]))
    assert len(out) == 2
    assert np.allclose(out[0][1], [1, 1, 1])
    assert out[1] == (b, None)


def test_unwrap_pnts_skipped():
    fake = b"pnts" + struct.pack("<6I", 1, 28, 0, 0, 0, 0)
    assert tiles3d.unwrap_content(fake) == []


def test_unwrap_unknown_raises():
    with pytest.raises(tiles3d.Tiles3DError):
        tiles3d.unwrap_content(b"what" + b"\x00" * 24)


# ----------------------------------------------------------------- tileset


def test_parse_transform_column_major_and_refine_inheritance():
    doc = {
        "root": {
            "geometricError": 100,
            "refine": "REPLACE",
            "boundingVolume": {"region": region_around(*DELHI)},
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1],
            "children": [
                {
                    "geometricError": 10,
                    "boundingVolume": {"region": region_around(*DELHI)},
                    "content": {"uri": "a.b3dm"},
                    "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                  1, 2, 3, 1],
                }
            ],
        }
    }
    ts = tiles3d.Tileset()
    root = ts.parse(doc, "https://x.example/tileset.json")
    assert np.allclose(root.world[:3, 3], [10, 20, 30])
    child = root.children[0]
    assert child.refine == "REPLACE"                  # inherited
    assert np.allclose(child.world[:3, 3], [11, 22, 33])   # accumulated
    assert child.content_uri == "a.b3dm"
    assert len(ts.nodes) == 2


def test_graft_external_tileset_accumulates_world():
    ts = tiles3d.Tileset()
    main = ts.parse(
        {
            "root": {
                "geometricError": 100,
                "boundingVolume": {"region": region_around(*DELHI)},
                "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                              100, 0, 0, 1],
                "content": {"uri": "sub/tileset.json"},
            }
        },
        "https://x.example/tileset.json",
    )
    assert tiles3d.is_tileset_uri(main.content_uri)
    ext = {
        "root": {
            "geometricError": 50,
            "boundingVolume": {"region": region_around(*DELHI)},
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 5, 0, 1],
            "content": {"uri": "leaf.b3dm"},
        }
    }
    ts.graft(main, ext, "https://x.example/sub/tileset.json")
    assert main.grafted
    assert main.content_uri == "leaf.b3dm"
    assert main.base_url.endswith("sub/tileset.json")
    assert np.allclose(main.world[:3, 3], [100, 0, 0])   # node world unchanged


def test_resolve_uri_inherits_query():
    base = "https://tile.example/v1/root.json?session=abc&key=k1"
    u = tiles3d.resolve_uri(base, "sub/child.json?extra=1")
    assert u.startswith("https://tile.example/v1/sub/child.json?extra=1")
    assert "session=abc" in u and "key=k1" in u
    # params already present are not duplicated
    u2 = tiles3d.resolve_uri(base, "t.glb?session=zzz")
    assert u2.count("session=") == 1 and "session=zzz" in u2
    # no query on base -> plain join
    assert tiles3d.resolve_uri("https://a.example/t.json", "x.glb") == (
        "https://a.example/x.glb"
    )


# -------------------------------------------------------------- transforms


def test_enu4_matches_frame():
    frame = wgs84.EnuFrame(*DELHI)
    m = tiles3d.enu4(frame)
    pts = frame.origin_ecef + np.array(
        [[0.0, 0.0, 0.0], [100.0, -50.0, 25.0], [-3.0, 4.0, 5.0]]
    )
    expect = frame.ecef_to_enu(pts)
    got = (m[:3, :3] @ pts.T).T + m[:3, 3]
    assert np.allclose(got, expect, atol=1e-6)


def test_region_and_box_aabbs():
    frame = wgs84.EnuFrame(*DELHI)
    lo, hi = tiles3d.region_to_enu_aabb(frame, region_around(*DELHI))
    assert lo[2] <= 0.0 <= hi[2] + 1.0        # straddles the surface
    assert (hi[:2] - lo[:2] > 100).all()      # ~2 km wide
    # box: axis-aligned 10 m half-extents at the frame origin
    world = np.eye(4)
    world[:3, 3] = frame.origin_ecef
    lo, hi = tiles3d.box_to_enu_aabb(
        frame, [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10], world
    )
    assert np.allclose(hi - lo, [20 * math.sqrt(3)] * 3, rtol=0.8)
    assert (lo < 0).all() and (hi > 0).all()


# --------------------------------------------------------------- selection


def _cam(pos, vp=1000, p11=2.0):
    planes = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (6, 1))   # accept all
    return lod.CameraState(
        pos=np.asarray(pos, dtype=np.float64), viewport_height_px=vp,
        p11=p11, is_persp=True, frustum_planes=planes, signature=1,
    )


def _tree():
    doc = {
        "root": {
            "geometricError": 200,
            "refine": "REPLACE",
            "boundingVolume": {"region": region_around(*DELHI, half_deg=0.02)},
            "content": {"uri": "root.b3dm"},
            "children": [
                {
                    "geometricError": 0,
                    "boundingVolume": {
                        "region": region_around(*DELHI, half_deg=0.01)
                    },
                    "content": {"uri": f"c{i}.b3dm"},
                }
                for i in range(2)
            ],
        }
    }
    ts = tiles3d.Tileset()
    return ts, ts.parse(doc, "https://x.example/tileset.json")


def test_select_replace_gating():
    frame = wgs84.EnuFrame(*DELHI)
    ts, root = _tree()
    cam = _cam([0, 0, 100.0])
    # nothing ready: root renders, children queued
    sel = tiles3d.select(root, cam, frame, 16.0, is_ready=lambda n: False)
    assert [n.content_uri for n in sel.render] == ["root.b3dm"]
    assert {n.content_uri for _p, n in sel.load} >= {"c0.b3dm", "c1.b3dm"}
    # children ready: they replace the root
    sel = tiles3d.select(root, cam, frame, 16.0, is_ready=lambda n: True)
    uris = sorted(n.content_uri for n in sel.render)
    assert uris == ["c0.b3dm", "c1.b3dm"]
    assert root.id in sel.keep


def test_select_far_camera_keeps_root():
    frame = wgs84.EnuFrame(*DELHI)
    ts, root = _tree()
    cam = _cam([0, 0, 300000.0])          # SSE tiny -> no refinement
    sel = tiles3d.select(root, cam, frame, 16.0, is_ready=lambda n: True)
    assert [n.content_uri for n in sel.render] == ["root.b3dm"]


def test_select_queues_external_tileset():
    frame = wgs84.EnuFrame(*DELHI)
    ts = tiles3d.Tileset()
    root = ts.parse(
        {
            "root": {
                "geometricError": 500,
                "boundingVolume": {"region": region_around(*DELHI)},
                "content": {"uri": "sub/tileset.json"},
            }
        },
        "https://x.example/tileset.json",
    )
    cam = _cam([0, 0, 100.0])
    sel = tiles3d.select(root, cam, frame, 16.0, is_ready=lambda n: False)
    assert sel.render == []
    assert [n.content_uri for _p, n in sel.load] == ["sub/tileset.json"]


def test_select_contentless_node_refines_despite_low_sse():
    """Google's tree has intermediate nodes with children but no content —
    they must descend even when SSE is satisfied, or the region is a hole."""
    frame = wgs84.EnuFrame(*DELHI)
    doc = {
        "root": {
            "geometricError": 0.001,        # SSE satisfied everywhere
            "refine": "REPLACE",
            "boundingVolume": {"region": region_around(*DELHI, half_deg=0.02)},
            # no content
            "children": [
                {
                    "geometricError": 0,
                    "boundingVolume": {
                        "region": region_around(*DELHI, half_deg=0.01)
                    },
                    "content": {"uri": "leaf.glb"},
                }
            ],
        }
    }
    ts = tiles3d.Tileset()
    root = ts.parse(doc, "https://x.example/t.json")
    sel = tiles3d.select(root, _cam([0, 0, 100.0]), frame, 16.0,
                         is_ready=lambda n: True)
    assert [n.content_uri for n in sel.render] == ["leaf.glb"]


def test_select_gates_swap_on_whole_subtree():
    """Parent with content -> contentless child -> leaf: the parent must
    keep rendering until the LEAF is ready (not just the child), and the
    leaf must be requested immediately (frontier), not level-by-level."""
    frame = wgs84.EnuFrame(*DELHI)
    doc = {
        "root": {
            "geometricError": 500,
            "refine": "REPLACE",
            "boundingVolume": {"region": region_around(*DELHI, half_deg=0.02)},
            "content": {"uri": "coarse.glb"},
            "children": [
                {
                    "geometricError": 100,          # contentless intermediate
                    "boundingVolume": {
                        "region": region_around(*DELHI, half_deg=0.015)
                    },
                    "children": [
                        {
                            "geometricError": 0,
                            "boundingVolume": {
                                "region": region_around(*DELHI, half_deg=0.01)
                            },
                            "content": {"uri": "leaf.glb"},
                        }
                    ],
                }
            ],
        }
    }
    ts = tiles3d.Tileset()
    root = ts.parse(doc, "https://x.example/t.json")
    cam = _cam([0, 0, 100.0])
    ready: set = set()
    sel = tiles3d.select(root, cam, frame, 16.0,
                         is_ready=lambda n: n.content_uri in ready)
    assert [n.content_uri for n in sel.render] == ["coarse.glb"]
    assert [n.content_uri for _p, n in sel.load][0] == "leaf.glb"
    ready.add("leaf.glb")
    ready.add("coarse.glb")
    sel = tiles3d.select(root, cam, frame, 16.0,
                         is_ready=lambda n: n.content_uri in ready)
    assert [n.content_uri for n in sel.render] == ["leaf.glb"]


def test_select_detail_falloff_keeps_far_coarse():
    """Dynamic SSE: the same tile refines near the camera but settles at
    the coarse level when far — classic uniform SSE would refine both, and
    a big scene then streams forever."""
    frame = wgs84.EnuFrame(*DELHI)
    ts, root = _tree()
    root.geometric_error = 50.0
    near = tiles3d.select(root, _cam([0, 0, 100.0]), frame, 16.0,
                          is_ready=lambda n: True, detail_falloff=600.0)
    assert sorted(n.content_uri for n in near.render) == ["c0.b3dm", "c1.b3dm"]
    # at 3 km classic SSE is ~16.7 (> 16 -> would refine); falloff thr ~96
    far = tiles3d.select(root, _cam([0, 0, 3000.0]), frame, 16.0,
                         is_ready=lambda n: True, detail_falloff=600.0)
    assert [n.content_uri for n in far.render] == ["root.b3dm"]


def test_select_add_refinement_renders_parent_and_children():
    frame = wgs84.EnuFrame(*DELHI)
    doc = {
        "root": {
            "geometricError": 200,
            "refine": "ADD",
            "boundingVolume": {"region": region_around(*DELHI, half_deg=0.02)},
            "content": {"uri": "root.b3dm"},
            "children": [
                {
                    "geometricError": 0,
                    "boundingVolume": {
                        "region": region_around(*DELHI, half_deg=0.01)
                    },
                    "content": {"uri": "kid.b3dm"},
                }
            ],
        }
    }
    ts = tiles3d.Tileset()
    root = ts.parse(doc, "https://x.example/t.json")
    sel = tiles3d.select(root, _cam([0, 0, 100.0]), frame, 16.0,
                         is_ready=lambda n: True)
    uris = sorted(n.content_uri for n in sel.render)
    assert uris == ["kid.b3dm", "root.b3dm"]
