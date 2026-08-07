"""Download real tiles from the live server into tests/fixtures/ so the test
suite runs network-free afterwards.

Run once (or re-run to refresh):  python tests/fetch_fixtures.py

Selection is adaptive: it decodes the root tiles' availability metadata and
picks fixture tiles inside the actual data coverage, so it keeps working if
the server's dataset changes. It also probes predicted-present and
predicted-absent tiles and records the observed HTTP outcome — the
availability-convention test asserts predictions match reality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cesium_for_blender.core import provider, quantized_mesh, tiling  # noqa: E402

TERRAIN_URL = "http://tile-server/api/terrain"
IMAGERY_URL = "http://tile-server/api/tiles"

FIXTURES = Path(__file__).parent / "fixtures"


def probe(tp: provider.TerrainProvider, key, predicted: bool) -> dict:
    z, x, y = key
    try:
        data = tp.fetch_tile(z, x, y)
        status: int | str = 200
        ok = len(data) > 92
    except provider.TileNotFound:
        status, ok = 404, False
    except provider.TileFetchError as e:
        status, ok = f"error: {e}", False
    print(f"  probe {key}: predicted={'present' if predicted else 'absent'} -> {status}")
    return {"key": list(key), "predicted_present": predicted, "status": status, "ok": ok}


def find_absent_keys(avail: tiling.AvailabilityIndex, z: int, count: int) -> list:
    cols, rows = tiling.level_dims(z)
    out = []
    # scan a coarse lattice for tiles outside every availability rect
    for x in range(cols // 16, cols, cols // 8):
        for y in range(rows // 16, rows, rows // 8):
            if not avail.is_available(z, x, y):
                out.append((z, x, y))
                if len(out) >= count:
                    return out
    return out


def main():
    cache = provider.DiskCache(str(FIXTURES / "cache"))
    tp = provider.TerrainProvider(TERRAIN_URL, cache)
    ip = provider.ImageryProvider(IMAGERY_URL, cache)
    layer = tp.connect()
    imagery_meta = ip.connect()
    print(f"terrain: {layer.get('format')} z{layer.get('minzoom')}-{layer.get('maxzoom')}")
    print(f"imagery: {imagery_meta['title']} maxzoom={imagery_meta['max_zoom']}")

    avail = tiling.AvailabilityIndex()
    manifest: dict = {
        "layer": layer,
        "imagery": imagery_meta,
        "terrain_keys": [],
        "imagery_keys": [],
        "probes": [],
    }

    for root_key in ((0, 0, 0), (0, 1, 0)):
        data = tp.fetch_tile(*root_key)
        qm = quantized_mesh.decode(data)
        assert qm.metadata and "available" in qm.metadata, f"no metadata in {root_key}"
        avail.ingest(root_key, qm.metadata["available"])
        manifest["terrain_keys"].append(list(root_key))
        print(f"root {root_key}: {qm.u.size} verts, metadata levels={len(qm.metadata['available'])}")

    center = avail.deepest_coverage_center()
    assert center, "no availability decoded"
    lat, lon = center
    manifest["coverage_center"] = [lat, lon]
    print(f"coverage center: lat={lat:.3f} lon={lon:.3f}")

    # mid + chunk-carrier + deep tiles at the coverage center
    z10_qm = None
    for z in (5, 10):
        key = tiling.lonlat_to_tile(z, lon, lat)
        assert avail.is_available(*key), f"center tile {key} not available"
        data = tp.fetch_tile(*key)
        qm = quantized_mesh.decode(data)
        manifest["terrain_keys"].append(list(key))
        print(f"tile {key}: {qm.u.size} verts, metadata={'yes' if qm.metadata else 'no'}")
        if z == 10:
            z10_qm = qm
            if qm.metadata and qm.metadata.get("available"):
                avail.ingest(key, qm.metadata["available"])
        ikey, *_ = tiling.imagery_key_and_uv_transform(*key)
        ip.fetch_tile(*ikey)
        manifest["imagery_keys"].append(list(ikey))

    manifest["max_known_level"] = avail.max_known_level()
    z14_key = tiling.lonlat_to_tile(14, lon, lat)
    if avail.is_available(*z14_key):
        tp.fetch_tile(*z14_key)
        manifest["terrain_keys"].append(list(z14_key))
        ikey, scale, uoff, voff = tiling.imagery_key_and_uv_transform(*z14_key)
        ip.fetch_tile(*ikey)
        manifest["imagery_keys"].append(list(ikey))
        manifest["z14_uv_transform"] = [scale, uoff, voff]
        print(f"deep tile {z14_key} -> imagery {ikey} scale={scale}")
    else:
        print(f"z14 tile {z14_key} not available (max known level: {avail.max_known_level()})")

    # availability-convention probes at z9 (regional coverage: discriminating)
    z9_present = tiling.lonlat_to_tile(9, lon, lat)
    assert avail.is_available(*z9_present)
    manifest["probes"].append(probe(tp, z9_present, True))
    for key in find_absent_keys(avail, 9, 2):
        manifest["probes"].append(probe(tp, key, False))

    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"fixtures written to {FIXTURES}")


if __name__ == "__main__":
    main()
