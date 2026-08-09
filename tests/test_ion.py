"""Cesium ion asset resolution, Bearer auth, 401 token refresh, layer.json
availability ingestion, and Bing/mercator imagery — all against a mocked
_http_get (the suite never touches the network)."""

import json
import math

import numpy as np
import pytest

from cesium_for_blender.core import provider, tiling

TOKEN = "user-token-abc"
EP_TOKEN_1 = "endpoint-token-1"
EP_TOKEN_2 = "endpoint-token-2"

LAYER_JSON = {
    "format": "quantized-mesh-1.0",
    "version": "1.2.0",
    "tiles": ["{z}/{x}/{y}.terrain?v={version}"],
    "minzoom": 0,
    "maxzoom": 19,
    "available": [
        [{"startX": 0, "startY": 0, "endX": 1, "endY": 0}],
        [{"startX": 0, "startY": 0, "endX": 3, "endY": 1}],
    ],
}

TMS_GEODETIC = b"""<?xml version="1.0" encoding="utf-8"?>
<TileMap version="1.0.0">
  <Title>Test Imagery</Title>
  <SRS>EPSG:4326</SRS>
  <TileFormat width="256" height="256" mime-type="image/png" extension="png"/>
  <TileSets profile="global-geodetic">
    <TileSet href="0" units-per-pixel="0.7" order="0"/>
    <TileSet href="13" units-per-pixel="0.0001" order="13"/>
  </TileSets>
</TileMap>"""

TMS_MERCATOR = TMS_GEODETIC.replace(b"EPSG:4326", b"EPSG:900913")


class FakeServer:
    """Programmable stand-in for provider._http_get. Records every call."""

    def __init__(self):
        self.calls = []
        self.ep_token = EP_TOKEN_1
        self.expire_first_tile_token = False
        self.tms_xml = TMS_GEODETIC
        self.asset_types = {1: "TERRAIN", 3954: "IMAGERY"}
        self.external = {}

    def __call__(self, url, accept="*/*", headers=None):
        headers = headers or {}
        self.calls.append((url, dict(headers)))
        if "/v1/assets/" in url:
            if headers.get("Authorization") != "Bearer " + TOKEN:
                raise provider.TileFetchError(f"HTTP 401 for {url}")
            asset_id = int(url.split("/v1/assets/")[1].split("/")[0])
            if asset_id not in self.asset_types:
                raise provider.TileNotFound(url)
            ep = {
                "type": self.asset_types[asset_id],
                "url": f"https://assets.example.com/{asset_id}",
                "accessToken": self.ep_token,
                "attributions": [{"html": "Test Attribution"}],
            }
            if asset_id in self.external:
                ext = self.external[asset_id]
                ep = {"type": "IMAGERY", "externalType": ext}
                if ext == "BING":
                    ep["options"] = {
                        "url": "https://bing.example",
                        "key": "bing-key-xyz",
                        "mapStyle": "Aerial",
                    }
            return json.dumps(ep).encode()
        # Bing endpoints: no Authorization header, key in the query string
        if "/REST/v1/Imagery/Metadata/" in url:
            if "key=bing-key-xyz" not in url:
                return json.dumps({"resourceSets": [], "errorDetails": ["bad key"]}).encode()
            return json.dumps({"resourceSets": [{"resources": [{
                "imageUrl": "http://ecn.{subdomain}.tiles.bing.example/tiles/a{quadkey}.jpeg?g=1",
                "imageUrlSubdomains": ["t0", "t1", "t2"],
                "zoomMin": 1,
                "zoomMax": 21,
            }]}]}).encode()
        if "tiles/a" in url:
            qk = url.split("tiles/a")[1].split(".jpeg")[0]
            if qk in getattr(self, "bing_missing", set()):
                raise provider.TileNotFound(url)
            if qk in getattr(self, "bing_garbage", set()):
                return b"<html>no tile here</html>"   # 200, not an image
            return b"\xff\xd8\xff" + qk.encode()   # JPEG magic + payload
        # asset tile service: requires the CURRENT endpoint token
        if headers.get("Authorization") != "Bearer " + self.ep_token:
            if self.expire_first_tile_token:
                # accept the next attempt (simulates an expired-then-refreshed
                # endpoint token)
                self.ep_token = EP_TOKEN_2
                self.expire_first_tile_token = False
            raise provider.TileFetchError(f"HTTP 401 for {url}")
        if url.endswith("/layer.json"):
            return json.dumps(LAYER_JSON).encode()
        if url.endswith("/tilemapresource.xml"):
            return self.tms_xml
        if url.endswith(".terrain?v=1.2.0"):
            return b"fake-terrain-bytes"
        raise provider.TileNotFound(url)


@pytest.fixture()
def server(monkeypatch):
    srv = FakeServer()
    monkeypatch.setattr(provider, "_http_get", srv)
    return srv


def _terrain(server, tmp_path, asset_id=1):
    ion = provider.IonAsset(asset_id, TOKEN)
    cache = provider.DiskCache(str(tmp_path), ion.namespace)
    return provider.TerrainProvider("", cache, ion=ion)


def test_ion_terrain_connect(server, tmp_path):
    tp = _terrain(server, tmp_path)
    layer = tp.connect()
    assert layer["format"] == "quantized-mesh-1.0"
    assert tp.base_url == "https://assets.example.com/1"
    assert tp.ion.attributions == ["Test Attribution"]
    # endpoint call used the USER token; layer.json used the ENDPOINT token
    ep_call, layer_call = server.calls
    assert ep_call[1]["Authorization"] == "Bearer " + TOKEN
    assert layer_call[1]["Authorization"] == "Bearer " + EP_TOKEN_1


def test_ion_tile_fetch_and_cache_namespace(server, tmp_path):
    tp = _terrain(server, tmp_path)
    tp.connect()
    data = tp.fetch_tile(0, 0, 0)
    assert data == b"fake-terrain-bytes"
    p = tp.cache.terrain_path(0, 0, 0)
    assert "ion-1" in p
    # second fetch comes from disk, no new HTTP call
    n = len(server.calls)
    assert tp.fetch_tile(0, 0, 0) == data
    assert len(server.calls) == n


def test_401_refreshes_endpoint_token_once(server, tmp_path):
    tp = _terrain(server, tmp_path)
    tp.connect()
    server.ep_token = "rotated-away"          # server invalidates our token
    server.expire_first_tile_token = True     # next resolve() gets EP_TOKEN_2
    data = tp.fetch_tile(0, 0, 0)
    assert data == b"fake-terrain-bytes"
    assert tp.ion.access_token == EP_TOKEN_2
    # exactly one extra endpoint resolve happened
    resolves = [c for c in server.calls if "/v1/assets/" in c[0]]
    assert len(resolves) == 2


def test_wrong_asset_type_rejected(server, tmp_path):
    tp = _terrain(server, tmp_path, asset_id=3954)   # IMAGERY, not TERRAIN
    with pytest.raises(provider.TileFetchError, match="expected TERRAIN"):
        tp.connect()


def test_external_3dtiles_asset_uses_provider_url_and_key(server, tmp_path):
    """Google Photorealistic: ion answers externalType=3DTILES with the
    provider root URL + API key in options; the key must ride the query."""
    server.asset_types[2275207] = "3DTILES"
    server.external[2275207] = "3DTILES"
    # reuse FakeServer's external-endpoint shape via a custom route
    orig = server.__class__.__call__

    def call(self, url, accept="*/*", headers=None):
        if "/v1/assets/2275207/endpoint" in url:
            self.calls.append((url, dict(headers or {})))
            return json.dumps({
                "type": "3DTILES",
                "externalType": "3DTILES",
                "options": {
                    "url": "https://tile.google.example/v1/3dtiles/root.json",
                    "key": "goog-key",
                },
            }).encode()
        return orig(self, url, accept, headers)

    server.__class__.__call__ = call
    try:
        ion = provider.IonAsset(2275207, TOKEN)
        prov = provider.Tiles3DProvider(
            "", provider.DiskCache(str(tmp_path), ion.namespace), ion=ion
        )
        root_url = prov.connect()
        assert root_url == (
            "https://tile.google.example/v1/3dtiles/root.json?key=goog-key"
        )
        assert ion.access_token is None      # external: no ion Bearer
    finally:
        server.__class__.__call__ = orig


def test_unsupported_external_asset_rejected(server, tmp_path):
    server.asset_types[99] = "IMAGERY"
    server.external[99] = "GOOGLE_EARTH_ENTERPRISE"
    ion = provider.IonAsset(99, TOKEN)
    ip = provider.ImageryProvider("", provider.DiskCache(str(tmp_path)), ion=ion)
    with pytest.raises(provider.TileFetchError, match="externally hosted"):
        ip.connect()


def test_bad_user_token_clear_error(server, tmp_path):
    ion = provider.IonAsset(1, "wrong-token")
    with pytest.raises(provider.TileFetchError, match="rejected the access token"):
        ion.resolve()


def test_ion_imagery_geodetic_connects(server, tmp_path):
    ion = provider.IonAsset(3954, TOKEN)
    ip = provider.ImageryProvider("", provider.DiskCache(str(tmp_path)), ion=ion)
    info = ip.connect()
    assert info["title"] == "Test Imagery"
    assert info["max_zoom"] == 13


def test_mercator_tms_accepted(server, tmp_path):
    server.tms_xml = TMS_MERCATOR
    ion = provider.IonAsset(3954, TOKEN)
    ip = provider.ImageryProvider("", provider.DiskCache(str(tmp_path)), ion=ion)
    ip.connect()
    assert ip.scheme == "mercator"
    assert ip.merc_max_zoom == 13
    assert ip.max_zoom == 12    # terrain-z equivalent: mercator z+1 drapes z


# ------------------------------------------------------------------ Bing


def _bing_provider(server, tmp_path):
    server.asset_types[2] = "IMAGERY"
    server.external[2] = "BING"
    ion = provider.IonAsset(2, TOKEN)
    cache = provider.DiskCache(str(tmp_path), ion.namespace)
    return provider.ImageryProvider("", cache, ion=ion)


def test_bing_connect(server, tmp_path):
    ip = _bing_provider(server, tmp_path)
    info = ip.connect()
    assert ip.scheme == "mercator"
    assert info["title"] == "Bing Aerial"
    assert ip.merc_min_zoom == 1 and ip.merc_max_zoom == 21
    assert ip.max_zoom == 20
    meta_call = next(c for c in server.calls if "/REST/v1/" in c[0])
    assert "key=bing-key-xyz" in meta_call[0]
    assert "uriScheme=https" in meta_call[0]


def test_bing_quadkey_and_tile_url(server, tmp_path):
    # Bing docs example: level 3, x=3, y_top=5 -> quadkey "213"
    assert provider._bing_quadkey(3, 3, 5) == "213"
    ip = _bing_provider(server, tmp_path)
    ip.connect()
    # our y is TMS (south origin): y_top = 2^3 - 1 - y  ->  y = 2 gives y_top 5
    url = ip.tile_url(3, 3, 2)
    assert "/tiles/a213.jpeg" in url
    assert "n=z" in url
    assert any(f"//ecn.t{i}." in url for i in range(3))   # subdomain rotated in


def test_bing_fetch_sniffs_jpeg_and_404s(server, tmp_path):
    ip = _bing_provider(server, tmp_path)
    ip.connect()
    path = ip.fetch_tile(3, 3, 2)
    assert path.endswith(".jpg") and "ion-2" in path
    server.bing_missing = {provider._bing_quadkey(4, 5, 9)}
    with pytest.raises(provider.TileNotFound):
        ip.fetch_tile(4, 5, (1 << 4) - 1 - 9)


def test_garbage_200_response_treated_as_absent(server, tmp_path):
    """A 200 with a non-image body (Bing at zooms it lacks) must behave
    like a 404 — caching it as 'imagery' made bpy fail to load it and tiles
    rendered pink."""
    ip = _bing_provider(server, tmp_path)
    ip.connect()
    server.bing_garbage = {provider._bing_quadkey(5, 3, 7)}
    y_tms = (1 << 5) - 1 - 7
    with pytest.raises(provider.TileNotFound):
        ip.fetch_tile(5, 3, y_tms)
    # nothing usable was cached
    assert ip.cache.imagery_path(5, 3, y_tms) is None


def test_stale_bin_cache_entries_ignored(server, tmp_path):
    """Pre-fix caches may hold .bin garbage — imagery_path must skip it."""
    ip = _bing_provider(server, tmp_path)
    d = ip.cache._dir("imagery", 9, 9)
    import os
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "9.bin"), "wb") as f:
        f.write(b"<html>not a tile</html>")
    assert ip.cache.imagery_path(9, 9, 9) is None


def test_stitched_path_shortcut(server, tmp_path):
    """paths_for_key prefers a persisted stitched composite when present."""
    import os
    ip = _bing_provider(server, tmp_path)
    key = (14, 100, 200, 100, 201)
    sp = ip.cache.stitched_path(key)
    assert sp.endswith("14_100_200_100_201.png")
    assert ip.paths_for_key(key) is None       # nothing cached yet
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    with open(sp, "wb") as f:
        f.write(b"\x89PNG fake")
    assert ip.paths_for_key(key) == [sp]


# ------------------------------------------------------- mercator scheme math


def test_merc_norm_xy_landmarks():
    x, y = tiling.merc_norm_xy(-math.pi, 0.0)
    assert x == pytest.approx(0.0) and y == pytest.approx(0.5)
    _, y_top = tiling.merc_norm_xy(0.0, tiling.MERC_LAT_LIMIT_RAD)
    _, y_bot = tiling.merc_norm_xy(0.0, -tiling.MERC_LAT_LIMIT_RAD)
    assert y_top == pytest.approx(1.0) and y_bot == pytest.approx(0.0)
    # beyond the limit clamps instead of diverging
    _, y_pole = tiling.merc_norm_xy(0.0, math.pi / 2)
    assert y_pole == pytest.approx(1.0)


def test_mercator_cover_invariant():
    """The cover must fully contain the geodetic rect (lat clamped), use one
    column and at most two rows, and never sit deeper than z+1."""
    samples = [(2, 0, 1), (5, 40, 20), (8, 300, 180), (10, 1180, 654),
               (3, 0, 6), (0, 0, 0), (13, 5851, 5397)]
    for z, x, y in samples:
        cover = tiling.mercator_cover(z, x, y, 21, 1)
        assert cover is not None, (z, x, y)
        m, tx0, ty0, tx1, ty1 = cover
        assert m <= z + 1
        assert tx1 == tx0
        assert 0 <= ty1 - ty0 <= 1
        west, south, east, north = tiling.tile_rect(z, x, y)
        x0, y0 = tiling.merc_norm_xy(west, south)
        x1, y1 = tiling.merc_norm_xy(east, north)
        rx0, ry0, rx1, ry1 = tiling.mercator_cover_rect(cover)
        eps = 1e-9
        assert rx0 <= x0 + eps and x1 - eps <= rx1
        assert ry0 <= y0 + eps and y1 - eps <= ry1


def test_mercator_cover_overzoom():
    """Leaf tiles (terrain at its data limit) get a deeper grid: at
    overzoom=2 the cover sits ~3 levels deeper with up to a 4x8 grid,
    still fully containing the tile — imagery sharpens past the mesh."""
    z, x, y = 13, 11705, 5398        # the Delhi z13 leaf tile (CWT limit)
    plain = tiling.mercator_cover(z, x, y, 21, 1)
    deep = tiling.mercator_cover(z, x, y, 21, 1, overzoom=2)
    m, tx0, ty0, tx1, ty1 = deep
    assert m == plain[0] + 2 or m == z + 3
    assert m > plain[0]
    assert tx1 - tx0 + 1 <= 4 and ty1 - ty0 + 1 <= 8
    west, south, east, north = tiling.tile_rect(z, x, y)
    x0, y0 = tiling.merc_norm_xy(west, south)
    x1, y1 = tiling.merc_norm_xy(east, north)
    rx0, ry0, rx1, ry1 = tiling.mercator_cover_rect(deep)
    eps = 1e-9
    assert rx0 <= x0 + eps and x1 - eps <= rx1
    assert ry0 <= y0 + eps and y1 - eps <= ry1


def test_mercator_cover_adjacent_consistency():
    """Field regression (lat 28.61, lon 77.20): under the old single-cover
    rule, whether a terrain tile nested in one mercator tile depended on row
    alignment, so neighbors could land 4+ imagery levels apart (13 next to
    9). The two-row cover must keep any neighborhood within one level, at or
    above the terrain level."""
    z = 13
    _, x, y = tiling.lonlat_to_tile(z, 77.20, 28.61)
    levels = set()
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            cover = tiling.mercator_cover(z, x + dx, y + dy, 21, 1)
            levels.add(cover[0])
    assert max(levels) - min(levels) <= 1
    assert min(levels) >= z


def test_mercator_uv_reprojection():
    """Corner vertices of a terrain tile must land on the geodetic rect's
    position inside the covering mercator tile, with v nonlinear in lat."""
    from cesium_for_blender.core import quantized_mesh, wgs84

    z, x, y = (5, 40, 20)
    merc = tiling.mercator_cover(z, x, y, 21, 1)
    rect = tiling.mercator_cover_rect(merc)
    qm = quantized_mesh.QMTile(
        center_ecef=np.zeros(3), min_h=0.0, max_h=0.0,
        bs_center=np.zeros(3), bs_radius=1.0, horizon_occlusion=np.zeros(3),
        u=np.array([0, 32767, 32767], dtype=np.int32),
        v=np.array([0, 0, 32767], dtype=np.int32),
        h=np.array([0, 0, 0], dtype=np.int32),
        indices=np.array([[0, 1, 2]], dtype=np.uint32),
        west_i=np.array([], dtype=np.uint32), south_i=np.array([], dtype=np.uint32),
        east_i=np.array([], dtype=np.uint32), north_i=np.array([], dtype=np.uint32),
    )
    west, south, east, north = tiling.tile_rect(z, x, y)
    frame = wgs84.EnuFrame(
        math.degrees((south + north) / 2), math.degrees((west + east) / 2)
    )
    md = quantized_mesh.tile_to_enu_mesh(
        qm, (z, x, y), frame, imagery_key=merc, mercator_rect=rect
    )
    uvs = md.loop_uvs.reshape(-1, 2)   # loops = tri (0,1,2) -> per-vertex uvs
    x0, y0, x1, y1 = rect
    ex_sw_x, ex_sw_y = tiling.merc_norm_xy(west, south)
    ex_ne_x, ex_ne_y = tiling.merc_norm_xy(east, north)
    assert uvs[0][0] == pytest.approx((ex_sw_x - x0) / (x1 - x0), abs=1e-5)
    assert uvs[0][1] == pytest.approx((ex_sw_y - y0) / (y1 - y0), abs=1e-5)
    assert uvs[2][0] == pytest.approx((ex_ne_x - x0) / (x1 - x0), abs=1e-5)
    assert uvs[2][1] == pytest.approx((ex_ne_y - y0) / (y1 - y0), abs=1e-5)
    assert md.imagery_key == merc


def test_layer_json_availability_is_absolute():
    """layer.json rects must land at level i exactly — offset detection would
    accept the subtree convention too and shift everything one level down."""
    avail = tiling.AvailabilityIndex()
    assert avail.ingest_layer_json(LAYER_JSON["available"])
    assert avail.is_available(1, 0, 0)
    assert avail.is_available(1, 3, 1)
    assert not avail.is_available(2, 0, 0)   # level 2 was never declared
    assert avail.max_known_level() == 1
    # idempotent: a second ingest is a no-op
    gen = avail.generation
    assert not avail.ingest_layer_json(LAYER_JSON["available"])
    assert avail.generation == gen


def test_layer_json_and_tile_metadata_compose():
    avail = tiling.AvailabilityIndex()
    avail.ingest_layer_json(LAYER_JSON["available"])
    # a z1 tile's metadata (subtree convention) extends availability deeper
    avail.ingest(
        (1, 0, 0),
        [[{"startX": 0, "startY": 0, "endX": 1, "endY": 1}]],
    )
    assert avail.is_available(2, 1, 1)
    assert avail.max_known_level() == 2


def test_url_namespace_slug():
    ns = provider.url_namespace("http://10.0.0.1:81/api/terrain")
    assert ns == "10.0.0.1_81_api_terrain"
    assert provider.url_namespace("https://a.example.com/x/") == "a.example.com_x"
