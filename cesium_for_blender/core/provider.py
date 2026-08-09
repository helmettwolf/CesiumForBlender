"""HTTP tile providers + disk cache. Pure module: stdlib + nothing else.

Server quirks (verified against the live server):
- GET only (HEAD returns 405).
- Terrain URL template comes from layer.json "tiles"[0] with a ?v={version}
  query — never hardcode the version.
- Imagery endpoint declares image/png but serves JPEG bytes: sniff magic bytes
  and store with the true extension so bpy.data.images.load succeeds.

Cesium ion:
- An asset is resolved via GET {ION_API}/v1/assets/{id}/endpoint with the
  user's token as an Authorization: Bearer header (never a query param).
- The endpoint reply carries the tile base URL plus a SHORT-LIVED asset-scoped
  accessToken; every tile request sends it as Bearer. On HTTP 401 the endpoint
  is re-resolved once and the request retried (CesiumJS does the same).
- externalType assets (Bing, Google) are not tile services we can drape;
  rejected at connect with a clear message.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

# metadata must be requested explicitly or ion omits the extension that feeds
# the availability index (the self-hosted server sends it regardless).
TERRAIN_ACCEPT = (
    "application/vnd.quantized-mesh;extensions=octvertexnormals-metadata,"
    "application/octet-stream;q=0.9,*/*;q=0.1"
)
USER_AGENT = "CesiumForBlender/0.1"
TIMEOUT_S = 10.0
ION_API = "https://api.cesium.com"


class TileNotFound(Exception):
    """404: the tile does not exist. Not retryable."""


class TileFetchError(Exception):
    """Network/server failure. Retryable.

    connection=True means the server itself was unreachable (timeout, refused,
    DNS) — the circuit breaker counts only these. connection=False is an HTTP
    error response; this server answers 500/502 for absent tiles (instead of
    404), so HTTP errors must never trip the breaker.
    """

    def __init__(self, msg: str, connection: bool = False):
        super().__init__(msg)
        self.connection = connection


class RequestCounter:
    """Session-wide HTTP request tally, shown in the panel. Thread-safe:
    workers bump it concurrently. not_found (404) is the server saying a
    tile does not exist — expected during ancestor fallback, not a failure."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total = 0
        self.ok = 0
        self.not_found = 0
        self.http_error = 0
        self.conn_error = 0

    def bump(self, field: str):
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)

    def reset(self):
        with self._lock:
            self.total = self.ok = self.not_found = 0
            self.http_error = self.conn_error = 0

    def snapshot(self) -> tuple[int, int, int, int, int]:
        with self._lock:
            return (self.total, self.ok, self.not_found,
                    self.http_error, self.conn_error)


REQUESTS = RequestCounter()


def default_cache_root() -> str:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return os.path.join(base, "CesiumForBlender", "cache")


def _http_get(url: str, accept: str = "*/*", headers: dict | None = None) -> bytes:
    hdrs = {"Accept": accept, "User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    REQUESTS.bump("total")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = resp.read()
            enc = (resp.headers.get("Content-Encoding") or "").lower()
        # ion's CDN gzips responses even without Accept-Encoding (urllib sends
        # none and does not auto-decompress). Sniff too, in case the header is
        # missing; a false-positive magic match just fails and keeps raw bytes.
        if "gzip" in enc or data[:2] == b"\x1f\x8b":
            try:
                data = gzip.decompress(data)
            except OSError:
                pass
        REQUESTS.bump("ok")
        return data
    except urllib.error.HTTPError as e:
        if e.code == 404:
            REQUESTS.bump("not_found")
            raise TileNotFound(url) from e
        REQUESTS.bump("http_error")
        raise TileFetchError(f"HTTP {e.code} for {url}", connection=False) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        REQUESTS.bump("conn_error")
        raise TileFetchError(
            f"{type(e).__name__}: {e} for {url}", connection=True
        ) from e


def url_namespace(url: str) -> str:
    """Filesystem-safe cache namespace for a server URL (host_port_path)."""
    stripped = re.sub(r"^[a-z]+://", "", url.strip().rstrip("/"), flags=re.I)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stripped)[:80] or "server"


class IonAsset:
    """One Cesium ion asset: resolves /v1/assets/{id}/endpoint into a tile
    base URL + short-lived Bearer token. resolve() is called again on 401.
    Thread-safe: workers may refresh concurrently (idempotent)."""

    def __init__(self, asset_id: int, token: str, api_base: str = ION_API):
        self.asset_id = int(asset_id)
        self.token = token.strip()
        self.api_base = api_base.rstrip("/")
        self.url: str | None = None
        self.access_token: str | None = None
        self.asset_type: str | None = None
        self.external_type: str | None = None   # e.g. "BING": not ion-hosted
        self.options: dict = {}                 # external service config (key…)
        self.attributions: list[str] = []
        self._lock = threading.Lock()

    @property
    def namespace(self) -> str:
        return f"ion-{self.asset_id}"

    def resolve(self) -> dict:
        if not self.token:
            raise TileFetchError("Cesium ion token is empty")
        ep_url = f"{self.api_base}/v1/assets/{self.asset_id}/endpoint"
        try:
            data = _http_get(
                ep_url,
                accept="application/json",
                headers={"Authorization": "Bearer " + self.token},
            )
        except TileFetchError as e:
            if "HTTP 401" in str(e):
                raise TileFetchError(
                    f"ion rejected the access token for asset {self.asset_id}"
                    " (401) — check the token and the asset id"
                ) from e
            raise
        except TileNotFound:
            raise TileFetchError(
                f"ion asset {self.asset_id} not found (404) — check the asset"
                " id and that the token can access it"
            ) from None
        ep = json.loads(data)
        external = ep.get("externalType") or None
        if external is None and "url" not in ep:
            raise TileFetchError(
                f"ion asset {self.asset_id} endpoint has no url"
                f" (type={ep.get('type')})"
            )
        with self._lock:
            self.external_type = external
            self.options = ep.get("options", {}) or {}
            if external is not None:
                # external services carry their own credentials in options
                # (e.g. Bing key); there is no ion Bearer token to send
                self.url = (self.options.get("url") or "").rstrip("/")
                self.access_token = None
            else:
                self.url = ep["url"].rstrip("/")
                self.access_token = ep.get("accessToken")
            self.asset_type = ep.get("type")
            self.attributions = [
                t
                for t in (
                    re.sub(r"<[^>]+>", "", a.get("html", "")).strip()
                    for a in ep.get("attributions", [])
                )
                if t
            ]
        return ep

    def headers(self) -> dict:
        with self._lock:
            if self.access_token:
                return {"Authorization": "Bearer " + self.access_token}
        return {}


def sniff_image_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"\x89PNG":
        return ".png"
    if data[:4] == b"II*\x00" or data[:4] == b"MM\x00*":
        return ".tif"
    return ".bin"


class DiskCache:
    """<root>/<namespace>/terrain/z/x/y.terrain and
    <root>/<namespace>/imagery/z/x/y.<sniffed ext>. The namespace keeps tiles
    from different sources (ion assets, servers) from colliding — coordinates
    alone are ambiguous across datasets. Atomic writes (tmp + os.replace) so a
    crashed Blender never leaves a truncated tile behind."""

    # .bin (unrecognized bytes) is deliberately NOT here: a server answering
    # 200 with a non-image body (Bing does this at zooms it lacks) must not
    # produce a "valid" cache entry — bpy cannot load it and tiles go pink
    IMG_EXTS = (".jpg", ".png", ".tif")

    def __init__(self, root: str | None = None, namespace: str = ""):
        base = root or default_cache_root()
        self.root = os.path.join(base, namespace) if namespace else base

    def _dir(self, kind: str, z: int, x: int) -> str:
        return os.path.join(self.root, kind, str(z), str(x))

    def terrain_path(self, z: int, x: int, y: int) -> str:
        return os.path.join(self._dir("terrain", z, x), f"{y}.terrain")

    def imagery_path(self, z: int, x: int, y: int) -> str | None:
        d = self._dir("imagery", z, x)
        for ext in self.IMG_EXTS:
            p = os.path.join(d, f"{y}{ext}")
            if os.path.isfile(p):
                return p
        return None

    def stitched_path(self, key: tuple) -> str:
        """Where a stitched mercator-cover composite is persisted (PNG).
        Stitching costs main-thread pixel work — done once, then rebuilds
        (and later sessions) load the file like any other tile."""
        return os.path.join(
            self.root, "stitched", "_".join(str(int(p)) for p in key) + ".png"
        )

    def blob_path(self, kind: str, name: str, ext: str) -> str:
        """<root>/<kind>/<name><ext> — for content addressed by URL hash
        (3D Tiles URIs are arbitrary paths, unlike z/x/y tiles)."""
        return os.path.join(self.root, kind, name + ext)

    def put_blob(self, kind: str, name: str, ext: str, data: bytes) -> str:
        d = os.path.join(self.root, kind)
        os.makedirs(d, exist_ok=True)
        final = os.path.join(d, name + ext)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, final)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return final

    def get_terrain(self, z: int, x: int, y: int) -> bytes | None:
        p = self.terrain_path(z, x, y)
        try:
            with open(p, "rb") as f:
                return f.read()
        except OSError:
            return None

    def put(self, kind: str, z: int, x: int, y: int, data: bytes, ext: str) -> str:
        d = self._dir(kind, z, x)
        os.makedirs(d, exist_ok=True)
        final = os.path.join(d, f"{y}{ext}")
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, final)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return final


class TerrainProvider:
    def __init__(self, base_url: str, cache: DiskCache, ion: IonAsset | None = None):
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.ion = ion
        self.layer: dict | None = None
        self._template: str | None = None

    def _get(self, url: str, accept: str) -> bytes:
        try:
            return _http_get(url, accept, headers=self.ion.headers() if self.ion else None)
        except TileFetchError as e:
            # expired ion endpoint token: re-resolve once, then retry
            if self.ion is not None and "HTTP 401" in str(e):
                self.ion.resolve()
                return _http_get(url, accept, headers=self.ion.headers())
            raise

    def connect(self) -> dict:
        if self.ion is not None:
            self.ion.resolve()
            if self.ion.external_type:
                raise TileFetchError(
                    f"ion terrain asset {self.ion.asset_id} is externally"
                    f" hosted ({self.ion.external_type}) — not a quantized-"
                    "mesh service this addon can stream"
                )
            if self.ion.asset_type != "TERRAIN":
                raise TileFetchError(
                    f"ion asset {self.ion.asset_id} is type"
                    f" {self.ion.asset_type or '?'}, expected TERRAIN"
                )
            self.base_url = self.ion.url
        data = self._get(self.base_url + "/layer.json", accept="application/json")
        layer = json.loads(data)
        tiles = layer.get("tiles") or ["{z}/{x}/{y}.terrain?v={version}"]
        version = layer.get("version", "1.0.0")
        self._template = (
            self.base_url + "/" + tiles[0].replace("{version}", str(version))
        )
        self.layer = layer
        return layer

    @property
    def max_zoom(self) -> int:
        return int(self.layer.get("maxzoom", 19)) if self.layer else 19

    def tile_url(self, z: int, x: int, y: int) -> str:
        if not self._template:
            raise TileFetchError("terrain provider not connected")
        return (
            self._template.replace("{z}", str(z))
            .replace("{x}", str(x))
            .replace("{y}", str(y))
        )

    def fetch_tile(self, z: int, x: int, y: int) -> bytes:
        cached = self.cache.get_terrain(z, x, y)
        if cached is not None:
            return cached
        data = self._get(self.tile_url(z, x, y), accept=TERRAIN_ACCEPT)
        self.cache.put("terrain", z, x, y, data, ".terrain")
        return data


class Tiles3DProvider:
    """3D Tiles content over HTTP: the base_url points at the root
    tileset.json (ion 3DTILES endpoints resolve to exactly that). Content is
    cached by URL hash — 3D Tiles URIs are arbitrary paths."""

    def __init__(self, base_url: str, cache: DiskCache, ion: IonAsset | None = None):
        self.base_url = base_url.strip()
        self.cache = cache
        self.ion = ion

    def _get(self, url: str, accept: str) -> bytes:
        try:
            return _http_get(url, accept, headers=self.ion.headers() if self.ion else None)
        except TileFetchError as e:
            if self.ion is not None and "HTTP 401" in str(e):
                self.ion.resolve()
                return _http_get(url, accept, headers=self.ion.headers())
            raise

    def connect(self) -> str:
        """Returns the root tileset.json URL."""
        if self.ion is not None:
            self.ion.resolve()
            if self.ion.external_type == "3DTILES":
                # externally hosted 3D Tiles (Google Photorealistic): the
                # endpoint carries the provider's root URL and API key in
                # options; the key rides the query string (Google protocol)
                # and query inheritance threads it plus the per-session
                # token through every child request
                url = (self.ion.options.get("url") or "").strip()
                if not url:
                    raise TileFetchError(
                        f"ion asset {self.ion.asset_id}: external 3DTILES"
                        " endpoint has no url"
                    )
                key = self.ion.options.get("key")
                if key and "key=" not in url:
                    url += ("&" if "?" in url else "?") + "key=" + key
                self.base_url = url
            elif self.ion.external_type:
                raise TileFetchError(
                    f"ion asset {self.ion.asset_id} is externally hosted"
                    f" ({self.ion.external_type}); not supported for 3D Tiles"
                )
            else:
                if self.ion.asset_type != "3DTILES":
                    raise TileFetchError(
                        f"ion asset {self.ion.asset_id} is type"
                        f" {self.ion.asset_type or '?'}, expected 3DTILES"
                    )
                self.base_url = self.ion.url
            if not self.base_url.lower().split("?")[0].endswith(".json"):
                self.base_url = self.base_url.rstrip("/") + "/tileset.json"
        if not self.base_url:
            raise TileFetchError("no tileset.json URL")
        return self.base_url

    @staticmethod
    def _hash(url: str) -> str:
        import hashlib

        return hashlib.sha1(url.encode()).hexdigest()

    def fetch_json(self, url: str) -> dict:
        h = self._hash(url)
        p = self.cache.blob_path("tiles3d", h, ".json")
        try:
            with open(p, "rb") as f:
                return json.loads(f.read())
        except (OSError, json.JSONDecodeError):
            pass
        data = self._get(url, accept="application/json,*/*")
        doc = json.loads(data)          # validate before caching
        self.cache.put_blob("tiles3d", h, ".json", data)
        return doc

    def fetch_content(self, url: str) -> bytes:
        h = self._hash(url)
        p = self.cache.blob_path("tiles3d", h, ".bin")
        try:
            with open(p, "rb") as f:
                return f.read()
        except OSError:
            pass
        data = self._get(url, accept="*/*")
        self.cache.put_blob("tiles3d", h, ".bin", data)
        return data

    def glb_path(self, url: str, index: int) -> str:
        return self.cache.blob_path("glb", f"{self._hash(url)}_{index}", ".glb")


def _bing_quadkey(m: int, x: int, y_top: int) -> str:
    """Bing quadkey for tile (m, x, y) with TOP-origin y (Bing/XYZ convention).
    Doc example: level 3, x=3, y=5 -> '213'."""
    digits = []
    for i in range(m, 0, -1):
        d = 0
        mask = 1 << (i - 1)
        if x & mask:
            d += 1
        if y_top & mask:
            d += 2
        digits.append(str(d))
    return "".join(digits)


class ImageryProvider:
    """TMS (geodetic or mercator) or Bing-quadkey imagery.

    scheme:
    - "geodetic": tiles share the terrain's (z,x,y) grid; UVs are a linear
      sub-window of an ancestor.
    - "mercator": square web-mercator grid (y=0 south, TMS-style); the
      streamer picks a covering mercator tile per terrain tile and vertices
      are reprojected. Bing (via an ion external asset) and EPSG:3857 TMS
      servers both land here; Bing only differs in URL construction.
    """

    def __init__(self, base_url: str, cache: DiskCache, ion: IonAsset | None = None):
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.ion = ion
        self.title: str | None = None
        self.scheme: str = "geodetic"
        self.max_zoom: int = 13       # terrain-z equivalent (LOD texel floor)
        self.merc_min_zoom: int = 0
        self.merc_max_zoom: int = 19
        self._bing_template: str | None = None
        self._bing_subdomains: list[str] = []
        self.tile_ext: str = ".png"  # what the SERVER calls it; bytes are sniffed
        # Dedup concurrent fetches of the same imagery tile (deep terrain tiles
        # share z13 ancestors): first thread fetches, others wait on its Event.
        self._inflight: dict[tuple, threading.Event] = {}
        self._inflight_lock = threading.Lock()

    def _get(self, url: str, accept: str) -> bytes:
        try:
            return _http_get(url, accept, headers=self.ion.headers() if self.ion else None)
        except TileFetchError as e:
            if self.ion is not None and "HTTP 401" in str(e):
                self.ion.resolve()
                return _http_get(url, accept, headers=self.ion.headers())
            raise

    def connect(self) -> dict:
        if self.ion is not None:
            self.ion.resolve()
            if self.ion.asset_type != "IMAGERY":
                raise TileFetchError(
                    f"ion asset {self.ion.asset_id} is type"
                    f" {self.ion.asset_type or '?'}, expected IMAGERY"
                )
            if self.ion.external_type == "BING":
                return self._connect_bing()
            if self.ion.external_type:
                raise TileFetchError(
                    f"ion imagery asset {self.ion.asset_id} is externally"
                    f" hosted ({self.ion.external_type}); only ion-hosted TMS"
                    " and Bing are supported"
                )
            self.base_url = self.ion.url
        data = self._get(
            self.base_url + "/tilemapresource.xml", accept="application/xml,*/*"
        )
        root = ET.fromstring(data)
        srs_el = root.find("SRS")
        srs = (srs_el.text or "").strip() if srs_el is not None else ""
        if any(tag in srs for tag in ("3857", "900913")):
            self.scheme = "mercator"
        title_el = root.find("Title")
        self.title = title_el.text if title_el is not None else None
        fmt = root.find("TileFormat")
        if fmt is not None:
            self.tile_ext = "." + fmt.get("extension", "png").lstrip(".")
        orders = [
            int(ts.get("order", "0"))
            for ts in root.findall("./TileSets/TileSet")
        ]
        if orders:
            if self.scheme == "mercator":
                self.merc_min_zoom = min(orders)
                self.merc_max_zoom = max(orders)
                # mercator level z+1 drapes terrain level z, so imagery stops
                # sharpening one terrain level earlier than its own max
                self.max_zoom = max(0, self.merc_max_zoom - 1)
            else:
                self.max_zoom = max(orders)
        return {"title": self.title, "max_zoom": self.max_zoom}

    def _connect_bing(self) -> dict:
        """ion external BING asset: ion supplies a Bing key in the endpoint
        options; tile URLs come from Bing's REST imagery metadata (quadkey
        template + subdomains). The key rides in the query string — that is
        the Bing REST protocol, and it is ion's service key, not the user's
        ion token."""
        opts = self.ion.options
        base = (opts.get("url") or "https://dev.virtualearth.net").rstrip("/")
        style = opts.get("mapStyle", "Aerial")
        meta_url = (
            f"{base}/REST/v1/Imagery/Metadata/{style}"
            f"?incl=ImageryProviders&key={opts.get('key', '')}&uriScheme=https"
        )
        meta = json.loads(_http_get(meta_url, accept="application/json"))
        try:
            res = meta["resourceSets"][0]["resources"][0]
            template = res["imageUrl"]
        except (KeyError, IndexError):
            raise TileFetchError(
                "Bing imagery metadata malformed:"
                f" {meta.get('errorDetails') or meta.get('statusDescription')}"
            ) from None
        self._bing_template = template.replace("{culture}", "en-US")
        self._bing_subdomains = list(res.get("imageUrlSubdomains") or ["t0"])
        self.scheme = "mercator"
        self.merc_min_zoom = max(1, int(res.get("zoomMin", 1)))
        self.merc_max_zoom = int(res.get("zoomMax", 21))
        self.max_zoom = max(0, self.merc_max_zoom - 1)
        self.title = f"Bing {style}"
        return {"title": self.title, "max_zoom": self.max_zoom}

    def paths_for_key(self, key: tuple) -> list[str] | None:
        """Cached disk paths for an imagery key: (z,x,y), or a mercator
        cover grid (m,tx0,ty0,tx1,ty1) in row-major south-first order,
        matching the stitcher. None unless every part is on disk."""
        if len(key) == 5:
            sp = self.cache.stitched_path(key)
            if os.path.isfile(sp):
                return [sp]
            m, tx0, ty0, tx1, ty1 = key
            paths = [
                self.cache.imagery_path(m, tx, ty)
                for ty in range(ty0, ty1 + 1)
                for tx in range(tx0, tx1 + 1)
            ]
        else:
            paths = [self.cache.imagery_path(*key)]
        return None if any(p is None for p in paths) else paths

    def tile_url(self, z: int, x: int, y: int) -> str:
        if self._bing_template is not None:
            y_top = (1 << z) - 1 - y            # scheme y is TMS (south origin)
            sub = self._bing_subdomains[(x + y_top) % len(self._bing_subdomains)]
            url = self._bing_template.replace("{subdomain}", sub).replace(
                "{quadkey}", _bing_quadkey(z, x, y_top)
            )
            # n=z makes Bing answer 404 for missing tiles instead of a
            # placeholder JPEG, so the ancestor-fallback walk can engage
            return url + ("&" if "?" in url else "?") + "n=z"
        return f"{self.base_url}/{z}/{x}/{y}{self.tile_ext}"

    def fetch_tile(self, z: int, x: int, y: int) -> str:
        """Returns the DISK PATH of the tile (bpy loads images from disk)."""
        key = (z, x, y)
        p = self.cache.imagery_path(z, x, y)
        if p is not None:
            return p
        with self._inflight_lock:
            ev = self._inflight.get(key)
            if ev is None:
                self._inflight[key] = threading.Event()
        if ev is not None:
            ev.wait(TIMEOUT_S * 3)
            p = self.cache.imagery_path(z, x, y)
            if p is not None:
                return p
            raise TileFetchError(f"inflight imagery fetch failed for {key}")
        try:
            data = self._get(self.tile_url(z, x, y), accept="image/*,*/*")
            ext = sniff_image_ext(data)
            if ext == ".bin":
                # 200 with a non-image body == "no tile here" (Bing at
                # zooms it lacks); treat as absent so the caller falls back
                raise TileNotFound(self.tile_url(z, x, y))
            path = self.cache.put("imagery", z, x, y, data, ext)
            return path
        finally:
            with self._inflight_lock:
                self._inflight.pop(key).set()
