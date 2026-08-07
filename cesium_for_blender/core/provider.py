"""HTTP tile providers + disk cache. Pure module: stdlib + nothing else.

Server quirks (verified against the live server):
- GET only (HEAD returns 405).
- Terrain URL template comes from layer.json "tiles"[0] with a ?v={version}
  query — never hardcode the version.
- Imagery endpoint declares image/png but serves JPEG bytes: sniff magic bytes
  and store with the true extension so bpy.data.images.load succeeds.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

TERRAIN_ACCEPT = (
    "application/vnd.quantized-mesh;extensions=octvertexnormals,"
    "application/octet-stream;q=0.9,*/*;q=0.1"
)
USER_AGENT = "CesiumForBlender/0.1"
TIMEOUT_S = 10.0


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


def default_cache_root() -> str:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return os.path.join(base, "CesiumForBlender", "cache")


def _http_get(url: str, accept: str = "*/*") -> bytes:
    req = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise TileNotFound(url) from e
        raise TileFetchError(f"HTTP {e.code} for {url}", connection=False) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise TileFetchError(
            f"{type(e).__name__}: {e} for {url}", connection=True
        ) from e


def sniff_image_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"\x89PNG":
        return ".png"
    if data[:4] == b"II*\x00" or data[:4] == b"MM\x00*":
        return ".tif"
    return ".bin"


class DiskCache:
    """<root>/terrain/z/x/y.terrain and <root>/imagery/z/x/y.<sniffed ext>.
    Atomic writes (tmp + os.replace) so a crashed Blender never leaves a
    truncated tile behind."""

    IMG_EXTS = (".jpg", ".png", ".tif", ".bin")

    def __init__(self, root: str | None = None):
        self.root = root or default_cache_root()

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
    def __init__(self, base_url: str, cache: DiskCache):
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.layer: dict | None = None
        self._template: str | None = None

    def connect(self) -> dict:
        data = _http_get(self.base_url + "/layer.json", accept="application/json")
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
        data = _http_get(self.tile_url(z, x, y), accept=TERRAIN_ACCEPT)
        self.cache.put("terrain", z, x, y, data, ".terrain")
        return data


class ImageryProvider:
    def __init__(self, base_url: str, cache: DiskCache):
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.title: str | None = None
        self.max_zoom: int = 13
        self.tile_ext: str = ".png"  # what the SERVER calls it; bytes are sniffed
        # Dedup concurrent fetches of the same imagery tile (deep terrain tiles
        # share z13 ancestors): first thread fetches, others wait on its Event.
        self._inflight: dict[tuple, threading.Event] = {}
        self._inflight_lock = threading.Lock()

    def connect(self) -> dict:
        data = _http_get(
            self.base_url + "/tilemapresource.xml", accept="application/xml,*/*"
        )
        root = ET.fromstring(data)
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
            self.max_zoom = max(orders)
        return {"title": self.title, "max_zoom": self.max_zoom}

    def tile_url(self, z: int, x: int, y: int) -> str:
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
            data = _http_get(self.tile_url(z, x, y), accept="image/*,*/*")
            path = self.cache.put("imagery", z, x, y, data, sniff_image_ext(data))
            return path
        finally:
            with self._inflight_lock:
                self._inflight.pop(key).set()
