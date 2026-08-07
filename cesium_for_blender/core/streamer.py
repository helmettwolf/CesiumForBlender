"""Streaming orchestrator: worker pool + timer pump + tile lifecycle.

Threading contract:
- Workers (ThreadPoolExecutor) run PURE code only: fetch, decode, ENU
  transform, imagery-to-disk. They never touch bpy.
- The bpy.app.timers callback (main thread) does everything else: camera
  reads, LOD selection, mesh/material builds (budgeted per tick), visibility
  diffs, eviction.
"""

from __future__ import annotations

import queue
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import bpy

from . import camera, lod, provider, quantized_mesh, scene_builder, tiling, wgs84
from .cache import TileCache, TileState

TICK_BUSY_S = 0.05
TICK_IDLE_S = 0.1
SELECTION_MAX_AGE_S = 0.5
BUILD_BUDGET_COUNT = 3
BUILD_BUDGET_S = 0.008
EVICT_PERIOD_S = 2.0
RETRY_BACKOFF_S = (1.0, 4.0, 15.0)
BREAKER_THRESHOLD = 8       # consecutive CONNECTION failures (not HTTP errors)
BREAKER_PROBE_PERIOD_S = 10.0


class Stats:
    def __init__(self):
        self.status = "disconnected"
        self.visible = 0
        self.built = 0
        self.fetching = 0
        self.queued = 0
        self.failed = 0
        self.dead = 0
        self.render_level_max = 0
        self.last_error = ""


class Streamer:
    def __init__(self):
        self.terrain: provider.TerrainProvider | None = None
        self.imagery: provider.ImageryProvider | None = None
        self.avail = tiling.AvailabilityIndex()
        self.frame: wgs84.EnuFrame | None = None
        self.tiles = TileCache()
        self.stats = Stats()
        self.connected = False
        self.running = False
        self.generation = 0           # bumped on origin change / clear
        self.executor: ThreadPoolExecutor | None = None
        self.results: queue.Queue = queue.Queue()
        self.inflight = 0
        self.max_requests = 6
        self.sse_threshold = 16.0
        self.max_level = 19
        self.tile_budget = 400
        self._current = lod.SelectionResult()
        self._render_set: set = set()
        self._last_sig = None
        self._last_selection_t = 0.0
        self._last_avail_gen = -1
        self._last_evict_t = 0.0
        self._conn_failures = 0
        self._breaker_probe_t = 0.0
        self._breaker_open = False

    # ------------------------------------------------------------- lifecycle

    def connect(self, terrain_url: str, imagery_url: str, cache_dir: str = ""):
        """Synchronous (operator-driven): layer.json + tilemapresource.xml +
        both root tiles (whose metadata seeds the availability index, enabling
        'go to data center' immediately)."""
        cache = provider.DiskCache(cache_dir or None)
        self.terrain = provider.TerrainProvider(terrain_url, cache)
        self.imagery = provider.ImageryProvider(imagery_url, cache)
        self.terrain.connect()
        self.imagery.connect()
        for root in ((0, 0, 0), (0, 1, 0)):
            try:
                qm = quantized_mesh.decode(self.terrain.fetch_tile(*root))
                if qm.metadata and qm.metadata.get("available"):
                    self.avail.ingest(root, qm.metadata["available"])
            except (provider.TileNotFound, provider.TileFetchError,
                    quantized_mesh.QMDecodeError) as e:
                print(f"[cesium] root {root} metadata unavailable: {e}")
        self.connected = True
        self.stats.status = "connected"

    def set_origin(self, lat_deg: float, lon_deg: float):
        self.frame = wgs84.EnuFrame(lat_deg, lon_deg)
        self.generation += 1
        for t in self.tiles.tiles.values():
            t.aabb = None             # ENU-dependent, recompute lazily

    def start(self):
        if self.running:
            return
        if not self.connected or self.frame is None:
            raise RuntimeError("connect and set an origin first")
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_requests, thread_name_prefix="cesium"
        )
        self.running = True
        self.stats.status = "streaming"
        bpy.app.timers.register(self._tick, first_interval=TICK_BUSY_S)

    def stop(self):
        self.running = False            # timer returns None on next tick
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
        self.inflight = 0
        # tiles stuck FETCHING will never deliver; requeue them
        for t in self.tiles.tiles.values():
            if t.state == TileState.FETCHING:
                t.state = TileState.QUEUED
        if self.stats.status == "streaming":
            self.stats.status = "stopped"

    def clear(self):
        self.stop()
        scene_builder.clear_all()
        self.tiles = TileCache()
        self.results = queue.Queue()
        self._current = lod.SelectionResult()
        self._render_set = set()
        self._last_sig = None
        self.generation += 1

    # ------------------------------------------------------------ worker side

    def _worker_load(self, key, gen: int):
        """Runs on the executor. Pure code only; posts to self.results."""
        try:
            z, x, y = key
            data = self.terrain.fetch_tile(z, x, y)
            qm = quantized_mesh.decode(data)
            if qm.metadata and qm.metadata.get("available"):
                self.avail.ingest(key, qm.metadata["available"])

            img_path = None
            img_key = None
            scale, uo, vo = 1.0, 0.0, 0.0
            az = min(z, self.imagery.max_zoom)
            while az >= 0:
                k, s, u, v = tiling.uv_transform_to_ancestor(z, x, y, az)
                try:
                    img_path = self.imagery.fetch_tile(*k)
                    img_key, scale, uo, vo = k, s, u, v
                    break
                except (provider.TileNotFound, provider.TileFetchError):
                    az -= 1

            md = quantized_mesh.tile_to_enu_mesh(
                qm, key, self.frame, scale, (uo, vo), imagery_key=img_key
            )
            self.results.put(("ok", key, gen, md, img_path))
        except provider.TileNotFound:
            self.results.put(("dead", key, gen, "404", None))
        except quantized_mesh.QMDecodeError as e:
            self.results.put(("dead", key, gen, f"decode: {e}", None))
        except provider.TileFetchError as e:
            self.results.put(
                ("retry", key, gen, str(e), bool(getattr(e, "connection", False)))
            )
        except Exception:
            self.results.put(("dead", key, gen, traceback.format_exc(), None))

    def _worker_probe(self, gen: int):
        try:
            self.terrain.connect()
            self.results.put(("probe_ok", None, gen, None, None))
        except Exception as e:
            self.results.put(("probe_fail", None, gen, str(e), None))

    # ------------------------------------------------------------- main loop

    def _tick(self):
        if not self.running:
            return None
        try:
            self._tick_inner()
        except Exception:
            # a crashing timer would silently unregister; keep running but log
            print("[cesium] tick error:\n" + traceback.format_exc())
            self.stats.last_error = "tick error (see console)"
        busy = (
            self.inflight > 0
            or not self.results.empty()
            or any(
                t.state == TileState.DECODED for t in self.tiles.tiles.values()
            )
        )
        return TICK_BUSY_S if busy else TICK_IDLE_S

    def _tick_inner(self):
        now = time.monotonic()
        cam = camera.get_camera_state()

        built_or_died = self._drain_results(now)

        need_selection = cam is not None and (
            cam.signature != self._last_sig
            or now - self._last_selection_t > SELECTION_MAX_AGE_S
            or self.avail.generation != self._last_avail_gen
            or built_or_died
        )
        if need_selection:
            self._current = lod.select_tiles(
                self.tiles, self.avail, cam, self.frame,
                self.sse_threshold, self.max_level, now,
                imagery_max_z=self.imagery.max_zoom if self.imagery else 13,
            )
            self._render_set = set(self._current.render)
            self._last_sig = cam.signature
            self._last_selection_t = now
            self._last_avail_gen = self.avail.generation
            self._apply_visibility()

        self._submit_loads(now)

        if now - self._last_evict_t > EVICT_PERIOD_S:
            self._last_evict_t = now
            self._evict()
            self.tiles.drop_stale_mesh_data()

        self._update_stats()

    def _drain_results(self, now: float) -> bool:
        deadline = now + BUILD_BUDGET_S
        builds = 0
        changed = False
        while builds < BUILD_BUDGET_COUNT and time.monotonic() < deadline:
            try:
                kind, key, gen, payload, extra = self.results.get_nowait()
            except queue.Empty:
                break
            if kind == "probe_ok":
                self._conn_failures = 0
                if self._breaker_open:
                    self._breaker_open = False
                    self.stats.status = "streaming"
                continue
            if kind == "probe_fail":
                self._breaker_probe_t = time.monotonic()
                continue

            self.inflight = max(0, self.inflight - 1)
            tile = self.tiles.get(key)
            if tile is None:
                continue
            if gen != self.generation:
                if tile.state == TileState.FETCHING:
                    tile.state = TileState.QUEUED   # stale frame; refetch later
                continue

            if kind == "ok":
                self._conn_failures = 0
                md, img_path = payload, extra
                tile.mesh_data = md
                tile.decoded_at = now
                tile.state = TileState.DECODED
                if md.geometric_error is not None:
                    tile.geometric_error = md.geometric_error
                tile.aabb = (md.aabb_min, md.aabb_max)
                tile.min_h, tile.max_h = md.min_h, md.max_h
                self._build_tile(tile, img_path)
                builds += 1
                changed = True
            elif kind == "dead":
                tile.state = TileState.DEAD
                if "404" not in str(payload):
                    print(f"[cesium] tile {key} dead: {payload}")
                changed = True
            elif kind == "retry":
                is_conn = bool(extra)
                if is_conn:
                    self._conn_failures += 1
                tile.retries += 1
                if tile.retries > len(RETRY_BACKOFF_S):
                    tile.state = TileState.DEAD
                    changed = True
                else:
                    tile.state = TileState.FAILED
                    tile.next_retry = now + RETRY_BACKOFF_S[tile.retries - 1]
                if (
                    self._conn_failures >= BREAKER_THRESHOLD
                    and not self._breaker_open
                ):
                    self._breaker_open = True
                    self.stats.status = "server unreachable — retrying"

        # build any left-over DECODED tiles within the same budget window
        if builds < BUILD_BUDGET_COUNT:
            for tile in self.tiles.tiles.values():
                if builds >= BUILD_BUDGET_COUNT or time.monotonic() >= deadline:
                    break
                if tile.state == TileState.DECODED and tile.mesh_data is not None:
                    self._build_tile(tile, None)
                    builds += 1
                    changed = True

        if changed:
            scene_builder.tag_redraw_view3d()
        return changed

    def _build_tile(self, tile, img_path: str | None):
        from . import map_style

        md = tile.mesh_data
        relief = (
            bpy.data.materials.get(map_style.RELIEF_MAT_NAME)
            if map_style.is_active()
            else None
        )
        if relief is not None:
            mat = relief
        else:
            if img_path is None and md.imagery_key is not None and self.imagery:
                img_path = self.imagery.cache.imagery_path(*md.imagery_key)
            if img_path is not None and md.imagery_key is not None:
                mat = scene_builder.get_or_create_material(md.imagery_key, img_path)
            else:
                mat = scene_builder.get_or_create_gray_material()
        obj = scene_builder.build_tile_object(tile.key, md, mat)
        obj.hide_viewport = True
        obj.hide_render = True
        tile.visible = False
        tile.state = TileState.BUILT
        tile.mesh_data = None

    def _apply_visibility(self):
        """Replacement refinement: shows before hides, so a parent never
        disappears in the same tick its children appear a frame late."""
        for key in self._render_set:
            t = self.tiles.get(key)
            if t is not None and t.state == TileState.BUILT and not t.visible:
                if scene_builder.set_tile_visible(key, True):
                    t.visible = True
        for t in self.tiles.tiles.values():
            if (
                t.state == TileState.BUILT
                and t.visible
                and t.key not in self._render_set
            ):
                if scene_builder.set_tile_visible(t.key, False):
                    t.visible = False

    def _submit_loads(self, now: float):
        if self.executor is None:
            return
        if self._breaker_open:
            if now - self._breaker_probe_t > BREAKER_PROBE_PERIOD_S:
                self._breaker_probe_t = now
                self.executor.submit(self._worker_probe, self.generation)
            return
        for _prio, key in self._current.load:
            if self.inflight >= self.max_requests:
                break
            tile = self.tiles.get(key)
            if tile is None:
                continue
            if tile.state == TileState.QUEUED or (
                tile.state == TileState.FAILED and tile.next_retry <= now
            ):
                tile.state = TileState.FETCHING
                self.inflight += 1
                self.executor.submit(self._worker_load, key, self.generation)

    def _evict(self):
        victims = self.tiles.evictable(self._current.keep, self.tile_budget)
        for tile in victims:
            scene_builder.destroy_tile_object(tile.key)
            tile.state = TileState.QUEUED
            tile.visible = False
            tile.mesh_data = None
        if victims:
            scene_builder.sweep_unused_materials()

    def _update_stats(self):
        counts = self.tiles.state_counts()
        s = self.stats
        s.built = counts.get("built", 0)
        s.fetching = counts.get("fetching", 0)
        s.queued = counts.get("queued", 0)
        s.failed = counts.get("failed", 0)
        s.dead = counts.get("dead", 0)
        s.visible = sum(1 for t in self.tiles.tiles.values() if t.visible)
        s.render_level_max = max(
            (k[0] for k in self._render_set), default=0
        )


_instance: Streamer | None = None


def get() -> Streamer:
    global _instance
    if _instance is None:
        _instance = Streamer()
    return _instance


def shutdown():
    global _instance
    if _instance is not None:
        _instance.stop()
        _instance = None
