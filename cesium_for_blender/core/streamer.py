"""Streaming orchestrator: worker pool + timer pump + tile lifecycle.

Threading contract:
- Workers (ThreadPoolExecutor) run PURE code only: fetch, decode, ENU
  transform, imagery-to-disk. They never touch bpy.
- The bpy.app.timers callback (main thread) does everything else: camera
  reads, LOD selection, mesh/material builds (budgeted per tick), visibility
  diffs, eviction.
"""

from __future__ import annotations

import os
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
# Deleting a burst of objects in one tick hitches the depsgraph; spread
# eviction over passes instead (they run every EVICT_PERIOD_S anyway).
EVICT_MAX_PER_PASS = 24
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
        # the post-rotation sharpen-up is fetch-bound (~6.5 builds/s at 6
        # concurrent requests, measured); CDNs are fine with more
        self.max_requests = 10
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
        self._last_ui_snap = None
        self._last_ui_redraw_t = 0.0

    # ------------------------------------------------------------- lifecycle

    def connect(
        self,
        terrain: provider.TerrainProvider,
        imagery: provider.ImageryProvider | None,
    ):
        """Synchronous (operator-driven): layer.json + tilemapresource.xml +
        both root tiles (whose metadata seeds the availability index, enabling
        'go to data center' immediately). The operator builds the providers
        (URL vs ion is a UI concern); imagery=None streams untextured terrain.
        """
        provider.REQUESTS.reset()
        self.terrain = terrain
        self.imagery = imagery
        self.terrain.connect()
        if self.imagery is not None:
            self.imagery.connect()
        # ion/CWT publish upper-level availability directly in layer.json
        # (absolute convention); tile metadata supplies the deeper levels
        layer_avail = (self.terrain.layer or {}).get("available")
        if layer_avail:
            self.avail.ingest_layer_json(layer_avail)
        for root in ((0, 0, 0), (0, 1, 0)):
            try:
                qm = quantized_mesh.decode(self.terrain.fetch_tile(*root))
                if qm.metadata and qm.metadata.get("available"):
                    self.avail.ingest(root, qm.metadata["available"])
            except (provider.TileNotFound, provider.TileFetchError,
                    quantized_mesh.QMDecodeError) as e:
                print(f"[cesium] root {root} metadata unavailable: {e}")
        self.connected = True
        self.stats.status = "streaming" if self.running else "connected"

    def set_origin(self, lat_deg: float, lon_deg: float):
        self.frame = wgs84.EnuFrame(lat_deg, lon_deg)
        self.generation += 1
        for t in self.tiles.tiles.values():
            t.aabb = None             # ENU-dependent, recompute lazily

    def start(self):
        if self.running:
            ensure_timer()
            return
        if not self.connected or self.frame is None:
            raise RuntimeError("connect and set an origin first")
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_requests, thread_name_prefix="cesium"
        )
        self.running = True
        self.stats.status = "streaming"
        ensure_timer()

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
            if self.avail.is_available(z, x, y):
                data = self.terrain.fetch_tile(z, x, y)
                qm = quantized_mesh.decode(data)
                if qm.metadata and qm.metadata.get("available"):
                    self.avail.ingest(key, qm.metadata["available"])
            else:
                # SYNTHETIC tile: the backend has no data this deep, so
                # upsample the nearest REAL ancestor (CesiumJS-style) — the
                # quadtree keeps subdividing and each smaller tile fetches
                # its own sharper imagery below
                ak = tiling.parent(z, x, y)
                while ak is not None and not self.avail.is_available(*ak):
                    ak = tiling.parent(*ak)
                if ak is None:
                    self.results.put(("dead", key, gen, "no real ancestor", None))
                    return
                aqm = quantized_mesh.decode(self.terrain.fetch_tile(*ak))
                qm = quantized_mesh.upsample(aqm, ak, key)

            img_paths = None
            img_key = None
            scale, uo, vo = 1.0, 0.0, 0.0
            merc_rect = None
            if self.imagery is not None and self.imagery.scheme == "mercator":
                cover = tiling.mercator_cover(
                    z, x, y,
                    self.imagery.merc_max_zoom, self.imagery.merc_min_zoom,
                )
                while cover is not None:
                    m, tx0, ty0, tx1, ty1 = cover
                    # a previously stitched composite on disk replaces both
                    # the source fetches and the main-thread stitch
                    sp = self.imagery.cache.stitched_path(cover)
                    if os.path.isfile(sp):
                        img_paths = [sp]
                        img_key = cover
                        merc_rect = tiling.mercator_cover_rect(cover)
                        break
                    try:
                        img_paths = [
                            self.imagery.fetch_tile(m, tx, ty)
                            for ty in range(ty0, ty1 + 1)
                            for tx in range(tx0, tx1 + 1)
                        ]
                        img_key = cover
                        merc_rect = tiling.mercator_cover_rect(cover)
                        break
                    except (provider.TileNotFound, provider.TileFetchError):
                        cover = (
                            tiling.mercator_cover(
                                z, x, y, m - 1, self.imagery.merc_min_zoom
                            )
                            if m > self.imagery.merc_min_zoom
                            else None
                        )
            elif self.imagery is not None:
                az = min(z, self.imagery.max_zoom)
                while az >= 0:
                    k, s, u, v = tiling.uv_transform_to_ancestor(z, x, y, az)
                    try:
                        img_paths = [self.imagery.fetch_tile(*k)]
                        img_key, scale, uo, vo = k, s, u, v
                        break
                    except (provider.TileNotFound, provider.TileFetchError):
                        az -= 1

            md = quantized_mesh.tile_to_enu_mesh(
                qm, key, self.frame, scale, (uo, vo), imagery_key=img_key,
                mercator_rect=merc_rect,
            )
            self.results.put(("ok", key, gen, md, img_paths))
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
                # no imagery -> no texel floor (nothing to sharpen): 0 makes
                # imagery_ge_floor return 0 for every level
                imagery_max_z=self.imagery.max_zoom if self.imagery else 0,
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

        # keep the panel's Status box live while anything moves (the sidebar
        # never redraws on its own — frozen counters read as "stalled")
        snap = (
            self.stats.built, self.stats.fetching, self.stats.queued,
            self.stats.failed, self.stats.dead, self.stats.visible,
            provider.REQUESTS.snapshot(),
        )
        if snap != self._last_ui_snap and now - self._last_ui_redraw_t > 0.25:
            self._last_ui_snap = snap
            self._last_ui_redraw_t = now
            scene_builder.tag_redraw_sidebar()

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
                md, img_paths = payload, extra
                tile.mesh_data = md
                tile.decoded_at = now
                tile.state = TileState.DECODED
                if md.geometric_error is not None:
                    tile.geometric_error = md.geometric_error
                tile.aabb = (md.aabb_min, md.aabb_max)
                tile.min_h, tile.max_h = md.min_h, md.max_h
                self._build_tile(tile, img_paths)
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

    def _build_tile(self, tile, img_paths: list | None):
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
            if img_paths is None and md.imagery_key is not None and self.imagery:
                img_paths = self.imagery.paths_for_key(md.imagery_key)
            if img_paths and md.imagery_key is not None:
                # tag keeps mercator material names from colliding with
                # geodetic ones that share the same (z,x,y) numbers
                tag = "M" if self.imagery.scheme == "mercator" else ""
                save_path = (
                    self.imagery.cache.stitched_path(md.imagery_key)
                    if len(img_paths) > 1
                    else None
                )
                mat = scene_builder.get_or_create_material(
                    md.imagery_key, img_paths, tag, save_path
                )
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
        victims = victims[:EVICT_MAX_PER_PASS]
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


def _timer_tick():
    """Module-level pump: a stable identity for bpy.app.timers (bound methods
    make is_registered unreliable across accesses) and a last-ditch exception
    guard so the timer survives anything _tick misses."""
    s = _instance
    if s is None:
        return None
    try:
        return s._tick()
    except BaseException:
        print("[cesium] timer crashed:\n" + traceback.format_exc())
        s.stats.last_error = "timer crash (see console)"
        return TICK_IDLE_S if s.running else None


def ensure_timer():
    """(Re)register the pump if it should be running but isn't. Called from
    start() and defensively from the panel draw: a bpy timer was observed
    vanishing in 4.5 with no traceback and running=True, so registration is
    treated as a state to converge on, not a one-time act."""
    s = _instance
    if s is None or not s.running:
        return
    if not bpy.app.timers.is_registered(_timer_tick):
        bpy.app.timers.register(_timer_tick, first_interval=TICK_BUSY_S)


def shutdown():
    global _instance
    if _instance is not None:
        _instance.stop()
        _instance = None
