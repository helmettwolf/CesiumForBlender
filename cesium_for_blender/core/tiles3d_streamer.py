"""3D Tiles streaming orchestrator: same contract as the terrain streamer.

Threading:
- Workers fetch content, unwrap b3dm/cmpt to GLB files on disk, and compute
  each GLB's ENU matrix — pure code only.
- The bpy timer (main thread) runs SSE selection, grafts external tilesets,
  imports GLBs via Blender's glTF importer (budgeted: ONE import per tick —
  imports are the expensive step), applies visibility, and evicts LRU.

The ENU frame is SHARED with the terrain streamer, so buildings land on the
streamed ground. Changing the origin resets built content (matrices and
AABBs are frame-relative).
"""

from __future__ import annotations

import enum
import os
import queue
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import bpy
import numpy as np
from mathutils import Matrix

from . import camera, lod, provider, scene_builder, tiles3d

COLLECTION_NAME = "Cesium 3D Tiles"
TICK_BUSY_S = 0.05
TICK_IDLE_S = 0.15
SELECTION_MAX_AGE_S = 0.5
RETRY_BACKOFF_S = (1.0, 4.0, 15.0)
IMPORTS_PER_TICK = 1
EVICT_PERIOD_S = 2.0
# Never evict content wanted this recently: a small camera move culls tiles
# for a moment, and deleting them means a full refetch+reimport the instant
# the camera settles — the "everything reloads when I move" experience. The
# budget is SOFT under pressure; stability beats a hard memory line.
EVICT_GRACE_S = 10.0
EVICT_MAX_PER_PASS = 16


class NState(enum.Enum):
    QUEUED = "queued"
    FETCHING = "fetching"
    READY = "ready"          # GLB files on disk, awaiting main-thread import
    BUILT = "built"
    FAILED = "failed"
    DEAD = "dead"


class NodeState:
    __slots__ = (
        "state", "retries", "next_retry", "glbs", "obj_names", "data_names",
        "visible", "last_wanted", "ready_at",
    )

    def __init__(self):
        self.state = NState.QUEUED
        self.retries = 0
        self.next_retry = 0.0
        self.glbs: list = []          # [(glb_path, enu_matrix 4x4 list)]
        self.obj_names: list = []
        self.data_names: list = []    # (meshes, materials, images) created
        self.visible = False
        self.last_wanted = 0.0
        self.ready_at = 0.0


class Stats3D:
    def __init__(self):
        self.status = "disconnected"
        self.visible = 0
        self.built = 0
        self.fetching = 0
        self.queued = 0
        self.dead = 0
        self.last_error = ""


class Tiles3DStreamer:
    def __init__(self):
        self.provider: provider.Tiles3DProvider | None = None
        self.tileset = tiles3d.Tileset()
        self.root: tiles3d.TileNode | None = None
        self.states: dict[int, NodeState] = {}
        self.frame = None
        self.stats = Stats3D()
        self.connected = False
        self.running = False
        self.generation = 0
        self.executor: ThreadPoolExecutor | None = None
        self.results: queue.Queue = queue.Queue()
        self.inflight = 0
        self.max_requests = 8
        self.sse_threshold = 16.0
        self.detail_falloff = 600.0   # dynamic-SSE distance (m)
        self.content_budget = 150
        self._current = tiles3d.Selection3D()
        self._render_ids: set[int] = set()
        self._last_sig = None
        self._last_selection_t = 0.0
        self._last_evict_t = 0.0

    # ------------------------------------------------------------- lifecycle

    def connect(self, prov: provider.Tiles3DProvider, frame):
        if frame is None:
            raise RuntimeError("set an origin first (shared with terrain)")
        root_url = prov.connect()
        doc = prov.fetch_json(root_url)
        self.provider = prov
        self.frame = frame
        self.tileset = tiles3d.Tileset()
        self.root = self.tileset.parse(doc, root_url)
        self.states = {}
        self.connected = True
        self.stats.status = "connected"
        return self.root

    def start(self):
        if self.running:
            ensure_timer()
            return
        if not self.connected or self.frame is None:
            raise RuntimeError("connect 3D Tiles first")
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_requests, thread_name_prefix="cesium3d"
        )
        self.running = True
        self.stats.status = "streaming"
        ensure_timer()

    def stop(self):
        self.running = False
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
        self.inflight = 0
        for st in self.states.values():
            if st.state == NState.FETCHING:
                st.state = NState.QUEUED
        if self.stats.status == "streaming":
            self.stats.status = "stopped"

    def clear(self):
        self.stop()
        for st in self.states.values():
            self._destroy(st)
        coll = bpy.data.collections.get(COLLECTION_NAME)
        if coll is not None:
            leftovers = []
            for obj in list(coll.objects):
                leftovers.append(obj)
                if obj.data is not None:
                    leftovers.append(obj.data)
            if leftovers:
                try:
                    bpy.data.batch_remove(leftovers)
                except ReferenceError:
                    pass
        self.states = {}
        self.results = queue.Queue()
        self._current = tiles3d.Selection3D()
        self._render_ids = set()
        self._last_sig = None
        self.generation += 1

    def reset_for_origin(self, frame):
        """Origin moved: matrices and AABBs are frame-relative — drop built
        content, keep the parsed tree, re-derive AABBs lazily."""
        was_running = self.running
        self.clear()
        self.frame = frame
        for node in self.tileset.nodes.values():
            node.aabb = None
        if was_running and self.connected:
            self.start()

    # ------------------------------------------------------------ worker side

    def _state(self, node) -> NodeState:
        st = self.states.get(node.id)
        if st is None:
            st = NodeState()
            self.states[node.id] = st
        return st

    def _worker_load(self, node_id, url, is_tileset, world, gen):
        try:
            if is_tileset:
                doc = self.provider.fetch_json(url)
                self.results.put(("tileset", node_id, gen, doc, url))
                return
            data = self.provider.fetch_content(url)
            parts = tiles3d.unwrap_content(data)
            enu = tiles3d.enu4(self.frame)
            glbs = []
            for i, (glb, rtc) in enumerate(parts):
                p = self.provider.glb_path(url, i)
                try:
                    with open(p, "rb") as f:
                        pass
                except OSError:
                    self.provider.cache.put_blob(
                        "glb", f"{self.provider._hash(url)}_{i}", ".glb", glb
                    )
                m = np.asarray(world, dtype=np.float64)
                if rtc is not None:
                    t = np.eye(4)
                    t[:3, 3] = rtc
                    m = m @ t
                glbs.append((p, (enu @ m).tolist()))
            self.results.put(("content", node_id, gen, glbs, None))
        except (provider.TileNotFound, tiles3d.Tiles3DError) as e:
            self.results.put(("dead", node_id, gen, str(e), None))
        except provider.TileFetchError as e:
            self.results.put(("retry", node_id, gen, str(e), None))
        except Exception:
            self.results.put(("dead", node_id, gen, traceback.format_exc(), None))

    # ------------------------------------------------------------- main loop

    def _tick(self):
        if not self.running:
            return None
        try:
            self._tick_inner()
        except Exception:
            print("[cesium3d] tick error:\n" + traceback.format_exc())
            self.stats.last_error = "tick error (see console)"
        busy = self.inflight > 0 or not self.results.empty() or any(
            st.state == NState.READY for st in self.states.values()
        )
        return TICK_BUSY_S if busy else TICK_IDLE_S

    def _tick_inner(self):
        now = time.monotonic()
        cam = camera.get_camera_state()

        changed = self._drain_results(now)
        imported = self._import_ready(now)

        need_selection = cam is not None and (
            cam.signature != self._last_sig
            or now - self._last_selection_t > SELECTION_MAX_AGE_S
            or changed
            or imported
        )
        if need_selection and self.root is not None:
            def is_ready(node):
                if node.content_uri and tiles3d.is_tileset_uri(node.content_uri):
                    return node.grafted
                st = self.states.get(node.id)
                return st is not None and st.state == NState.BUILT

            self._current = tiles3d.select(
                self.root, cam, self.frame, self.sse_threshold, is_ready,
                max_dist=lod.horizon_limit(cam.pos, self.frame),
                detail_falloff=self.detail_falloff,
            )
            self._render_ids = {
                n.id for n in self._current.render
                if not (n.content_uri and tiles3d.is_tileset_uri(n.content_uri))
            }
            for _p, n in self._current.load:
                self._state(n).last_wanted = now
            # everything the traversal touched counts as wanted — eviction
            # grace is measured against this
            for nid in self._current.keep:
                st = self.states.get(nid)
                if st is not None:
                    st.last_wanted = now
            self._last_sig = cam.signature
            self._last_selection_t = now
            self._apply_visibility()

        self._submit_loads(now)
        if now - self._last_evict_t > EVICT_PERIOD_S:
            self._last_evict_t = now
            self._evict(now)
        self._update_stats()

    def _drain_results(self, now) -> bool:
        changed = False
        while True:
            try:
                kind, node_id, gen, payload, extra = self.results.get_nowait()
            except queue.Empty:
                break
            self.inflight = max(0, self.inflight - 1)
            if gen != self.generation:
                continue
            node = self.tileset.nodes.get(node_id)
            st = self.states.get(node_id)
            if node is None or st is None:
                continue
            if kind == "tileset":
                try:
                    self.tileset.graft(node, payload, extra)
                    if node.content_uri and not tiles3d.is_tileset_uri(
                        node.content_uri
                    ):
                        # the graft rewrote this node's content to the
                        # external root's DRAWABLE uri — fetch it, or the
                        # node renders as a hole (built, zero objects)
                        st.state = NState.QUEUED
                        st.retries = 0
                    else:
                        st.state = NState.BUILT   # nothing drawable itself
                except tiles3d.Tiles3DError as e:
                    st.state = NState.DEAD
                    print(f"[cesium3d] graft failed for node {node_id}: {e}")
                changed = True
            elif kind == "content":
                st.glbs = payload
                st.state = NState.READY
                st.ready_at = now
                changed = True
            elif kind == "dead":
                st.state = NState.DEAD
                print(f"[cesium3d] node {node_id} dead: {str(payload)[:200]}")
                changed = True
            elif kind == "retry":
                st.retries += 1
                if st.retries > len(RETRY_BACKOFF_S):
                    st.state = NState.DEAD
                else:
                    st.state = NState.FAILED
                    st.next_retry = now + RETRY_BACKOFF_S[st.retries - 1]
                changed = True
        return changed

    def _import_ready(self, now) -> bool:
        done = 0
        for node_id, st in self.states.items():
            if done >= IMPORTS_PER_TICK:
                break
            if st.state != NState.READY:
                continue
            try:
                self._build(node_id, st)
            except Exception:
                # missing glb on disk / importer CANCELLED under some
                # contexts: refetch + rebuild with backoff, dead only after
                # repeated failures — a silent 0-object BUILT is a hole
                st.glbs = []
                st.retries += 1
                if st.retries > len(RETRY_BACKOFF_S):
                    st.state = NState.DEAD
                else:
                    st.state = NState.FAILED
                    st.next_retry = now + RETRY_BACKOFF_S[st.retries - 1]
                print(f"[cesium3d] build failed for node {node_id} "
                      f"(retry {st.retries}):\n" + traceback.format_exc())
            done += 1
        if done:
            scene_builder.tag_redraw_view3d()
        return bool(done)

    def _build(self, node_id, st: NodeState):
        coll = bpy.data.collections.get(COLLECTION_NAME)
        if coll is None:
            coll = bpy.data.collections.new(COLLECTION_NAME)
        if bpy.context.scene.collection.children.get(coll.name) is None:
            bpy.context.scene.collection.children.link(coll)

        obj_names: list[str] = []
        meshes: set[str] = set()
        mats: set[str] = set()
        imgs: set[str] = set()
        for path, mat4 in st.glbs:
            if not os.path.isfile(path):
                raise RuntimeError(f"glb not on disk: {path}")
            before_o = set(bpy.data.objects.keys())
            before_m = set(bpy.data.meshes.keys())
            before_ma = set(bpy.data.materials.keys())
            before_i = set(bpy.data.images.keys())
            result = bpy.ops.import_scene.gltf(filepath=path)
            if "FINISHED" not in result:
                raise RuntimeError(f"gltf import returned {result} for {path}")
            new_objs = [
                bpy.data.objects[n]
                for n in bpy.data.objects.keys()
                if n not in before_o
            ]
            if not new_objs:
                raise RuntimeError(f"gltf import produced no objects: {path}")
            meshes |= set(bpy.data.meshes.keys()) - before_m
            mats |= set(bpy.data.materials.keys()) - before_ma
            imgs |= set(bpy.data.images.keys()) - before_i
            m = Matrix(mat4)
            for obj in new_objs:
                if obj.parent is None:
                    obj.matrix_world = m @ obj.matrix_world
                for c in obj.users_collection:
                    if c is not coll:
                        c.objects.unlink(obj)
                if coll.objects.get(obj.name) is None:
                    coll.objects.link(obj)
                obj.hide_viewport = True
                obj.hide_render = True
                obj_names.append(obj.name)
        st.obj_names = obj_names
        st.data_names = [sorted(meshes), sorted(mats), sorted(imgs)]
        st.glbs = []
        st.visible = False
        st.state = NState.BUILT

    def _apply_visibility(self):
        for node_id in self._render_ids:
            st = self.states.get(node_id)
            if st is not None and st.state == NState.BUILT and not st.visible:
                self._set_visible(st, True)
        for node_id, st in self.states.items():
            if st.state == NState.BUILT and st.visible and node_id not in self._render_ids:
                self._set_visible(st, False)

    def _set_visible(self, st: NodeState, visible: bool):
        hidden = not visible
        for name in st.obj_names:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.hide_viewport = hidden
                obj.hide_render = hidden
        st.visible = visible

    def _submit_loads(self, now):
        if self.executor is None:
            return
        for _prio, node in self._current.load:
            if self.inflight >= self.max_requests:
                break
            st = self._state(node)
            if st.state == NState.QUEUED or (
                st.state == NState.FAILED and st.next_retry <= now
            ):
                st.state = NState.FETCHING
                self.inflight += 1
                is_ts = tiles3d.is_tileset_uri(node.content_uri)
                url = tiles3d.resolve_uri(node.base_url, node.content_uri)
                self.executor.submit(
                    self._worker_load, node.id, url, is_ts,
                    node.world.tolist(), self.generation,
                )

    def _evict(self, now: float):
        built = [
            (node_id, st) for node_id, st in self.states.items()
            if st.state == NState.BUILT and st.obj_names
        ]
        if len(built) <= self.content_budget:
            return
        keep = self._current.keep
        victims = [
            (node_id, st) for node_id, st in built
            if not st.visible
            and node_id not in keep
            and now - st.last_wanted > EVICT_GRACE_S
        ]
        victims.sort(key=lambda kv: kv[1].last_wanted)
        n = min(len(built) - self.content_budget, EVICT_MAX_PER_PASS)
        for node_id, st in victims[:n]:
            self._destroy(st)
            st.state = NState.QUEUED

    def _destroy(self, st: NodeState):
        """Single C-side batch removal — looping .remove() over dozens of
        datablocks while the viewport draws them is both slow and a
        crash-prone path (observed segfault clearing a loaded scene)."""
        ids = []
        for name in st.obj_names:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                ids.append(obj)
        if st.data_names:
            meshes, mats, imgs = st.data_names
            ids += [m for n in meshes if (m := bpy.data.meshes.get(n))]
            ids += [m for n in mats if (m := bpy.data.materials.get(n))]
            ids += [i for n in imgs if (i := bpy.data.images.get(n))]
        if ids:
            try:
                bpy.data.batch_remove(ids)
            except ReferenceError:
                pass
        st.obj_names = []
        st.data_names = []
        st.visible = False

    def _update_stats(self):
        s = self.stats
        s.built = sum(
            1 for st in self.states.values()
            if st.state == NState.BUILT and st.obj_names
        )
        s.fetching = sum(
            1 for st in self.states.values() if st.state == NState.FETCHING
        )
        s.queued = sum(
            1 for st in self.states.values() if st.state == NState.QUEUED
        )
        s.dead = sum(1 for st in self.states.values() if st.state == NState.DEAD)
        s.visible = sum(1 for st in self.states.values() if st.visible)


_instance: Tiles3DStreamer | None = None


def get() -> Tiles3DStreamer:
    global _instance
    if _instance is None:
        _instance = Tiles3DStreamer()
    return _instance


def _timer_tick():
    s = _instance
    if s is None:
        return None
    try:
        return s._tick()
    except BaseException:
        print("[cesium3d] timer crashed:\n" + traceback.format_exc())
        s.stats.last_error = "timer crash (see console)"
        return TICK_IDLE_S if s.running else None


def ensure_timer():
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
