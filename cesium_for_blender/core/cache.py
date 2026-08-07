"""Tile state machine + in-memory tile cache. Pure module, no bpy.

Only the main thread mutates Tile.state (workers communicate through the
streamer's results queue), so no per-tile locking is needed.
"""

from __future__ import annotations

import enum
import time

import numpy as np


class TileState(enum.Enum):
    QUEUED = "queued"        # known, not yet submitted
    FETCHING = "fetching"    # submitted to the worker pool
    DECODED = "decoded"      # mesh data ready, awaiting main-thread build
    BUILT = "built"          # Blender object exists (visible flag is separate)
    FAILED = "failed"        # retryable failure, backoff pending
    DEAD = "dead"            # 404 or malformed — never retried this session


# geometric error of a level-0 geodetic tile (CesiumJS heuristic for a
# two-root-tile scheme); halved per level until tile metadata overrides it.
DEFAULT_GE0 = 77067.0


def default_geometric_error(z: int) -> float:
    return DEFAULT_GE0 / (1 << z)


class Tile:
    __slots__ = (
        "key", "state", "visible", "geometric_error", "ge_floor", "aabb",
        "mesh_data", "decoded_at", "last_wanted", "retries", "next_retry",
    )

    def __init__(self, key: tuple):
        self.key = key
        self.state = TileState.QUEUED
        self.visible = False
        self.geometric_error = default_geometric_error(key[0])
        self.ge_floor: float | None = None   # imagery-texel floor, lazy (lod.py)
        self.aabb: tuple[np.ndarray, np.ndarray] | None = None
        self.mesh_data = None          # TileMeshData between DECODED and BUILT
        self.decoded_at = 0.0
        self.last_wanted = 0.0
        self.retries = 0
        self.next_retry = 0.0


class TileCache:
    def __init__(self):
        self.tiles: dict[tuple, Tile] = {}

    def get(self, key: tuple) -> Tile | None:
        return self.tiles.get(key)

    def get_or_create(self, key: tuple) -> Tile:
        t = self.tiles.get(key)
        if t is None:
            t = Tile(key)
            self.tiles[key] = t
        return t

    def state_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.tiles.values():
            counts[t.state.value] = counts.get(t.state.value, 0) + 1
        return counts

    def evictable(self, keep: set, budget: int) -> list[Tile]:
        """BUILT, hidden, unwanted tiles beyond `budget`, LRU-first."""
        built = [t for t in self.tiles.values() if t.state == TileState.BUILT]
        if len(built) <= budget:
            return []
        candidates = [
            t for t in built if not t.visible and t.key not in keep
        ]
        candidates.sort(key=lambda t: t.last_wanted)
        return candidates[: max(0, len(built) - budget)]

    def drop_stale_mesh_data(self, older_than_s: float = 60.0) -> int:
        """Free decoded-but-never-built payloads (camera moved away before the
        build budget got to them)."""
        now = time.monotonic()
        n = 0
        for t in self.tiles.values():
            if (
                t.state == TileState.DECODED
                and t.mesh_data is not None
                and now - t.decoded_at > older_than_s
            ):
                t.mesh_data = None
                t.state = TileState.QUEUED
                n += 1
        return n
