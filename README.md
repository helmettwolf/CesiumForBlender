# Cesium for Blender

Camera-driven LOD streaming of Cesium **quantized-mesh terrain** and **TMS
satellite imagery** into the Blender viewport — like CesiumJS, but pure Python
inside Blender. Move the viewport camera and tiles stream, refine, coarsen,
hide, and unload automatically.

![status](https://img.shields.io/badge/status-v0.1-blue) Blender 4.0+ · no
external dependencies (numpy + stdlib only).

## How it works

- **LOD selection** — every camera move triggers a quadtree traversal from the
  two EPSG:4326 root tiles. A tile refines while its screen-space error
  (`geometricError · viewportHeight · p11 / (2 · distance)`) exceeds the SSE
  threshold (default 16 px). Geometric errors come from the server's per-tile
  `metadata` extension, floored by an imagery-texel term so flat regions keep
  refining until the *texture* is sharp, not just the geometry.
- **Replacement refinement** — children replace their parent only when all
  four are built, so the ground never has holes while loading.
- **Culling** — view-frustum test plus ellipsoid horizon culling (tiles beyond
  the horizon are never loaded or rendered).
- **Threaded pipeline** — fetch → decode → ENU transform run on worker
  threads; only mesh/material creation touches `bpy`, budgeted to ≤3 builds
  or 8 ms per timer tick so the viewport never stutters.
- **Lifecycle** — tiles leaving the selection are hidden; beyond the tile
  budget (default 400 built) they are LRU-deleted with a strict teardown order
  (object → mesh → material → image), keeping datablock counts exactly equal
  to live tiles. Raw tiles are cached on disk (`%LOCALAPPDATA%\CesiumForBlender\cache`).
- **Precision** — vertices are stored in a local east-north-up frame anchored
  at a user-chosen lat/lon (ECEF magnitudes would destroy float32 viewport
  precision). +X east, +Y north, +Z up, meters.

## Install

```
python build_zip.py
```

Blender → Edit → Preferences → Add-ons → Install… → `dist/cesium_for_blender.zip`
→ enable **Cesium for Blender**.

## Use (N-panel → "Cesium" tab)

1. **Connect** — reads `layer.json` + `tilemapresource.xml` and seeds tile
   availability from the root tiles' metadata.
2. **Data Center** (or type a lat/lon and press **Go To**) — anchors the
   Blender origin there, frames the view, and raises the viewport clip range.
3. **Start Streaming** — fly around; LOD follows the viewport camera.
   The Status box shows visible/built/fetching counts and the deepest level.
4. **Stop** keeps the loaded tiles; **Clear Tiles** removes everything.

Diagnostic: **Load Single Tile** synchronously builds one tile (bypasses the
streamer) — useful to sanity-check a server.

### Atmosphere → Add Clouds

One click drops a procedural volumetric cloud layer over wherever you're
looking: the base height is found by ray-casting the terrain under the camera,
and the coverage/altitude/thickness/density knobs live in the redo panel (F9).
It also retunes EEVEE's froxel volumetrics for terrain scale (`volumetric_end`
60 km, 256 samples — the 100 m default would clip the whole layer). Renders in
under a second at 1080p. Notable EEVEE Next gotcha baked into the
implementation: volume shaders do not evaluate Texture Coordinate
Generated/Object outputs (constant → uniform fog); the density field must be
driven by `Geometry > Position` (world space) with the slab geometry baked
into node constants.

## Server expectations

Tested against a self-hosted stack (`layer.json` quantized-mesh-1.0, TMS
`tilemapresource.xml`, both EPSG:4326 / global-geodetic so terrain (z,x,y) ==
imagery (z,x,y)):

- Availability is read from the `metadata` (id 4) extension with the
  *subtree* convention (`available[i]` → level `z+1+i`), auto-detected with a
  fallback; `metadataAvailability`-style chunking (root: 1–10, z10 tiles:
  11–19) is handled transparently.
- Imagery declared as PNG but served as JPEG is detected by magic bytes.
- Absent tiles answered with 500/502 (instead of 404) are retried briefly,
  then marked dead; only *connection*-level failures trip the
  server-unreachable circuit breaker.

## Development

```
python tests/fetch_fixtures.py   # once: downloads real tiles into tests/fixtures
python -m pytest tests/          # pure-python suite, no Blender needed
```

`cesium_for_blender/core/` (except `camera/scene_builder/streamer`) imports
without `bpy` — the decoder, tiling math, availability index, providers, and
LOD traversal are all unit-tested outside Blender.

## Roadmap

- Skirts to hide T-junction cracks between LOD levels (edge-vertex lists are
  already decoded).
- Oct-encoded vertex normals → custom split normals (decoded, not yet applied).
- 3D Tiles (b3dm/glTF) streaming behind the same provider/LOD contract.
