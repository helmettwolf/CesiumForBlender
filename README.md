# Cesium for Blender

Camera-driven LOD streaming of Cesium **quantized-mesh terrain**, **TMS/Bing
satellite imagery**, and **3D Tiles** (OSM Buildings, Google Photorealistic
3D Tiles) into the Blender viewport — like CesiumJS, but pure Python inside
Blender. Move the viewport camera and tiles stream, refine, coarsen, hide,
and unload automatically.

![status](https://img.shields.io/badge/status-v0.1-blue) Blender 4.0+ · no
external dependencies (numpy + stdlib only).

## How it works

- **LOD selection** — every camera move triggers a quadtree traversal from the
  two EPSG:4326 root tiles. A tile refines while its screen-space error
  (`geometricError · viewportHeight · p11 / (2 · distance)`) exceeds the SSE
  threshold (default 16 px). Geometric errors come from the server's per-tile
  `metadata` extension, floored by an imagery-texel term so flat regions keep
  refining until the *texture* is sharp, not just the geometry.
- **Upsampled (synthetic) tiles** — past the terrain dataset's resolution
  limit (e.g. Cesium World Terrain stops at z13 over much of Asia),
  subdivision continues CesiumJS-style: missing children are synthesized by
  resampling the nearest REAL ancestor's TIN on a regular grid (barycentric,
  in quantized uv space — sibling edges sample identical points, so no new
  cracks), and each smaller tile drapes its own sharper imagery. Refinement
  stops when the *imagery* is exhausted, not the terrain.
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

1. **Sources** — pick terrain and imagery sources independently:
   - **Server URL** — self-hosted `layer.json` / `tilemapresource.xml` stack.
   - **Cesium ion** — asset ID + access token
     ([cesium.com/ion/tokens](https://cesium.com/ion/tokens)); terrain asset 1
     is Cesium World Terrain. The asset endpoint is resolved via ion's REST
     API and tiles are fetched with the returned short-lived Bearer token
     (auto-refreshed on 401). The token can also come from a
     `CESIUM_ION_TOKEN` environment variable instead of the panel field.
     Imagery assets may be ion-hosted TMS (geodetic **or** web-mercator,
     e.g. 3954 = Sentinel-2) or **Bing** external assets (2 = Aerial,
     3 = Aerial with labels, 4 = Road): ion hands over a Bing key, tiles are
     fetched by quadkey from Bing's REST metadata template. Mercator imagery
     is draped by covering each terrain tile with ONE mercator column and at
     most TWO stacked rows at the deepest level that fits (columns align at
     mercator z+1; rows are nonlinear in latitude, so a single-tile cover
     would depend on row alignment and let adjacent terrain tiles land many
     imagery levels apart). Row pairs are stitched into one texture at build
     time and UVs are reprojected per vertex — u linear in longitude, v
     through the mercator latitude function — keeping neighbors within one
     imagery level. LEAF tiles (terrain at its dataset limit, e.g. CWT
     stopping at z13 over much of Asia) drape an overzoomed grid two levels
     deeper (up to 4x8 tiles stitched, ~8x sharper) so imagery keeps
     improving past the mesh resolution. Other external types (Google
     Earth Enterprise, ArcGIS) are rejected with a clear message.
   - **None** (imagery) — terrain only, flat gray; pairs well with the
     Relief Map Style.
2. **Connect** — reads `layer.json` + `tilemapresource.xml` (or the ion
   endpoints) and seeds tile availability from `layer.json`'s `available`
   array plus the root tiles' metadata. If imagery fails to connect,
   streaming continues terrain-only with a warning.
3. **Data Center** (or type a lat/lon and press **Go To**) — anchors the
   Blender origin there, frames the view, and raises the viewport clip range.
4. **Start Streaming** — fly around; LOD follows the viewport camera.
   The Status box shows visible/built/fetching counts and the deepest level.
5. **Stop** keeps the loaded tiles; **Clear Tiles** removes everything.

Diagnostic: **Load Single Tile** synchronously builds one tile (bypasses the
streamer) — useful to sanity-check a server.

### 3D Tiles

The **3D Tiles** box streams a 3D Tiles asset into the same ENU world as the
terrain (buildings land on the streamed ground), with its own camera-driven
SSE refinement. Sources: an ion asset ID — 96188 = OSM Buildings, 2275207 =
Google Photorealistic 3D Tiles — or a direct `tileset.json` URL. For
Photorealistic-style datasets that carry their own ground, set the terrain
source to **None** and just anchor an origin.

- **Format coverage** — explicit 1.0/1.1 trees with lazy external-tileset
  grafting, REPLACE and ADD refinement, region/box/sphere bounding volumes
  (with the same ellipsoid-bulge padding horizon culling needs), per-tile
  column-major transforms, accumulated. Content: `b3dm` (feature-table
  `RTC_CENTER` and glTF `CESIUM_RTC` handled), direct `glb`, recursive
  `cmpt`; `pnts`/`i3dm` are skipped. Externally-hosted ion assets (Google)
  resolve to the provider's root URL + API key, and query parameters are
  inherited from each tileset's URL down to its children — that is how
  Google's per-session keys thread through.
- **Import path** — workers fetch and unwrap content to GLB files in the
  disk cache; Blender's own glTF importer builds meshes/materials on the
  main thread (one import per tick), so Draco and PBR come free. Builds are
  validated (missing or cancelled imports retry with backoff).
- **Detail Falloff** (dynamic screen-space error) — the refinement
  threshold grows with distance, so full detail concentrates within roughly
  the falloff radius (default 600 m) and far areas settle at coarse levels.
  Without it, a street-level view of a dense city demands maximum detail to
  the horizon and streaming never converges.
- **Stability** — a parent is only swapped out when its entire visible
  replacement subtree is ready (through contentless intermediate nodes), so
  moving the camera never leaves holes; eviction runs on a 2 s cadence with
  a 10 s wanted-recently grace and a soft LRU budget, so a small camera move
  upgrades meshes in place instead of reloading the area. Teardown is a
  single `batch_remove` per node (object → mesh → material → image).

### Atmosphere & Style → Relief Map Style

One click swaps the satellite imagery for an Owen Powell-style relief-model
look ([his BlenderNation writeup](https://www.blendernation.com/2016/09/03/owen-powell-maps-terrain-models/)):
hypsometric height tint (valley green → buff → rock → pale summits, range
auto-calibrated from the streamed tiles' quantized-mesh headers),
slope-driven bare-rock blending, matte clay material, a neutral studio
backdrop, soft sun, and a composited mist pass for the diorama fade.
Elevation is curvature-corrected (`z + d²/2R`) so tints stay level far from
the ENU origin. Optional contour lines via the F9 panel. Click again to
restore imagery and the previous sky. Streaming continues to work — new
tiles pick up whichever style is active.

### Atmosphere & Style → Add Clouds

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
- The disk cache is namespaced per source (`ion-<assetid>` or a slug of the
  server URL) so tile coordinates from different datasets never collide.
- Gzip-encoded terrain tiles (ion's CDN) are detected by magic bytes and
  decompressed transparently.

## Development

```
python tests/fetch_fixtures.py   # once: downloads real tiles into tests/fixtures
python -m pytest tests/          # pure-python suite, no Blender needed
```

`cesium_for_blender/core/` (except `camera/scene_builder/streamer`) imports
without `bpy` — the decoder, tiling math, availability index, providers, and
LOD traversal are all unit-tested outside Blender.

## Roadmap

3D Tiles streaming shipped (see above) — remaining gaps and ideas:

- 3D Tiles implicit tiling (1.1 subtrees) and point-cloud content
  (`pnts`/`i3dm`) — the b3dm/glb/cmpt pipeline incl. Google Photorealistic
  is implemented.
- Skirts to hide T-junction cracks between LOD levels (edge-vertex lists are
  already decoded).
- Oct-encoded vertex normals → custom split normals (decoded, not yet applied).
