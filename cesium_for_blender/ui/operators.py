import os
import traceback

import bpy

from ..core import (camera, clouds, map_style, provider, quantized_mesh,
                    scene_builder, streamer, tiling)


def _ion_token(st) -> str:
    return st.ion_token.strip() or os.environ.get("CESIUM_ION_TOKEN", "").strip()


class CESIUM_OT_connect(bpy.types.Operator):
    bl_idname = "cesium.connect"
    bl_label = "Connect"
    bl_description = (
        "Connect terrain + imagery sources (server layer.json/"
        "tilemapresource.xml, or Cesium ion asset endpoints) and seed tile"
        " availability"
    )

    def execute(self, context):
        st = context.scene.cesium
        s = streamer.get()
        root = st.cache_dir or None

        if st.terrain_source == "ION" or st.imagery_source == "ION":
            token = _ion_token(st)
            if not token:
                self.report(
                    {"ERROR"},
                    "Cesium ion token missing — paste it in the panel or set"
                    " CESIUM_ION_TOKEN",
                )
                return {"CANCELLED"}

        if st.terrain_source == "ION":
            ion = provider.IonAsset(st.ion_terrain_asset, token)
            terrain = provider.TerrainProvider(
                "", provider.DiskCache(root, ion.namespace), ion=ion
            )
        else:
            terrain = provider.TerrainProvider(
                st.terrain_url,
                provider.DiskCache(root, provider.url_namespace(st.terrain_url)),
            )

        imagery = None
        if st.imagery_source == "ION":
            ion = provider.IonAsset(st.ion_imagery_asset, token)
            imagery = provider.ImageryProvider(
                "", provider.DiskCache(root, ion.namespace), ion=ion
            )
        elif st.imagery_source == "URL":
            imagery = provider.ImageryProvider(
                st.imagery_url,
                provider.DiskCache(root, provider.url_namespace(st.imagery_url)),
            )

        err = None
        for attempt in (1, 2):     # transient endpoint/metadata hiccups are
            try:                   # common enough to deserve one retry
                s.connect(terrain, imagery)
                s.stats.last_error = ""
                err = None
                break
            except Exception as e:
                err = e
        if err is not None:
            # a dead imagery source shouldn't block terrain — but say so
            # LOUDLY (status box), not just a toast: a silent downgrade
            # reads as "LOD is broken" when every new tile builds gray
            if imagery is not None:
                try:
                    s.connect(terrain, None)
                    s.stats.last_error = f"imagery off: {err}"
                    self.report(
                        {"WARNING"},
                        f"Imagery unavailable ({err}) — streaming terrain"
                        " only; press Connect to retry",
                    )
                    return {"FINISHED"}
                except Exception:
                    pass
            s.stats.status = "connect failed"
            self.report({"ERROR"}, f"Connect failed: {err}")
            return {"CANCELLED"}
        layer = s.terrain.layer or {}
        imagery_desc = (
            f"imagery '{s.imagery.title}' z0-{s.imagery.max_zoom}"
            if s.imagery is not None
            else "no imagery"
        )
        credit = ""
        if terrain.ion is not None and terrain.ion.attributions:
            credit = f" — {terrain.ion.attributions[0]}"
        self.report(
            {"INFO"},
            f"Connected: {layer.get('format', '?')} z{layer.get('minzoom', 0)}-"
            f"{layer.get('maxzoom', '?')}, {imagery_desc}{credit}",
        )
        return {"FINISHED"}


class CESIUM_OT_goto_data_center(bpy.types.Operator):
    bl_idname = "cesium.goto_data_center"
    bl_label = "Go To Data Center"
    bl_description = "Set the origin to the center of the densest data region"

    def execute(self, context):
        s = streamer.get()
        center = s.avail.deepest_coverage_center()
        if center is None:
            self.report({"ERROR"}, "No availability data — connect first")
            return {"CANCELLED"}
        st = context.scene.cesium
        st.origin_lat, st.origin_lon = center
        return bpy.ops.cesium.set_origin()


class CESIUM_OT_set_origin(bpy.types.Operator):
    bl_idname = "cesium.set_origin"
    bl_label = "Set Origin / Go To"
    bl_description = (
        "Anchor the Blender world origin at this lat/lon and frame the view "
        "there. Clears already-built tiles (their coordinates depend on the origin)"
    )

    def execute(self, context):
        st = context.scene.cesium
        s = streamer.get()
        was_running = s.running
        if s.tiles.tiles or was_running:
            s.clear()
        s.set_origin(st.origin_lat, st.origin_lon)
        camera.frame_origin()
        if was_running:
            s.start()
        self.report(
            {"INFO"}, f"Origin at lat={st.origin_lat:.5f} lon={st.origin_lon:.5f}"
        )
        return {"FINISHED"}


class CESIUM_OT_start(bpy.types.Operator):
    bl_idname = "cesium.start"
    bl_label = "Start Streaming"
    bl_description = "Stream terrain with camera-driven LOD"

    def execute(self, context):
        st = context.scene.cesium
        s = streamer.get()
        if not s.connected:
            self.report({"ERROR"}, "Connect first")
            return {"CANCELLED"}
        if s.frame is None:
            self.report({"ERROR"}, "Set an origin first")
            return {"CANCELLED"}
        s.sse_threshold = st.sse_threshold
        s.max_requests = st.max_requests
        s.tile_budget = st.tile_budget
        s.max_level = st.max_level
        try:
            s.start()
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class CESIUM_OT_stop(bpy.types.Operator):
    bl_idname = "cesium.stop"
    bl_label = "Stop"
    bl_description = "Stop streaming (built tiles stay in the scene)"

    def execute(self, context):
        streamer.get().stop()
        return {"FINISHED"}


class CESIUM_OT_clear(bpy.types.Operator):
    bl_idname = "cesium.clear"
    bl_label = "Clear Tiles"
    bl_description = "Stop streaming and remove every streamed tile from the scene"

    def execute(self, context):
        streamer.get().clear()
        return {"FINISHED"}


class CESIUM_OT_load_single_tile(bpy.types.Operator):
    """Diagnostic: synchronously fetch+build one tile (bypasses the streamer)."""

    bl_idname = "cesium.load_single_tile"
    bl_label = "Load Single Tile"
    bl_options = {"REGISTER", "UNDO"}

    zoom: bpy.props.IntProperty(name="Z", default=10, min=0, max=19)
    tile_x: bpy.props.IntProperty(name="X", default=0, min=0)
    tile_y: bpy.props.IntProperty(name="Y", default=0, min=0)
    at_origin: bpy.props.BoolProperty(
        name="Tile At Origin",
        description="Ignore X/Y and load the tile containing the current origin",
        default=True,
    )

    def execute(self, context):
        st = context.scene.cesium
        s = streamer.get()
        if not s.connected:
            self.report({"ERROR"}, "Connect first")
            return {"CANCELLED"}
        if s.frame is None:
            self.report({"ERROR"}, "Set an origin first")
            return {"CANCELLED"}
        key = (
            tiling.lonlat_to_tile(self.zoom, st.origin_lon, st.origin_lat)
            if self.at_origin
            else (self.zoom, self.tile_x, self.tile_y)
        )
        try:
            data = s.terrain.fetch_tile(*key)
            qm = quantized_mesh.decode(data)
            if qm.metadata and qm.metadata.get("available"):
                s.avail.ingest(key, qm.metadata["available"])
            ikey, img_paths = None, None
            scale, uo, vo = 1.0, 0.0, 0.0
            merc_rect = None
            if s.imagery is not None and s.imagery.scheme == "mercator":
                cover = tiling.mercator_cover(
                    *key, s.imagery.merc_max_zoom, s.imagery.merc_min_zoom
                )
                if cover is not None:
                    m, tx0, ty0, tx1, ty1 = cover
                    img_paths = [
                        s.imagery.fetch_tile(m, tx, ty)
                        for ty in range(ty0, ty1 + 1)
                        for tx in range(tx0, tx1 + 1)
                    ]
                    ikey = cover
                    merc_rect = tiling.mercator_cover_rect(cover)
            elif s.imagery is not None:
                ikey, scale, uo, vo = tiling.imagery_key_and_uv_transform(
                    *key, s.imagery.max_zoom
                )
                img_paths = [s.imagery.fetch_tile(*ikey)]
            md = quantized_mesh.tile_to_enu_mesh(
                qm, key, s.frame, scale, (uo, vo), imagery_key=ikey,
                mercator_rect=merc_rect,
            )
            scene_builder.destroy_tile_object(key)  # allow re-running
            if img_paths:
                tag = "M" if s.imagery.scheme == "mercator" else ""
                mat = scene_builder.get_or_create_material(ikey, img_paths, tag)
            else:
                mat = scene_builder.get_or_create_gray_material()
            obj = scene_builder.build_tile_object(key, md, mat)
        except Exception as e:
            traceback.print_exc()
            self.report({"ERROR"}, f"{key}: {e}")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Tile {key}: {md.positions.shape[0]} verts, {md.tri_count} tris, "
            f"imagery {ikey if ikey is not None else 'none'}",
        )
        return {"FINISHED"}


class CESIUM_OT_add_clouds(bpy.types.Operator):
    """Add (or re-center) a volumetric cloud layer above the active camera.
    Tweak coverage/altitude/density in the redo panel (F9) after clicking."""

    bl_idname = "cesium.add_clouds"
    bl_label = "Add Clouds"
    bl_options = {"REGISTER", "UNDO"}

    base_agl: bpy.props.FloatProperty(
        name="Base Above Ground (m)", default=2200.0, min=200.0, max=15000.0,
        description="Cloud base height above the terrain under the camera"
        " (found by ray-cast)",
    )
    thickness: bpy.props.FloatProperty(
        name="Thickness (m)", default=1600.0, min=200.0, max=8000.0,
    )
    coverage: bpy.props.FloatProperty(
        name="Coverage", default=0.45, min=0.05, max=0.95,
        description="Fraction of sky covered by cloud",
    )
    density: bpy.props.FloatProperty(
        name="Density", default=0.008, min=0.0005, max=0.05, precision=4,
        description="Optical density — lower is wispier",
    )
    size_km: bpy.props.FloatProperty(
        name="Extent (km)", default=240.0, min=20.0, max=1000.0,
    )
    self_shadow: bpy.props.BoolProperty(
        name="Self Shadow", default=True,
        description="Volumetric self-shadowing — more realistic, costs render time",
    )

    def execute(self, context):
        obj = clouds.add_or_update(
            context,
            base_agl=self.base_agl,
            thickness=self.thickness,
            coverage=self.coverage,
            density=self.density,
            size_km=self.size_km,
            self_shadow=self.self_shadow,
        )
        self.report(
            {"INFO"},
            f"Cloud layer at z={obj.location.z - self.thickness / 2:.0f} m, "
            f"{self.size_km:.0f} km extent",
        )
        return {"FINISHED"}


class CESIUM_OT_relief_style(bpy.types.Operator):
    """Toggle the relief-map terrain style (hypsometric tint + slope rock +
    studio mist look, after Owen Powell's GIS dioramas). Elevation range is
    auto-read from the streamed tiles; tweak in the redo panel (F9)"""

    bl_idname = "cesium.relief_style"
    bl_label = "Relief Map Style"
    bl_options = {"REGISTER", "UNDO"}

    contour_interval: bpy.props.FloatProperty(
        name="Contour Interval (m)", default=0.0, min=0.0, max=2000.0,
        description="Elevation contour lines; 0 disables them",
    )
    studio: bpy.props.BoolProperty(
        name="Studio Backdrop + Mist", default=True,
        description="Neutral gray world, soft sun, and a composited mist pass"
        " — the diorama look. Disable to keep the current sky and lighting",
    )

    def execute(self, context):
        if map_style.is_active():
            n = map_style.remove(context)
            self.report({"INFO"}, f"Relief style off — imagery restored on {n} tiles")
            return {"FINISHED"}
        s = streamer.get()
        heights = [
            (t.min_h, t.max_h)
            for t in s.tiles.tiles.values()
            if t.visible and t.min_h is not None
        ]
        if heights:
            min_e = min(h[0] for h in heights)
            max_e = max(h[1] for h in heights)
            if max_e - min_e < 200.0:
                max_e = min_e + 200.0
        else:
            min_e, max_e = 0.0, 4000.0
        n = map_style.apply(context, min_e, max_e, self.contour_interval, self.studio)
        self.report(
            {"INFO"},
            f"Relief style on {n} tiles, elevation ramp {min_e:.0f}-{max_e:.0f} m",
        )
        return {"FINISHED"}


class CESIUM_OT_remove_clouds(bpy.types.Operator):
    bl_idname = "cesium.remove_clouds"
    bl_label = "Remove Clouds"
    bl_description = "Delete the volumetric cloud layer"

    def execute(self, context):
        clouds.remove_clouds()
        return {"FINISHED"}


_CLASSES = (
    CESIUM_OT_connect,
    CESIUM_OT_goto_data_center,
    CESIUM_OT_set_origin,
    CESIUM_OT_start,
    CESIUM_OT_stop,
    CESIUM_OT_clear,
    CESIUM_OT_load_single_tile,
    CESIUM_OT_add_clouds,
    CESIUM_OT_remove_clouds,
    CESIUM_OT_relief_style,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
