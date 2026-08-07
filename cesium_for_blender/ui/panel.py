import bpy

from ..core import streamer


class CESIUM_PT_main(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Cesium"
    bl_label = "Cesium Terrain"

    def draw(self, context):
        st = context.scene.cesium
        s = streamer.get()
        layout = self.layout

        box = layout.box()
        box.label(text="Server", icon="URL")
        box.prop(st, "terrain_url", text="Terrain")
        box.prop(st, "imagery_url", text="Imagery")
        row = box.row()
        row.operator("cesium.connect", icon="LINKED")
        row.label(text=s.stats.status)

        box = layout.box()
        box.label(text="Origin", icon="WORLD")
        row = box.row(align=True)
        row.prop(st, "origin_lat", text="Lat")
        row.prop(st, "origin_lon", text="Lon")
        row = box.row(align=True)
        row.operator("cesium.set_origin", text="Go To", icon="VIEWZOOM")
        row.operator("cesium.goto_data_center", text="Data Center", icon="PIVOT_BOUNDBOX")

        box = layout.box()
        box.label(text="Streaming", icon="PLAY")
        box.prop(st, "sse_threshold")
        row = box.row(align=True)
        row.prop(st, "max_requests", text="Requests")
        row.prop(st, "tile_budget", text="Budget")
        box.prop(st, "max_level")
        row = box.row(align=True)
        if s.running:
            row.operator("cesium.stop", icon="PAUSE")
        else:
            row.operator("cesium.start", icon="PLAY")
        row.operator("cesium.clear", icon="TRASH")

        box = layout.box()
        box.label(text="Atmosphere & Style", icon="WORLD_DATA")
        row = box.row(align=True)
        row.operator("cesium.add_clouds", icon="OUTLINER_OB_VOLUME")
        row.operator("cesium.remove_clouds", text="", icon="X")
        from ..core import map_style
        box.operator(
            "cesium.relief_style",
            icon="SHADING_RENDERED",
            text="Relief Map Style" if not map_style.is_active() else "Back To Satellite",
            depress=map_style.is_active(),
        )

        box = layout.box()
        box.label(text="Status", icon="INFO")
        col = box.column(align=True)
        col.label(text=f"Visible: {s.stats.visible}   Built: {s.stats.built}")
        col.label(
            text=f"Fetching: {s.stats.fetching}   Queued: {s.stats.queued}"
        )
        col.label(text=f"Failed: {s.stats.failed}   Dead: {s.stats.dead}")
        col.label(text=f"Deepest visible level: {s.stats.render_level_max}")
        if s.stats.last_error:
            col.label(text=s.stats.last_error, icon="ERROR")

        box = layout.box()
        box.label(text="Diagnostics", icon="TOOL_SETTINGS")
        box.prop(st, "cache_dir")
        box.operator("cesium.load_single_tile", icon="MESH_GRID")


def register():
    bpy.utils.register_class(CESIUM_PT_main)


def unregister():
    bpy.utils.unregister_class(CESIUM_PT_main)
