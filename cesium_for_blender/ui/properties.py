import bpy


class CesiumSettings(bpy.types.PropertyGroup):
    terrain_url: bpy.props.StringProperty(
        name="Terrain URL",
        description="Quantized-mesh endpoint (serves layer.json)",
        default="http://tile-server/api/terrain",
    )
    imagery_url: bpy.props.StringProperty(
        name="Imagery URL",
        description="TMS endpoint (serves tilemapresource.xml)",
        default="http://tile-server/api/tiles",
    )
    origin_lat: bpy.props.FloatProperty(
        name="Origin Lat", default=0.0, min=-90.0, max=90.0, precision=6,
        description="Geodetic latitude of the Blender world origin",
    )
    origin_lon: bpy.props.FloatProperty(
        name="Origin Lon", default=0.0, min=-180.0, max=180.0, precision=6,
        description="Geodetic longitude of the Blender world origin",
    )
    sse_threshold: bpy.props.FloatProperty(
        name="SSE Threshold (px)", default=16.0, min=1.0, max=128.0,
        description="Refine tiles while their screen-space error exceeds this"
        " many pixels — lower is sharper and heavier",
    )
    max_requests: bpy.props.IntProperty(
        name="Max Requests", default=6, min=1, max=16,
        description="Concurrent tile downloads",
    )
    tile_budget: bpy.props.IntProperty(
        name="Tile Budget", default=400, min=50, max=5000,
        description="Built tiles kept in memory; hidden tiles beyond this are"
        " deleted (LRU)",
    )
    max_level: bpy.props.IntProperty(
        name="Max Level", default=19, min=0, max=19,
        description="Never refine terrain beyond this zoom level",
    )
    cache_dir: bpy.props.StringProperty(
        name="Cache", subtype="DIR_PATH", default="",
        description="Disk cache for downloaded tiles (empty = LOCALAPPDATA)",
    )


def register():
    bpy.utils.register_class(CesiumSettings)
    bpy.types.Scene.cesium = bpy.props.PointerProperty(type=CesiumSettings)


def unregister():
    del bpy.types.Scene.cesium
    bpy.utils.unregister_class(CesiumSettings)
