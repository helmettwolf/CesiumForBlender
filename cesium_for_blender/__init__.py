"""Cesium for Blender: camera-driven LOD streaming of quantized-mesh terrain
with draped TMS imagery.

The core/ package is pure Python (numpy + stdlib) and importable without
Blender — the test suite depends on that. Everything bpy-flavored is guarded
behind the import check below.
"""

bl_info = {
    "name": "Cesium for Blender",
    "author": "Arun Yadav",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Cesium",
    "description": "Stream quantized-mesh terrain + TMS imagery with camera-driven LOD",
    "category": "Import-Export",
}

try:
    import bpy  # noqa: F401
    _HAS_BPY = True
except ImportError:
    _HAS_BPY = False

if _HAS_BPY:
    # Support addon reload (F8 / bpy.ops.script.reload): on re-execution the
    # previously imported names are still bound in the module namespace.
    if "properties" in locals():
        import importlib

        from .core import (cache, camera, lod, provider, quantized_mesh,
                           scene_builder, streamer, tiling, wgs84)

        streamer.shutdown()
        for _m in (wgs84, tiling, quantized_mesh, provider, cache, lod,
                   camera, scene_builder, streamer):
            importlib.reload(_m)
        importlib.reload(properties)  # noqa: F821
        importlib.reload(operators)   # noqa: F821
        importlib.reload(panel)       # noqa: F821

    from .ui import operators, panel, properties

    def register():
        properties.register()
        operators.register()
        panel.register()

    def unregister():
        from .core import streamer
        streamer.shutdown()
        panel.unregister()
        operators.unregister()
        properties.unregister()
