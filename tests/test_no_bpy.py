"""The core package must be importable without Blender."""

import sys


def test_pure_modules_do_not_import_bpy():
    for name in ("wgs84", "tiling", "quantized_mesh", "provider", "cache", "lod"):
        __import__(f"cesium_for_blender.core.{name}")
    assert "bpy" not in sys.modules
