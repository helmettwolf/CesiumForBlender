"""Blender datablock construction/destruction for streamed tiles.

BPY module, main thread only. All streamed objects live in the
"Cesium Terrain" collection. Datablock teardown order matters:
object -> mesh -> material (users==0) -> image (users==0), so nothing is
orphaned and memory stays bounded during long streaming sessions.
"""

from __future__ import annotations

import bpy
import numpy as np

COLLECTION_NAME = "Cesium Terrain"
MAT_PREFIX = "CesiumImg_"
GRAY_MAT_NAME = "CesiumUntextured"


def tile_object_name(key) -> str:
    return "T_%d_%d_%d" % tuple(key)


def material_name(img_key) -> str:
    return MAT_PREFIX + "%d_%d_%d" % tuple(img_key)


def ensure_collection() -> bpy.types.Collection:
    coll = bpy.data.collections.get(COLLECTION_NAME)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION_NAME)
    scene_children = bpy.context.scene.collection.children
    if scene_children.get(coll.name) is None:
        scene_children.link(coll)
    return coll


def get_or_create_material(img_key, img_path: str) -> bpy.types.Material:
    name = material_name(img_key)
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        bsdf.inputs["Roughness"].default_value = 1.0
        spec = bsdf.inputs.get("Specular IOR Level")
        if spec is not None:
            spec.default_value = 0.1
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.location = (-400, 300)
    tex.extension = "EXTEND"
    tex.interpolation = "Linear"
    img = bpy.data.images.load(img_path, check_existing=True)
    img.colorspace_settings.name = "sRGB"
    tex.image = img
    if bsdf is not None:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def get_or_create_gray_material() -> bpy.types.Material:
    mat = bpy.data.materials.get(GRAY_MAT_NAME)
    if mat is None:
        mat = bpy.data.materials.new(GRAY_MAT_NAME)
        mat.use_nodes = True
        bsdf = next(
            (n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None
        )
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (0.35, 0.34, 0.32, 1.0)
            bsdf.inputs["Roughness"].default_value = 1.0
    return mat


def build_tile_object(key, md, material: bpy.types.Material) -> bpy.types.Object:
    """Fast-path mesh build via foreach_set — no bmesh, no from_pydata."""
    coll = ensure_collection()
    name = tile_object_name(key)
    if bpy.data.objects.get(name) is not None:
        # leftover from a previous session/reload — replace, never .001-dupe
        destroy_tile_object(key)
    n = md.positions.shape[0]
    t = md.tri_count

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(n)
    mesh.vertices.foreach_set("co", np.ascontiguousarray(md.positions, dtype=np.float32).ravel())
    mesh.loops.add(3 * t)
    mesh.loops.foreach_set("vertex_index", md.loop_vertex_indices)
    mesh.polygons.add(t)
    mesh.polygons.foreach_set("loop_start", np.arange(0, 3 * t, 3, dtype=np.int32))
    mesh.update(calc_edges=True)
    mesh.validate()

    uv = mesh.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", md.loop_uvs)
    mesh.polygons.foreach_set("use_smooth", np.ones(t, dtype=np.bool_))
    mesh.materials.append(material)

    obj = bpy.data.objects.new(name, mesh)
    obj["cesium_key"] = list(key)
    coll.objects.link(obj)
    return obj


def set_tile_visible(key, visible: bool) -> bool:
    obj = bpy.data.objects.get(tile_object_name(key))
    if obj is None:
        return False
    hidden = not visible
    if obj.hide_viewport != hidden:
        obj.hide_viewport = hidden
        obj.hide_render = hidden
    return True


def destroy_tile_object(key):
    obj = bpy.data.objects.get(tile_object_name(key))
    if obj is None:
        return
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def sweep_unused_materials():
    """Remove tile materials (and their images) no mesh references anymore."""
    for mat in list(bpy.data.materials):
        if not mat.name.startswith(MAT_PREFIX) or mat.users > 0:
            continue
        img = None
        if mat.use_nodes:
            img = next(
                (n.image for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"),
                None,
            )
        bpy.data.materials.remove(mat)
        if img is not None and img.users == 0:
            bpy.data.images.remove(img)


def clear_all():
    coll = bpy.data.collections.get(COLLECTION_NAME)
    if coll is not None:
        for obj in list(coll.objects):
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    sweep_unused_materials()
    gray = bpy.data.materials.get(GRAY_MAT_NAME)
    if gray is not None and gray.users == 0:
        bpy.data.materials.remove(gray)


def tag_redraw_view3d():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def datablock_counts() -> dict:
    return {
        "objects": sum(
            1 for o in bpy.data.objects if o.name.startswith("T_")
        ),
        "meshes": sum(1 for m in bpy.data.meshes if m.name.startswith("T_")),
        "materials": sum(
            1 for m in bpy.data.materials if m.name.startswith(MAT_PREFIX)
        ),
        "images": len(bpy.data.images),
    }
