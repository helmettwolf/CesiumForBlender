"""Blender datablock construction/destruction for streamed tiles.

BPY module, main thread only. All streamed objects live in the
"Cesium Terrain" collection. Datablock teardown order matters:
object -> mesh -> material (users==0) -> image (users==0), so nothing is
orphaned and memory stays bounded during long streaming sessions.
"""

from __future__ import annotations

import os

import bpy
import numpy as np

COLLECTION_NAME = "Cesium Terrain"
MAT_PREFIX = "CesiumImg_"
GRAY_MAT_NAME = "CesiumUntextured"


def tile_object_name(key) -> str:
    return "T_%d_%d_%d" % tuple(key)


def material_name(img_key, tag: str = "") -> str:
    """tag ('M' for mercator) keeps schemes with overlapping (z,x,y) numbers
    from sharing a material name. Mercator cover keys are 4-tuples
    (m, x, ty0, ty1)."""
    return (
        MAT_PREFIX
        + (tag + "_" if tag else "")
        + "_".join(str(int(p)) for p in img_key)
    )


def ensure_collection() -> bpy.types.Collection:
    coll = bpy.data.collections.get(COLLECTION_NAME)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION_NAME)
    scene_children = bpy.context.scene.collection.children
    if scene_children.get(coll.name) is None:
        scene_children.link(coll)
    return coll


def _stitched_image(
    name: str, paths: list[str], cols: int, save_path: str | None = None
) -> bpy.types.Image | None:
    """Stitch a row-major grid of tile files (south row first, west first
    within a row — matching Blender's bottom-up pixel layout) into one
    image. Source datablocks are freed immediately; pixel values are copied
    raw (both sides sRGB-encoded), so no color shift. The result is saved
    to save_path (disk cache) so rebuilds — and later sessions — load a
    file instead of repeating main-thread pixel work, and the image is
    file-backed (a purely generated datablock can drop its buffer on
    undo/reload and render magenta)."""
    img = bpy.data.images.get(name)
    if img is not None:
        return img
    parts = []
    for p in paths:
        try:
            src = bpy.data.images.load(p, check_existing=True)
        except RuntimeError:
            return None
        w, h = src.size
        if w == 0 or h == 0:
            return None
        buf = np.empty(w * h * 4, dtype=np.float32)
        src.pixels.foreach_get(buf)
        if src.users == 0:
            bpy.data.images.remove(src)
        parts.append(buf.reshape(h, w, 4))
    if any(p.shape != parts[0].shape for p in parts):
        return None
    rows = [
        np.concatenate(parts[r * cols:(r + 1) * cols], axis=1)
        for r in range(len(parts) // cols)
    ]
    full = rows[0] if len(rows) == 1 else np.concatenate(rows, axis=0)
    img = bpy.data.images.new(
        name, width=full.shape[1], height=full.shape[0], alpha=True
    )
    img.colorspace_settings.name = "sRGB"
    img.pixels.foreach_set(np.ascontiguousarray(full).ravel())
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.filepath_raw = save_path
            img.file_format = "PNG"
            img.save()
            img.source = "FILE"
        except (OSError, RuntimeError):
            pass    # in-memory image still works for this session
    return img


def get_or_create_material(
    img_key, img_paths, tag: str = "", save_path: str | None = None
) -> bpy.types.Material:
    """img_paths: row-major grid of tile files (south row first); a single
    entry is used directly, multiples are stitched (and persisted to
    save_path). Mercator cover keys (m, tx0, ty0, tx1, ty1) carry the grid
    width. Any image failure degrades to the gray material — a texture node
    without an image renders magenta, which reads as a bug."""
    if isinstance(img_paths, str):
        img_paths = [img_paths]
    name = material_name(img_key, tag)
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat

    img = None
    try:
        if len(img_paths) > 1:
            cols = (
                int(img_key[3]) - int(img_key[1]) + 1 if len(img_key) == 5 else 1
            )
            img = _stitched_image(
                "CesiumStitch_" + name[len(MAT_PREFIX):], img_paths, cols,
                save_path,
            )
        if img is None:
            img = bpy.data.images.load(img_paths[0], check_existing=True)
            img.colorspace_settings.name = "sRGB"
        if img.size[0] == 0:
            img = None
    except RuntimeError:
        img = None
    if img is None:
        print(f"[cesium] imagery unusable for {img_key}: {img_paths[:1]}")
        return get_or_create_gray_material()

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
    if md.imagery_key is not None:
        # remembered so style toggles can restore the exact texture whose
        # projection is baked into this mesh's UVs (scheme-agnostic)
        obj["cesium_img"] = list(md.imagery_key)
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


def tag_redraw_sidebar():
    """Redraw only the N-panel region — the panel does not refresh on its
    own, so without this the Status counters freeze and streaming looks
    stalled even when it is working."""
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type == "UI":
                    region.tag_redraw()


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
