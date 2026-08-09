"""Relief-map terrain style, after Owen Powell's Blender GIS dioramas
(BlenderNation, 2016): hypsometric height gradient + slope-based rock,
matte model-like material, neutral studio backdrop with a composited mist
pass for aerial perspective.

BPY module, main thread only. One shared material replaces the per-tile
satellite imagery while active; toggling off restores imagery from the disk
cache. The density field uses world position (same EEVEE constraint as
clouds.py: Texture Coordinate outputs don't evaluate in some contexts, and
world position also makes the gradient seamless across tile objects).
"""

from __future__ import annotations

import bpy

RELIEF_MAT_NAME = "CesiumRelief"
MAP_WORLD_NAME = "CesiumMapWorld"
PREV_WORLD_PROP = "cesium_prev_world"

EARTH_R = 6378137.0

# Beddgelert-inspired hypsometric stops (position 0..1 over the elevation
# range): valley green -> light green -> buff -> rock brown -> pale summit.
PALETTE = [
    (0.00, (0.169, 0.292, 0.152, 1.0)),
    (0.22, (0.302, 0.410, 0.209, 1.0)),
    (0.45, (0.557, 0.502, 0.312, 1.0)),
    (0.68, (0.475, 0.386, 0.284, 1.0)),
    (0.86, (0.727, 0.696, 0.630, 1.0)),
    (1.00, (0.937, 0.926, 0.898, 1.0)),
]
ROCK_COLOR = (0.412, 0.365, 0.306, 1.0)
BACKDROP = (0.545, 0.531, 0.494)          # warm studio gray, matches reference

ACTIVE = False


def is_active() -> bool:
    return ACTIVE


def _build_relief_material(min_e: float, max_e: float, contour_interval: float):
    mat = bpy.data.materials.get(RELIEF_MAT_NAME)
    if mat is None:
        mat = bpy.data.materials.new(RELIEF_MAT_NAME)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (620, 0)
    bsdf.inputs["Roughness"].default_value = 0.95
    spec = bsdf.inputs.get("Specular IOR Level")
    if spec is not None:
        spec.default_value = 0.12

    geo = nt.nodes.new("ShaderNodeNewGeometry")
    geo.location = (-1300, 0)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-1100, 100)

    # curvature-corrected elevation: e = z + (x^2 + y^2) / (2R), so contours
    # and tints stay level even 100+ km from the ENU origin where the globe
    # surface has dropped away
    x2 = nt.nodes.new("ShaderNodeMath")
    x2.operation = "MULTIPLY"
    x2.location = (-940, 220)
    y2 = nt.nodes.new("ShaderNodeMath")
    y2.operation = "MULTIPLY"
    y2.location = (-940, 60)
    xy2 = nt.nodes.new("ShaderNodeMath")
    xy2.operation = "ADD"
    xy2.location = (-780, 140)
    bulge = nt.nodes.new("ShaderNodeMath")
    bulge.operation = "DIVIDE"
    bulge.location = (-620, 140)
    bulge.inputs[1].default_value = 2.0 * EARTH_R
    elev = nt.nodes.new("ShaderNodeMath")
    elev.operation = "ADD"
    elev.location = (-460, 60)

    norm_e = nt.nodes.new("ShaderNodeMapRange")
    norm_e.location = (-300, 60)
    norm_e.inputs["From Min"].default_value = min_e
    norm_e.inputs["From Max"].default_value = max_e

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-120, 120)
    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[0])
    first = ramp.color_ramp.elements[0]
    first.position, first.color = PALETTE[0][0], PALETTE[0][1]
    for pos, color in PALETTE[1:]:
        el = ramp.color_ramp.elements.new(pos)
        el.color = color

    # slope factor from the surface normal: flat -> hypsometric tint,
    # steep -> bare rock (reads like the reference's crags)
    nsep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nsep.location = (-460, -220)
    slope = nt.nodes.new("ShaderNodeMapRange")
    slope.location = (-300, -220)
    slope.interpolation_type = "SMOOTHSTEP"
    slope.inputs["From Min"].default_value = 0.70   # normal.z: steep
    slope.inputs["From Max"].default_value = 0.93   # ~flat
    slope.inputs["To Min"].default_value = 1.0
    slope.inputs["To Max"].default_value = 0.0

    mix_rock = nt.nodes.new("ShaderNodeMix")
    mix_rock.data_type = "RGBA"
    mix_rock.location = (180, 0)
    mix_rock.inputs["B"].default_value = ROCK_COLOR

    ln = nt.links.new
    ln(geo.outputs["Position"], sep.inputs["Vector"])
    ln(sep.outputs["X"], x2.inputs[0])
    ln(sep.outputs["X"], x2.inputs[1])
    ln(sep.outputs["Y"], y2.inputs[0])
    ln(sep.outputs["Y"], y2.inputs[1])
    ln(x2.outputs[0], xy2.inputs[0])
    ln(y2.outputs[0], xy2.inputs[1])
    ln(xy2.outputs[0], bulge.inputs[0])
    ln(sep.outputs["Z"], elev.inputs[0])
    ln(bulge.outputs[0], elev.inputs[1])
    ln(elev.outputs[0], norm_e.inputs["Value"])
    ln(norm_e.outputs["Result"], ramp.inputs["Fac"])
    ln(geo.outputs["True Normal"], nsep.inputs["Vector"])
    ln(nsep.outputs["Z"], slope.inputs["Value"])
    ln(ramp.outputs["Color"], mix_rock.inputs["A"])
    ln(slope.outputs["Result"], mix_rock.inputs["Factor"])

    color_out = mix_rock.outputs["Result"]

    if contour_interval > 0:
        # thin darkening line where elevation crosses each interval
        emod = nt.nodes.new("ShaderNodeMath")
        emod.operation = "MODULO"
        emod.location = (-120, -420)
        emod.inputs[1].default_value = contour_interval
        efrac = nt.nodes.new("ShaderNodeMath")
        efrac.operation = "DIVIDE"
        efrac.location = (40, -420)
        efrac.inputs[1].default_value = contour_interval
        band = nt.nodes.new("ShaderNodeMapRange")
        band.location = (200, -420)
        band.interpolation_type = "SMOOTHSTEP"
        band.inputs["From Min"].default_value = 0.0
        band.inputs["From Max"].default_value = 0.035
        band.inputs["To Min"].default_value = 0.55   # line darkness
        band.inputs["To Max"].default_value = 1.0
        dark = nt.nodes.new("ShaderNodeMix")
        dark.data_type = "RGBA"
        dark.blend_type = "MULTIPLY"
        dark.location = (400, -120)
        dark.inputs["Factor"].default_value = 1.0
        ln(elev.outputs[0], emod.inputs[0])
        ln(emod.outputs[0], efrac.inputs[0])
        ln(efrac.outputs[0], band.inputs["Value"])
        ln(mix_rock.outputs["Result"], dark.inputs["A"])
        gray = nt.nodes.new("ShaderNodeCombineColor")
        gray.location = (240, -240)
        ln(band.outputs["Result"], gray.inputs["Red"])
        ln(band.outputs["Result"], gray.inputs["Green"])
        ln(band.outputs["Result"], gray.inputs["Blue"])
        ln(gray.outputs["Color"], dark.inputs["B"])
        color_out = dark.outputs["Result"]

    ln(color_out, bsdf.inputs["Base Color"])
    ln(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _studio_world(scene):
    """Swap to a plain warm-gray backdrop (previous world is remembered on
    the scene for restore)."""
    if scene.world is not None and scene.world.name != MAP_WORLD_NAME:
        scene[PREV_WORLD_PROP] = scene.world.name
    world = bpy.data.worlds.get(MAP_WORLD_NAME)
    if world is None:
        world = bpy.data.worlds.new(MAP_WORLD_NAME)
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (*BACKDROP, 1.0)
    bg.inputs["Strength"].default_value = 0.7
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    world.mist_settings.start = 15000.0
    world.mist_settings.depth = 120000.0
    world.mist_settings.falloff = "QUADRATIC"
    scene.world = world


def _studio_light():
    sun = bpy.data.objects.get("Sun")
    if sun is None or sun.type != "LIGHT":
        return
    import math

    sun.data.energy = 6.0
    sun.data.color = (1.0, 0.955, 0.88)
    sun.data.angle = math.radians(2.0)          # soft model-like shadows
    sun.rotation_euler = (math.radians(55), 0.0, math.radians(135))


def _mist_compositor(scene, enable: bool):
    """Composite the mist pass toward the backdrop color — distant terrain
    fades out like the reference dioramas, and the empty sky becomes the
    studio backdrop."""
    vl = scene.view_layers[0]
    vl.use_pass_mist = enable
    scene.use_nodes = enable
    if not enable:
        return
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    rl.location = (-300, 0)
    mix = nt.nodes.new("CompositorNodeMixRGB")
    mix.location = (0, 0)
    mix.inputs[2].default_value = (*BACKDROP, 1.0)
    comp = nt.nodes.new("CompositorNodeComposite")
    comp.location = (250, 0)
    nt.links.new(rl.outputs["Image"], mix.inputs[1])
    nt.links.new(rl.outputs["Mist"], mix.inputs[0])
    nt.links.new(mix.outputs["Image"], comp.inputs[0])


def _tile_objects():
    coll = bpy.data.collections.get("Cesium Terrain")
    return list(coll.objects) if coll is not None else []


def apply(context, min_e: float, max_e: float, contour_interval: float,
          studio: bool) -> int:
    global ACTIVE
    mat = _build_relief_material(min_e, max_e, contour_interval)
    n = 0
    for obj in _tile_objects():
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            n += 1
    if studio:
        _studio_world(context.scene)
        _studio_light()
        _mist_compositor(context.scene, True)
    ACTIVE = True
    return n


def remove(context) -> int:
    """Back to satellite imagery (from the disk cache) and the previous world."""
    global ACTIVE
    from . import scene_builder, streamer, tiling

    s = streamer.get()
    n = 0
    for obj in _tile_objects():
        if obj.type != "MESH" or "cesium_key" not in obj:
            continue
        z, x, y = obj["cesium_key"]
        mat = None
        if s.imagery is not None:
            # the exact imagery tile whose projection is baked into the UVs
            # is remembered on the object (works for both geodetic and
            # mercator schemes); fall back to recomputing the geodetic key
            # for objects built before that property existed
            ikey = obj.get("cesium_img")
            if ikey is not None:
                ikey = tuple(ikey)
            elif s.imagery.scheme == "geodetic":
                ikey, *_ = tiling.imagery_key_and_uv_transform(
                    z, x, y, s.imagery.max_zoom
                )
            if ikey is not None:
                paths = s.imagery.paths_for_key(ikey)
                if paths is not None:
                    tag = "M" if s.imagery.scheme == "mercator" else ""
                    sp = (
                        s.imagery.cache.stitched_path(ikey)
                        if len(paths) > 1
                        else None
                    )
                    mat = scene_builder.get_or_create_material(
                        ikey, paths, tag, sp
                    )
        if mat is None:
            mat = scene_builder.get_or_create_gray_material()
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        n += 1
    prev = context.scene.get(PREV_WORLD_PROP)
    if prev and bpy.data.worlds.get(prev) is not None:
        context.scene.world = bpy.data.worlds[prev]
    _mist_compositor(context.scene, False)
    relief = bpy.data.materials.get(RELIEF_MAT_NAME)
    if relief is not None and relief.users == 0:
        bpy.data.materials.remove(relief)
    ACTIVE = False
    return n
