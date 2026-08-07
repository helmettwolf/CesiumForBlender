"""Procedural volumetric cloud layer over the streamed terrain.

BPY module, main thread only. One flat volume slab centered on the active
camera: noise-cell density with parabolic vertical falloff and edge fade,
computed entirely from WORLD position — EEVEE Next volume shaders do not
evaluate Texture Coordinate Generated/Object outputs (they come back
constant, collapsing any density pattern into uniform fog), but Geometry >
Position works. Slab geometry is baked into the node constants at build
time; the operator re-runs the build whenever it moves the layer.

The operator also retunes the scene's froxel range: the default
volumetric_end of 100 m would clip a cloud field kilometers away entirely.
"""

from __future__ import annotations

import bpy
from mathutils import Vector

CLOUD_OBJ_NAME = "Cesium Clouds"
CLOUD_MAT_NAME = "CesiumClouds"

CELL_SIZE_M = 15000.0        # horizontal cumulus cell scale
VERT_NOISE_M = 2500.0        # vertical noise variation scale

_CUBE_VERTS = [
    (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
]
_CUBE_FACES = [
    (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
    (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
]


def remove_clouds():
    obj = bpy.data.objects.get(CLOUD_OBJ_NAME)
    if obj is not None:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    mat = bpy.data.materials.get(CLOUD_MAT_NAME)
    if mat is not None and mat.users == 0:
        bpy.data.materials.remove(mat)


def _camera_position(context) -> Vector:
    # Prefer the viewport eye — clouds belong over what the user is looking
    # at, which the streamer is also refining. Fall back to the render camera.
    from . import camera as camera_mod

    state = camera_mod.get_camera_state()
    if state is not None:
        return Vector(state.pos)
    cam = context.scene.camera
    if cam is not None:
        return cam.matrix_world.translation.copy()
    return Vector((0.0, 0.0, 0.0))


def _math(nt, op, loc, v0=None, v1=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.location = loc
    if v0 is not None:
        n.inputs[0].default_value = v0
    if v1 is not None:
        n.inputs[1].default_value = v1
    return n


def _build_material(
    coverage: float,
    density: float,
    center_xy: tuple[float, float],
    base_z: float,
    thickness: float,
    half_size: float,
    anisotropy: float = 0.15,
):
    """World-position-driven density field. All slab geometry is baked in as
    node constants — rebuild whenever the slab moves."""
    mat = bpy.data.materials.get(CLOUD_MAT_NAME)
    if mat is None:
        mat = bpy.data.materials.new(CLOUD_MAT_NAME)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (1100, 0)
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.location = (850, 0)
    vol.inputs["Anisotropy"].default_value = anisotropy

    geo = nt.nodes.new("ShaderNodeNewGeometry")
    geo.location = (-1400, 0)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-1200, -200)

    # --- cumulus cells from world-space noise ---
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-1200, 150)
    mapping.inputs["Scale"].default_value = (
        1.0 / CELL_SIZE_M, 1.0 / CELL_SIZE_M, 1.0 / VERT_NOISE_M,
    )
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-1000, 150)
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.6
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.location = (-780, 150)
    ramp.interpolation_type = "SMOOTHSTEP"
    # Noise Fac (detail 5) clusters tightly around 0.5, so the usable
    # threshold band is narrow. Mapping calibrated against live renders:
    # thr 0.63 -> sparse wisps, 0.575 -> broken cumulus, 0.46 -> solid deck.
    thr = 0.684 - 0.26 * coverage
    ramp.inputs["From Min"].default_value = thr - 0.045
    ramp.inputs["From Max"].default_value = thr + 0.045

    # --- vertical parabola 4t(1-t), t = (z - base_z) / thickness ---
    z_off = _math(nt, "SUBTRACT", (-1000, -200), v1=base_z)
    z_t = _math(nt, "DIVIDE", (-840, -200), v1=thickness)
    one_minus_t = _math(nt, "SUBTRACT", (-680, -260), v0=1.0)
    t_par = _math(nt, "MULTIPLY", (-520, -200))
    t_par4 = _math(nt, "MULTIPLY", (-360, -200), v1=4.0)
    t_clamp = _math(nt, "MAXIMUM", (-200, -200), v1=0.0)

    # --- edge fade: Chebyshev distance from slab center in world xy ---
    dx = _math(nt, "SUBTRACT", (-1000, -420), v1=center_xy[0])
    dxa = _math(nt, "ABSOLUTE", (-840, -420))
    dy = _math(nt, "SUBTRACT", (-1000, -560), v1=center_xy[1])
    dya = _math(nt, "ABSOLUTE", (-840, -560))
    dmax = _math(nt, "MAXIMUM", (-680, -490))
    edge = nt.nodes.new("ShaderNodeMapRange")
    edge.location = (-520, -490)
    edge.interpolation_type = "SMOOTHSTEP"
    edge.inputs["From Min"].default_value = 0.72 * half_size
    edge.inputs["From Max"].default_value = half_size
    edge.inputs["To Min"].default_value = 1.0
    edge.inputs["To Max"].default_value = 0.0

    mul_shape = _math(nt, "MULTIPLY", (0, 0))
    mul_edge = _math(nt, "MULTIPLY", (160, 0))
    mul_density = _math(nt, "MULTIPLY", (320, 0), v1=density)

    ln = nt.links.new
    ln(geo.outputs["Position"], mapping.inputs["Vector"])
    ln(mapping.outputs["Vector"], noise.inputs["Vector"])
    ln(noise.outputs["Fac"], ramp.inputs["Value"])
    ln(geo.outputs["Position"], sep.inputs["Vector"])
    ln(sep.outputs["Z"], z_off.inputs[0])
    ln(z_off.outputs[0], z_t.inputs[0])
    ln(z_t.outputs[0], one_minus_t.inputs[1])
    ln(z_t.outputs[0], t_par.inputs[0])
    ln(one_minus_t.outputs[0], t_par.inputs[1])
    ln(t_par.outputs[0], t_par4.inputs[0])
    ln(t_par4.outputs[0], t_clamp.inputs[0])
    ln(sep.outputs["X"], dx.inputs[0])
    ln(dx.outputs[0], dxa.inputs[0])
    ln(sep.outputs["Y"], dy.inputs[0])
    ln(dy.outputs[0], dya.inputs[0])
    ln(dxa.outputs[0], dmax.inputs[0])
    ln(dya.outputs[0], dmax.inputs[1])
    ln(dmax.outputs[0], edge.inputs["Value"])
    ln(ramp.outputs["Result"], mul_shape.inputs[0])
    ln(t_clamp.outputs[0], mul_shape.inputs[1])
    ln(mul_shape.outputs[0], mul_edge.inputs[0])
    ln(edge.outputs["Result"], mul_edge.inputs[1])
    ln(mul_edge.outputs[0], mul_density.inputs[0])
    ln(mul_density.outputs[0], vol.inputs["Density"])
    ln(vol.outputs["Volume"], out.inputs["Volume"])
    return mat


def _tune_volumetrics(scene, self_shadow: bool):
    """EEVEE froxel settings for a cloud field kilometers away. The froxel
    grid slices view depth, so volumetric_end directly sets slice thickness:
    60 km with 256 samples gives ~250 m slices — enough to resolve the slab
    without smearing it into mist."""
    ee = scene.eevee
    for attr, value in (
        ("volumetric_start", 50.0),
        ("volumetric_end", 60000.0),
        ("volumetric_tile_size", "8"),
        ("volumetric_samples", 256),
        ("volumetric_sample_distribution", 0.4),
        ("use_volumetric_shadows", self_shadow),
        ("volumetric_shadow_samples", 16),
    ):
        try:
            setattr(ee, attr, value)
        except (AttributeError, TypeError):
            pass


def _ground_z_under(context, pos: Vector) -> float | None:
    """Ray-cast straight down from high above (pos.x, pos.y) to find terrain
    height, skipping the cloud slab itself."""
    dg = context.evaluated_depsgraph_get()
    origin = Vector((pos.x, pos.y, pos.z + 100_000.0))
    direction = Vector((0.0, 0.0, -1.0))
    for _ in range(4):
        hit, loc, _n, _i, obj, _m = context.scene.ray_cast(dg, origin, direction)
        if not hit:
            return None
        if obj is None or obj.name != CLOUD_OBJ_NAME:
            return loc.z
        origin = loc + Vector((0.0, 0.0, -1.0))   # passed through the slab face
    return None


def add_or_update(
    context,
    base_agl: float = 2200.0,
    thickness: float = 1800.0,
    coverage: float = 0.45,
    density: float = 0.008,
    size_km: float = 160.0,
    self_shadow: bool = True,
) -> bpy.types.Object:
    cam_pos = _camera_position(context)

    obj = bpy.data.objects.get(CLOUD_OBJ_NAME)
    if obj is None:
        mesh = bpy.data.meshes.new(CLOUD_OBJ_NAME)
        mesh.from_pydata(_CUBE_VERTS, [], _CUBE_FACES)
        mesh.update()
        obj = bpy.data.objects.new(CLOUD_OBJ_NAME, mesh)
        context.scene.collection.objects.link(obj)
    obj.display_type = "BOUNDS"   # never occludes the viewport in solid mode

    size_m = size_km * 1000.0
    obj.scale = (size_m, size_m, thickness)
    ground_z = _ground_z_under(context, cam_pos)
    if ground_z is None:
        ground_z = cam_pos.z - 3000.0   # no terrain hit: assume camera ~3 km up
    base_z = ground_z + base_agl
    obj.location = (cam_pos.x, cam_pos.y, base_z + thickness * 0.5)

    mat = _build_material(
        coverage,
        density,
        center_xy=(cam_pos.x, cam_pos.y),
        base_z=base_z,
        thickness=thickness,
        half_size=size_m * 0.5,
    )
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    _tune_volumetrics(context.scene, self_shadow)
    return obj
