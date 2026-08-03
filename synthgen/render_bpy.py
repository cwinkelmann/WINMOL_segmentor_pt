"""Render one SceneSpec to rgb/mask/depth PNGs. The only module that imports bpy.

Run as a subprocess so a renderer crash costs one tile, not the run:

    python -m synthgen.render_bpy <spec.json> <out_dir>

Masks come from Cycles' object-index pass (stems carry ``pass_index=1``), not a
second black-and-white render: one render, so the mask cannot drift out of
registration with the image. Depth comes from the Z pass, mapped through a known
altitude window so every tile shares one physical scale.
"""
import colorsys
import math
import os
import sys

import bpy

STEM_INDEX = 1
_GSD_HINT = [0.02]        # scene GSD, set per render so relief can be sized in pixels
_DEPTH_MARGIN_M = 3.0        # window half-width around the camera altitude


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def _configure_render(scene, spec, samples=48):
    _GSD_HINT[0] = spec["gsd_m_per_px"]
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = scene.render.resolution_y = spec["tile_px"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    # GPU when the machine has one; harmless no-op on CPU-only hosts
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = dev.type in ("CUDA", "OPTIX")
        scene.cycles.device = "GPU"
    except Exception:                                     # pragma: no cover
        scene.cycles.device = "CPU"

    layer = scene.view_layers[0]
    layer.use_pass_object_index = True
    layer.use_pass_z = True


def _bark_material(stem):
    mat = bpy.data.materials.new("bark")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    # brown bark, jittered per stem for decay/moisture/species variety
    hue = (0.075 + 0.025 * stem["bark_hue_shift"]) % 1.0
    sat = 0.18 + 0.14 * (1.0 - stem["bark_darkness"])     # weathered bark is grey-brown
    val = 0.16 * (1.0 - 0.55 * stem["bark_darkness"]) + 0.04
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.75 + 0.2 * stem["bark_darkness"]
    # bark relief: noise -> bump, so grazing light produces the streaked texture
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 45.0
    noise.inputs["Detail"].default_value = 8.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _ground_material(ground):
    mat = bpy.data.materials.new("ground")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    b = ground["brightness"]
    bsdf.inputs["Base Color"].default_value = (0.14 * b, 0.12 * b, 0.08 * b, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    # litter/soil mottling; a real-orthomosaic crop replaces this when supplied
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 12.0
    noise.inputs["Detail"].default_value = 10.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.10 * b, 0.09 * b, 0.06 * b, 1.0)
    ramp.color_ramp.elements[1].color = (0.26 * b, 0.22 * b, 0.14 * b, 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _build_ground(spec):
    size = spec["camera"]["footprint_m"] * 1.6          # overfill so edges never show
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.pass_index = 0
    plane.data.materials.append(_ground_material(spec["ground"]))
    return plane


def _build_stem(stem):
    """Tapered, bent tube from a 3-point curve: per-point radius gives the taper,
    the lifted middle point gives the bend."""
    L, r = stem["length_m"], stem["diameter_m"] / 2.0
    curve = bpy.data.curves.new("stem", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 6
    curve.bevel_depth = r
    curve.bevel_resolution = 6
    curve.use_fill_caps = True

    spline = curve.splines.new("NURBS")
    spline.points.add(2)                                 # 3 points total
    bend = stem["bend_m"]
    for i, (t, rad) in enumerate(((-0.5, 1.0), (0.0, 0.5 * (1.0 + stem["taper"])),
                                  (0.5, stem["taper"]))):
        spline.points[i].co = (t * L, bend if i == 1 else 0.0, 0.0, 1.0)
        spline.points[i].radius = rad
    spline.use_endpoint_u = True

    obj = bpy.data.objects.new("stem", curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_bark_material(stem))
    obj.pass_index = STEM_INDEX                          # this is what becomes the mask

    # a buried stem sinks into the plane; a crossing one rests above it
    z = stem["elevation_m"] - stem["burial"] * r
    obj.location = (stem["center_xy_m"][0], stem["center_xy_m"][1], z + r)
    obj.rotation_euler = (0.0, 0.0, math.radians(stem["azimuth_deg"]))

    obj = _add_bark_relief(obj, stem)
    for k in range(stem["branch_stubs"]):
        _add_branch_stub(obj, stem, k)
    if stem["root_plate"]:
        _add_root_plate(obj, stem)
    return obj



def _add_bark_relief(obj, stem):
    """Displace the stem surface with noise so bark relief is geometry, not a
    shading trick.

    A bump map fakes relief in the RGB pass only — the depth pass still sees a
    perfect cylinder, and a model trained on RGBD would learn that stems are
    unnaturally smooth in depth. Displacing real vertices puts the same height
    profile into both passes (and into the silhouette, so the mask follows it).

    Scale matters more than realism here: true bark texture is ~1 cm, which is
    sub-pixel at a 2 cm/px GSD and measurably changes nothing. What a UAV tile
    actually resolves is coarser irregularity — knots, swellings, out-of-round
    cross-sections at 10-30 cm — so the noise is tuned to span several pixels
    rather than to imitate bark. Fine bark stays a shading-only bump map.
    """
    gsd = _GSD_HINT[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target="MESH")      # curves cannot carry a Displace
    mesh = bpy.context.active_object

    # enough vertices for the noise to have somewhere to go
    sub = mesh.modifiers.new("subdiv", "SUBSURF")
    sub.subdivision_type = "SIMPLE"
    sub.levels = sub.render_levels = 1

    tex = bpy.data.textures.new(f"bark_{mesh.name}", type="CLOUDS")
    # features ~8x the pixel size so displacement survives rasterization
    tex.noise_scale = max(8.0 * gsd, stem["diameter_m"] * 0.9)
    tex.noise_depth = 4
    tex.intensity = 1.0

    disp = mesh.modifiers.new("bark_relief", "DISPLACE")
    disp.texture = tex
    disp.texture_coords = "LOCAL"
    disp.mid_level = 0.5
    # amplitude floor of ~2 px so the relief is visible in depth, capped at a
    # quarter diameter so a stem stays a stem
    amp = stem["diameter_m"] * (0.10 + 0.14 * stem["bark_darkness"])
    disp.strength = float(min(max(amp, 4.0 * gsd), stem["diameter_m"] * 0.25))
    mesh.select_set(False)
    return mesh


def _add_branch_stub(parent, stem, k):
    r = stem["diameter_m"] / 2.0
    bpy.ops.mesh.primitive_cone_add(radius1=r * 0.28, radius2=r * 0.12, depth=r * 3.0)
    stub = bpy.context.active_object
    stub.pass_index = STEM_INDEX
    stub.data.materials.append(_bark_material(stem))
    along = (0.5 - (k + 1) / (stem["branch_stubs"] + 1.0)) * stem["length_m"]
    stub.location = (along, 0.0, 0.0)
    stub.rotation_euler = (math.radians(65 + 25 * ((k % 3) - 1)), 0.0,
                           math.radians(37.0 * k))
    stub.parent = parent


def _add_root_plate(parent, stem):
    r = stem["diameter_m"] / 2.0
    bpy.ops.mesh.primitive_cylinder_add(radius=r * 3.5, depth=r * 0.5)
    plate = bpy.context.active_object
    plate.pass_index = STEM_INDEX
    plate.data.materials.append(_bark_material(stem))
    plate.location = (-stem["length_m"] / 2.0, 0.0, r * 0.8)
    plate.rotation_euler = (math.radians(80.0), 0.0, 0.0)
    plate.parent = parent


def _build_clutter(spec, rng_state):
    """Leaves/branches/stones scattered on the floor, a share of them over stems.

    Clutter is pass_index 0: it is background, so anything lying across a stem
    occludes it in both image and mask exactly as real litter does.
    """
    import random
    rng = random.Random(rng_state)
    c, half = spec["clutter"], spec["camera"]["footprint_m"] / 2.0
    counts = (("leaf", int(c["leaf_density"] * 140), 0.05),
              ("branch", int(c["branch_density"] * 40), 0.10),
              ("stone", int(c["stone_density"] * 25), 0.06))
    mat = _ground_material({"brightness": spec["ground"]["brightness"] * 1.15})
    for kind, n, size in counts:
        for i in range(n):
            over = rng.random() < c["over_stem_fraction"] and spec["stems"]
            if over:
                s = rng.choice(spec["stems"])
                x, y = s["center_xy_m"]
                x += rng.uniform(-0.4, 0.4) * s["length_m"] * math.cos(
                    math.radians(s["azimuth_deg"]))
                y += rng.uniform(-0.4, 0.4) * s["length_m"] * math.sin(
                    math.radians(s["azimuth_deg"]))
                z = s["elevation_m"] + s["diameter_m"] * 0.55
            else:
                x, y = rng.uniform(-half, half), rng.uniform(-half, half)
                z = 0.01
            if kind == "branch":
                bpy.ops.mesh.primitive_cylinder_add(radius=size * 0.18, depth=size * 8)
            elif kind == "stone":
                bpy.ops.mesh.primitive_ico_sphere_add(radius=size, subdivisions=1)
            else:
                bpy.ops.mesh.primitive_plane_add(size=size * 2)
            o = bpy.context.active_object
            o.pass_index = 0
            o.location = (x, y, z)
            o.rotation_euler = (rng.uniform(0, 0.4), rng.uniform(0, 0.4),
                                rng.uniform(0, math.tau))
            o.data.materials.append(mat)


def _build_sun(spec):
    s = spec["sun"]
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 50))
    sun = bpy.context.active_object
    sun.data.energy = s["strength"]
    sun.data.angle = math.radians(1.5 + 8.0 * s["canopy_shadow"])   # softer under canopy
    el, az = math.radians(s["elevation_deg"]), math.radians(s["azimuth_deg"])
    sun.rotation_euler = (math.pi / 2 - el, 0.0, az)

    world = bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    world.use_nodes = True
    sky = 0.35 * (1.0 - 0.7 * s["canopy_shadow"])
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        sky * 0.8, sky * 0.9, sky, 1.0)

    if s["canopy_shadow"] > 0.15:
        _build_canopy_proxies(spec, s["canopy_shadow"])


def _build_canopy_proxies(spec, shadow):
    """Off-frame blockers that cast the dappled shade real forest floors sit in —
    the paper credits this variance for much of its false-negative reduction."""
    import random
    rng = random.Random(spec["seed"] + 977)
    half = spec["camera"]["footprint_m"] / 2.0
    for _ in range(int(3 + 9 * shadow)):
        bpy.ops.mesh.primitive_plane_add(size=half * rng.uniform(0.3, 1.1))
        blocker = bpy.context.active_object
        blocker.pass_index = 0
        # invisible to camera rays but still casting shadows: a blocker between
        # camera and ground would otherwise render as a giant pale slab
        blocker.visible_camera = False
        blocker.location = (rng.uniform(-half, half), rng.uniform(-half, half),
                            spec["camera"]["altitude_m"] + rng.uniform(3.0, 12.0))
        blocker.rotation_euler = (rng.uniform(0, 0.5), rng.uniform(0, 0.5),
                                  rng.uniform(0, math.tau))


def _build_camera(scene, spec):
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = spec["camera"]["focal_mm"]
    cam_data.sensor_width = spec["camera"]["sensor_mm"]
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (0.0, 0.0, spec["camera"]["altitude_m"])
    cam.rotation_euler = (0.0, 0.0, 0.0)                 # nadir
    scene.camera = cam
    return cam


def _flat_emission(color, name):
    """Unlit constant-colour shader: what a pixel shows does not depend on lights,
    so mask/depth passes need one sample and no denoising."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def _depth_material(spec):
    """Camera distance mapped through a fixed altitude window -> greyscale.

    The window is anchored to the camera altitude rather than per-tile min/max, so
    a given grey level means the same height in every tile of a dataset.
    """
    alt = spec["camera"]["altitude_m"]
    mat = bpy.data.materials.new("depth")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    cam = nt.nodes.new("ShaderNodeCameraData")
    rng = nt.nodes.new("ShaderNodeMapRange")
    rng.inputs["From Min"].default_value = alt - _DEPTH_MARGIN_M
    rng.inputs["From Max"].default_value = alt + 0.5
    rng.inputs["To Min"].default_value = 1.0            # near the camera = bright
    rng.inputs["To Max"].default_value = 0.0
    nt.links.new(cam.outputs["View Z Depth"], rng.inputs["Value"])
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(rng.outputs["Result"], em.inputs["Color"])
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def _render_to(scene, path, samples, denoise, raw=False, mode="RGB", depth_bits="8"):
    """`raw=True` renders data, not a picture: no tone mapping and a near-delta
    pixel filter, so an emitted 1.0 lands as 255 and edges stay hard. Without it
    AgX maps white to ~196 and the filter smears mask edges into grey."""
    scene.cycles.samples = samples
    scene.cycles.use_denoising = denoise
    saved_view, saved_filter = scene.view_settings.view_transform, scene.render.filter_size
    saved_dither = scene.render.dither_intensity
    if raw:
        scene.view_settings.view_transform = "Standard"
        scene.render.filter_size = 0.01
        scene.render.dither_intensity = 0.0     # default 1.0 adds +/-1 noise to 8-bit
    try:
        scene.render.filepath = path
        im = scene.render.image_settings
        im.file_format = "PNG"
        # order matters: file_format resets colour settings, so set them after
        im.color_mode = mode
        im.color_depth = depth_bits
        bpy.ops.render.render(write_still=True)
    finally:
        scene.view_settings.view_transform = saved_view
        scene.render.filter_size = saved_filter
        scene.render.dither_intensity = saved_dither


def _geometry(scene):
    return [o for o in scene.objects if o.type in ("MESH", "CURVE") and o.data]


def _render_mask(scene, out_dir):
    """Stems white, everything else black — the same guarantee the object-index
    pass gives (one geometry state, so registration is exact) without depending on
    the compositor API, which moved between Blender 4 and 5."""
    white = _flat_emission((1.0, 1.0, 1.0), "mask_fg")
    black = _flat_emission((0.0, 0.0, 0.0), "mask_bg")
    saved = {o: list(o.data.materials) for o in _geometry(scene)}
    world_saved = scene.world
    scene.world = None                                   # no sky light bleeding in
    try:
        for obj in saved:
            obj.data.materials.clear()
            obj.data.materials.append(white if obj.pass_index == STEM_INDEX else black)
        _render_to(scene, os.path.join(out_dir, "mask"), samples=1, denoise=False,
                   raw=True, mode="BW")
    finally:
        for obj, mats in saved.items():
            obj.data.materials.clear()
            for m in mats:
                obj.data.materials.append(m)
        scene.world = world_saved


def _render_depth(scene, spec, out_dir):
    layer = scene.view_layers[0]
    saved, world_saved = layer.material_override, scene.world
    scene.world = None
    try:
        layer.material_override = _depth_material(spec)
        _render_to(scene, os.path.join(out_dir, "depth"), samples=1, denoise=False,
                   raw=True, mode="BW", depth_bits="16")
    finally:
        layer.material_override = saved
        scene.world = world_saved


def render_spec(spec, out_dir, samples=48):
    """Build and render one scene. `spec` is a SceneSpec as a plain dict."""
    os.makedirs(out_dir, exist_ok=True)
    scene = _reset_scene()
    _configure_render(scene, spec, samples=samples)
    _build_ground(spec)
    for stem in spec["stems"]:
        _build_stem(stem)
    _build_clutter(spec, spec["seed"])
    _build_sun(spec)
    _build_camera(scene, spec)
    _render_to(scene, os.path.join(out_dir, "rgb"), samples=samples, denoise=True)
    _render_mask(scene, out_dir)
    _render_depth(scene, spec, out_dir)
    return out_dir


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    import json
    with open(argv[0]) as f:
        spec = json.load(f)
    render_spec(spec, argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
