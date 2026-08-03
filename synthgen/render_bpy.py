"""Render one SceneSpec to rgb/mask/depth PNGs. The only module that imports bpy.

Run as a subprocess so a renderer crash costs one tile, not the run:

    python -m synthgen.render_bpy <spec.json> <out_dir>

Masks come from Cycles' object-index pass (stems carry ``pass_index=1``), not a
second black-and-white render: one render, so the mask cannot drift out of
registration with the image. Depth comes from the Z pass, mapped through a known
altitude window so every tile shares one physical scale.
"""
import math
import os
import sys

import bpy
import mathutils

STEM_INDEX = 1
_DEPTH_MARGIN_M = 3.0        # window half-width around the camera altitude


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def _configure_render(scene, spec, samples=48):
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
    hue = 0.06 + 0.04 * stem["bark_hue_shift"]
    val = 0.22 * (1.0 - 0.6 * stem["bark_darkness"]) + 0.05
    bsdf.inputs["Base Color"].default_value = (*mathutils.Color((hue, 0.55, val)).from_hsv(
        hue % 1.0, 0.55, val), 1.0)
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

    for k in range(stem["branch_stubs"]):
        _add_branch_stub(obj, stem, k)
    if stem["root_plate"]:
        _add_root_plate(obj, stem)
    return obj


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
        blocker.location = (rng.uniform(-half, half), rng.uniform(-half, half),
                            rng.uniform(4.0, 12.0))
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


def _wire_outputs(scene, spec, out_dir):
    """Compositor: beauty -> rgb.png, object index -> mask.png, Z -> depth.png."""
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")

    rgb = nt.nodes.new("CompositorNodeOutputFile")
    rgb.base_path = out_dir
    rgb.file_slots[0].path = "rgb"
    rgb.format.file_format = "PNG"
    rgb.format.color_mode = "RGB"
    nt.links.new(rl.outputs["Image"], rgb.inputs[0])

    # IndexOB is a discrete id map: >0.5 isolates the stems, no anti-aliased edge
    # to threshold ambiguously
    step = nt.nodes.new("CompositorNodeMath")
    step.operation = "GREATER_THAN"
    step.inputs[1].default_value = 0.5
    nt.links.new(rl.outputs["IndexOB"], step.inputs[0])
    mask = nt.nodes.new("CompositorNodeOutputFile")
    mask.base_path = out_dir
    mask.file_slots[0].path = "mask"
    mask.format.file_format = "PNG"
    mask.format.color_mode = "BW"
    nt.links.new(step.outputs[0], mask.inputs[0])

    # map a fixed altitude window to [0,1] so depth shares one physical scale
    # across tiles instead of being per-tile normalized
    alt = spec["camera"]["altitude_m"]
    rng_node = nt.nodes.new("CompositorNodeMapRange")
    rng_node.inputs[1].default_value = alt - _DEPTH_MARGIN_M
    rng_node.inputs[2].default_value = alt + 0.5
    rng_node.inputs[3].default_value = 1.0               # near = bright
    rng_node.inputs[4].default_value = 0.0
    nt.links.new(rl.outputs["Depth"], rng_node.inputs[0])
    depth = nt.nodes.new("CompositorNodeOutputFile")
    depth.base_path = out_dir
    depth.file_slots[0].path = "depth"
    depth.format.file_format = "PNG"
    depth.format.color_mode = "BW"
    depth.format.color_depth = "16"
    nt.links.new(rng_node.outputs[0], depth.inputs[0])


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
    _wire_outputs(scene, spec, out_dir)
    bpy.ops.render.render(write_still=False)             # File Output nodes do the writing
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
