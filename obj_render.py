"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 24 \
        --camera_dist 1.2 \
        --res 512 \
        --glossy no

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
import numpy as np
import imageio


import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=50)
parser.add_argument("--camera_dist", type=int, default=1.2)
parser.add_argument("--res", type=int, default=64)
parser.add_argument("--glossy", type=str, default='yes')

args = parser.parse_args()

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.res
render.resolution_y = args.res
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 0
scene.cycles.glossy_bounces = 0
scene.cycles.transparent_max_bounces = 0
scene.cycles.transmission_bounces = 0
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True
AABB = 1

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
#cam.location = (0, 0, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# setup lighting
bpy.ops.object.light_add(type="AREA")
light2 = bpy.data.lights["Area"]
light2.energy = 30000
bpy.data.objects["Area"].location[2] = 0.5
bpy.data.objects["Area"].scale[0] = 100
bpy.data.objects["Area"].scale[1] = 100
bpy.data.objects["Area"].scale[2] = 100


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def randomize_lighting() -> None:
    light2.energy = random.uniform(5000, 35000)
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 0
    bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)


def reset_lighting() -> None:
    light2.energy = 3_0
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 0
    bpy.data.objects["Area"].location[2] = 0.5


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    camera = dict()
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    os.makedirs(os.path.join(args.output_dir, object_uid, 'masks'), exist_ok=True)
    normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    params = dict()
    camera_extr_dict = []
    for i in range(args.num_images):
        # set the camera position
        theta = (i / args.num_images) * math.pi * 2
        rads = np.random.choice([90,75,60,45])
        phi = math.radians(rads)
        point = (
                args.camera_dist * math.sin(phi) * math.cos(theta),
                args.camera_dist * math.sin(phi) * math.sin(theta),
                args.camera_dist * math.cos(phi),
            )
        reset_lighting()
        cam.location = point
        # render the image
        render_path = os.path.join(args.output_dir, object_uid, 'images', f"{i}.png")
        scene.render.filepath = render_path
        if args.glossy != 'yes':
            for light in bpy.data.lights:
                light.specular_factor = 0.0
        bpy.ops.render.render(write_still=True)
        frame_data = {
                'file_path': render_path,
                'transform_matrix': listify_matrix(bpy.context.scene.camera.matrix_world)
            }

        camera_extr_dict.append(frame_data)
        
        im = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(mask_path, (im[:,:,3]>0)*255)
        mask_path = os.path.join(args.output_dir, object_uid, 'masks', f"{i}.png")
        png = Image.open(render_path)
        png.load() 
        png.split()[3].save(mask_path)
   
        intrinsics = get_calibration_matrix_K_from_blender()
        extrinsics = np.append(get_camera_parameters_extrinsic(bpy.context.scene),[[0,0,0,1]],0)
        params[f"camera_mat_{i}"] = intrinsics
        params[f"world_mat_{i}"] = extrinsics 
    np.savez(os.path.join(args.output_dir, object_uid, "cameras.npz"), **params)
        
        
    transforms = get_intr()
    transforms['frames'] = camera_extr_dict
    import json
    with open(os.path.join(args.output_dir, object_uid, 'transforms.json'), 'w') as fp:
        json.dump(transforms, fp)
        
            


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path

def get_intr():
    
    scene = bpy.context.scene
    camera = scene.camera
    
    camera_angle_x = camera.data.angle_x
    camera_angle_y = camera.data.angle_y

    f_in_mm = camera.data.lens
    scale = scene.render.resolution_percentage / 100
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    width_res_in_px = scene.render.resolution_x * scale
    height_res_in_px = scene.render.resolution_y * scale

    optical_center_x = width_res_in_px / 2
    optical_center_y = height_res_in_px / 2

    sensor_size_in_mm = camera.data.sensor_height if camera.data.sensor_fit == 'VERTICAL' else camera.data.sensor_width

    size_x = scene.render.pixel_aspect_x * width_res_in_px
    size_y = scene.render.pixel_aspect_y * height_res_in_px

    if camera.data.sensor_fit == 'AUTO':
        sensor_fit = 'HORIZONTAL' if size_x >= size_y else 'VERTICAL'
    else :
        sensor_fit = camera.data.sensor_fit

    view_fac_in_px = width_res_in_px if sensor_fit == 'HORIZONTAL' else pixel_aspect_ratio * height_res_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px

    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    camera_intr_dict = {
            'camera_angle_x': camera_angle_x,
            'camera_angle_y': camera_angle_y,
            'fl_x': s_u,
            'fl_y': s_v,
            'k1': 0.0,
            'k2': 0.0,
            'p1': 0.0,
            'p2': 0.0,
            'cx': optical_center_x,
            'cy': optical_center_y,
            'w': width_res_in_px,
            'h': height_res_in_px,
            'aabb_scale': AABB
        }
    return camera_intr_dict

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def get_calibration_matrix_K_from_blender(mode='complete'):

    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale # px
    height = scene.render.resolution_y * scale # px

    camdata = scene.camera.data

    if mode == 'simple':

        aspect_ratio = width / height
        K = np.zeros((3,3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()
    
    if mode == 'complete':

        focal = camdata.lens # mm
        sensor_width = camdata.sensor_width # mm
        sensor_height = camdata.sensor_height # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal), 
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio 
            s_v = height / sensor_height
        else: 
            # the sensor width is fixed (sensor fit is horizontal), 
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0 # only use rectangular pixels

        K = np.array([
            [alpha_u,    skew, u_0,  0],
            [      0, alpha_v, v_0,  0],
            [      0,       0,   1,  0],
            [      0,       0,   0,  1]
        ], dtype=np.float32)
    
    return K

def get_camera_parameters_extrinsic(scene):
    """ Get extrinsic camera parameters. 
    
      There are 3 coordinate systems involved:
         1. The World coordinates: "world"
            - right-handed
         2. The Blender camera coordinates: "bcam"
            - x is horizontal
            - y is up
            - right-handed: negative z look-at direction
         3. The desired computer vision camera coordinates: "cv"
            - x is horizontal
            - y is down (to align to the actual pixel coordinates 
               used in digital images)
            - right-handed: positive z look-at direction
      ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    """
    # bcam stands for blender camera
    bcam = scene.camera
    R_bcam2cv = np.array([[1,  0,  0],
                          [0, -1,  0],
                          [0,  0, -1]])

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location = np.array([bcam.matrix_world.decompose()[0]]).T
    R_world2bcam = np.array(bcam.matrix_world.decompose()[1].to_matrix().transposed())

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*bcam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = np.matmul(R_world2bcam.dot(-1), location)

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.matmul(R_bcam2cv, R_world2bcam)
    T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)

    extr = np.concatenate((R_world2cv, T_world2cv), axis=1)
    return extr


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
