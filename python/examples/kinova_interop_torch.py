"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import numpy as np
import imageio
import math
import os

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch


gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="PyTorch tensor interop example",
                               custom_parameters=[
                                   {"name": "--headless", "action": "store_true", "help": ""}])

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# This determines whether physics tensors are on CPU or GPU
sim_params.use_gpu_pipeline = True
if not args.use_gpu_pipeline:
    print("Warning: Forcing GPU pipeline.")

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")


use_viewer = not args.headless
# create viewer
if use_viewer:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")
else:
    viewer = None

# Load kinova asset
asset_root = "../../assets"
kinova_asset_file = "urdf/kinova_gen3/urdf/GEN3_URDF_V12.urdf"
kinova_end_effector = "EndEffector_Link"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = False
asset_options.armature = 0.01

print("Loading asset '%s' from '%s'" % (kinova_asset_file, asset_root))
kinova_asset = gym.load_asset(sim, asset_root, kinova_asset_file, asset_options)

# create table asset
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)


# create box asset
box_size = 0.045
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)
box_pose = gymapi.Transform()

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# set up env grid
num_envs = 16
envs_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# define actor pose in its env
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

# define camera properties
cam_props = gymapi.CameraProperties()
cam_props.width = 256
cam_props.height = 256
cam_props.enable_tensors = True


# keep track of some var
envs = []
box_idxs = []
box_handles = []
kinova_handles = []
cams = []
sensors = []
cam_tensors = []
sensors_tensors = []

# create envs
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    # add actor
    kinova_handle = gym.create_actor(env, kinova_asset, pose, "kinova", i, 0) #Todo not sure about last arg
    kinova_handles.append(kinova_handle)
    
    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add box
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    box_pose.p.z = table_dims.z + 0.5 * box_size+2
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    box_handles.append(box_handle)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get global index of box in rigid body state tensor
    box_idx = gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    # add camera
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam_handle, env, gymapi.Vec3(1.5, 0, 1), gymapi.Vec3(0, 0, 0.1))
    cams.append(cam_handle)

    # obtain camera tensor
    cam_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)
    print("Got camera tensor with shape", cam_tensor.shape)

    # wrap camera tensor in a pytorch tensor
    torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
    cam_tensors.append(torch_cam_tensor)
    print("  Torch camera tensor device:", torch_cam_tensor.device)
    print("  Torch camera tensor shape:", torch_cam_tensor.shape)

    # add sensor
    sensor_handle = gym.create_camera_sensor(env, cam_props)
    camera_offset = gymapi.Vec3(0, 0.1, 0)
    camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 1), np.deg2rad(-90))
    kinova_bracklet_handle = gym.find_actor_rigid_body_handle(env, kinova_handle, kinova_end_effector)
    gym.attach_camera_to_body(sensor_handle, env, kinova_bracklet_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
    sensors.append(sensor_handle)

    #obtain sensor tensor
    sensor_tensor = gym.get_camera_image_gpu_tensor(sim, env, sensor_handle, gymapi.IMAGE_COLOR)
    print("Got sensor tensor with shape", sensor_tensor.shape)

    # wrap sensor tensor in a pytorch tensor
    torch_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor)
    sensors_tensors.append(torch_sensor_tensor)
    print("  Torch sensor tensor device:", torch_sensor_tensor.device)
    print("  Torch sensor tensor shape:", torch_sensor_tensor.shape)



# get joint limits and ranges for Kinova Gen3
kinova_dof_props = gym.get_actor_dof_properties(envs[0], kinova_handles[0])
kinova_lower_limits = kinova_dof_props['lower']
kinova_upper_limits = kinova_dof_props['upper']
kinova_ranges = kinova_upper_limits - kinova_lower_limits
kinova_mids = 0.5 * (kinova_upper_limits + kinova_lower_limits)
kinova_num_dofs = len(kinova_dof_props)

# override default stiffness and damping values
kinova_dof_props['stiffness'].fill(1000.0)
kinova_dof_props['damping'].fill(1000.0)

# Give a desired pose for first 2 robot joints to improve stability
kinova_dof_props["driveMode"][0:2] = gymapi.DOF_MODE_POS

for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], kinova_handles[i], kinova_dof_props)

# Set kinova pose so that each joint is in the middle of its actuation range
for i in range(num_envs):
    # return an array in size of the kinova DoF, including states of a Degree of Freedom in the Asset architecture, that is
    # DOF position, in radians if it’s a revolute DOF, or meters, if it’s a prismatic DOF
    # DOF velocity, in radians/s if it’s a revolute DOF, or m/s, if it’s a prismatic DOF
    kinova_dof_states = gym.get_actor_dof_states(envs[i], kinova_handles[i], gymapi.STATE_NONE)
    for j in range(kinova_num_dofs):
        kinova_dof_states['pos'][j] = kinova_mids[j]
    gym.set_actor_dof_states(envs[i], kinova_handles[i], kinova_dof_states, gymapi.STATE_POS)


# point camera at middle env
cam_pos = gymapi.Vec3(8, 2, 6)
cam_target = gymapi.Vec3(-8, 0, -6)
middle_env = envs[num_envs // 2 + envs_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# prepare tensor access
gym.prepare_sim(sim)

# get GPU physics state tensor
_dof_state_tensor = gym.acquire_dof_state_tensor(sim)
print("Gym state tensor shape:", _dof_state_tensor.shape)
print("Gym state tensor data @ 0x%x" % _dof_state_tensor.data_address)

# wrap physics state tensor in a pytorch tensor
dof_state_tensor = gymtorch.wrap_tensor(_dof_state_tensor)
print("Torch state tensor device:", dof_state_tensor.device)
print("Torch state tensor shape:", dof_state_tensor.shape)
print("Torch state tensor data @ 0x%x" % dof_state_tensor.data_ptr())
saved_dof_state_tensor = dof_state_tensor.clone()

# create some wrapper tensors for different slices
# num_bodies = root_tensor.shape[0]
# rb_positions = root_tensor[:, 0:3]
# rb_orientations = root_tensor[:, 3:7]
# rb_linvels = root_tensor[:, 7:10]
# rb_angvels = root_tensor[:, 10:13]





# create directory for saved images
img_dir = "interop_images"
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

frame_count = 0
next_fps_report = 2.0
t1 = 0
step = 0
# Time to wait in seconds before moving robot
next_kinova_update_time = 2.5

while viewer is None or not gym.query_viewer_has_closed(viewer):
   
    frame_no = gym.get_frame_count(sim)

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh state data in the tensor
    gym.refresh_rigid_body_state_tensor(sim)

    gym.step_graphics(sim)

    # render sensors and refresh camera tensors
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    # write out state and sensors periodically during the first little while
    if frame_no < 60 and frame_no % 10 == 0:

        print("========= Frame %d ==========" % frame_no)

        #print the state tensors
        # print("RB states:")
        # print(root_tensor.cpu().detach().numpy().shape)
        # print("RB positions:")
        # print(rb_positions.shape)
        # print("RB orientations:")
        # print(rb_orientations.shape)
        # print("RB linear velocities:")
        # print(rb_linvels.shape)
        # print("RB angular velocities:")
        # print(rb_angvels.shape)

        for i in range(num_envs):
            # write tensor to image
            fname = os.path.join(img_dir, "cam-%04d-%04d.png" % (frame_no, i))
            cam_img = cam_tensors[i].cpu().numpy()
            imageio.imwrite(fname, cam_img)

            fname = os.path.join(img_dir, "sensor-%04d-%04d.png" % (frame_no, i))
            sensor_img = sensors_tensors[i].cpu().numpy()
            imageio.imwrite(fname, sensor_img)

    t = gym.get_elapsed_time(sim)
    if t >= next_fps_report:
        t2 = gym.get_elapsed_time(sim)
        fps = frame_count / (t2 - t1)
        print("FPS %.1f (%.1f)" % (fps, fps * num_envs))
        frame_count = 0
        t1 = gym.get_elapsed_time(sim)
        next_fps_report = t1 + 2.0

    gym.end_access_image_tensors(sim)

    if viewer is not None:
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    frame_count += 1
    step += 1
   
    

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)